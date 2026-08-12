#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include "firefly/core/logging.h"
#include "firefly/device/allocator.h"
#include "firefly/execution/engine.h"
#include "firefly/execution/result_queue.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/model/config_loader.h"
#include "firefly/model/loader.h"
#include "firefly/model/model.h"
#include "firefly/model/registry.h"
#include "firefly/model/speculative.h"
#include "firefly/model/tokenizer.h"
#include "firefly/model/weight_pool.h"
#include "firefly/service/server.h"

using namespace firefly;

namespace
{
Result<execution::KVCacheFormat> kv_cache_format_from_environment()
{
    const char* environment = std::getenv("FIREFLY_KV_CACHE_DTYPE");
    if (environment == nullptr || std::string_view(environment).empty() || std::string_view(environment) == "model")
    {
        return execution::KVCacheFormat::Model;
    }
    if (std::string_view(environment) == "int8") return execution::KVCacheFormat::Int8;
    return unexpected(Error{ErrorCode::InvalidArgument,
                            "FIREFLY_KV_CACHE_DTYPE must be 'model' or 'int8'"});
}

const char* kv_cache_format_name(execution::KVCacheFormat format)
{
    return format == execution::KVCacheFormat::Int8 ? "int8" : "model";
}

Result<int> parse_positive_int(std::string_view text, std::string_view option)
{
    int value = 0;
    const auto [end, parse_error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parse_error != std::errc{} || end != text.data() + text.size() || value <= 0)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                std::string(option) + " must be a positive integer"});
    return value;
}

Result<double> parse_fraction(std::string_view text, std::string_view option)
{
    double value = 0.0;
    const auto [end, parse_error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parse_error != std::errc{} || end != text.data() + text.size() || value <= 0.0 || value > 1.0)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                std::string(option) + " must be a number in (0, 1]"});
    return value;
}

Status run_server(int argc, char** argv)
{
    std::string model_dir = "qwen3";
    std::string draft_model_dir;
    int         max_prefill_chunk_size = 16384;
    double      gpu_memory_utilization = 0.90;
    log::Options log_options = FIREFLY_TRY(log::options_from_environment());
    if (const char* environment = std::getenv("FIREFLY_GPU_MEMORY_UTILIZATION"))
        gpu_memory_utilization = FIREFLY_TRY(parse_fraction(environment, "FIREFLY_GPU_MEMORY_UTILIZATION"));

    for (int i = 1; i < argc; ++i)
    {
        std::string_view argument = argv[i];
        if (argument == "--max-prefill-chunk-size" && i + 1 < argc)
            max_prefill_chunk_size = FIREFLY_TRY(parse_positive_int(argv[++i], argument));
        else if (argument == "--draft-model" && i + 1 < argc)
            draft_model_dir = argv[++i];
        else if (argument == "--log-detail")
            log_options.detailed = true;
        else if (argument == "--log-level" && i + 1 < argc)
        {
            auto level = log::parse_level(argv[++i]);
            if (!level) return unexpected(Error{ErrorCode::InvalidArgument, "invalid --log-level value"});
            log_options.level = *level;
        }
        else if (argument == "--log-color" && i + 1 < argc)
        {
            auto color = log::parse_color_mode(argv[++i]);
            if (!color) return unexpected(Error{ErrorCode::InvalidArgument, "invalid --log-color value"});
            log_options.color = *color;
        }
        else if (!argument.empty() && argument.front() != '-')
            model_dir = argument;
        else
            return unexpected(Error{ErrorCode::InvalidArgument,
                                    "unknown or incomplete command-line option: " + std::string(argument)});
    }
    FIREFLY_TRY(log::configure(log_options));

    FIREFLY_TRY(device::check_cuda(cudaSetDevice(0), "select server CUDA device"));
    int device = -1;
    FIREFLY_TRY(device::check_cuda(cudaGetDevice(&device), "query server CUDA device"));
    FIREFLY_LOG_INFO("startup", "initializing Firefly server device={}", device);

    FIREFLY_TRY(DeviceAllocator<Device::CUDA>::init(0, UINT64_MAX));

        std::string config_path = model_dir + "/config.json";
        FIREFLY_LOG_INFO("startup", "loading model config path={}", config_path);
        auto                     descriptor = FIREFLY_TRY(model::load_model_descriptor(config_path));
        auto                     config = descriptor.config;
        const auto&              arch = descriptor.architecture;
        execution::EngineOptions engine_options{
            .max_prefill_chunk_size = max_prefill_chunk_size,
            .kv_cache_format = FIREFLY_TRY(kv_cache_format_from_environment()),
            .gpu_memory_utilization = gpu_memory_utilization,
        };
        std::unique_ptr<model::Model> base_model = FIREFLY_TRY(model::ModelRegistry::get().create(descriptor));
        const auto runtime_requirements = base_model->runtime_requirements();
        if (runtime_requirements.prefill_chunk_limit > 0)
            engine_options.max_prefill_chunk_size =
                std::min(engine_options.max_prefill_chunk_size, runtime_requirements.prefill_chunk_limit);
        const bool kv_cache_quantized = engine_options.kv_cache_format == execution::KVCacheFormat::Int8;
        size_t     kv_bytes_per_token = static_cast<size_t>(runtime_requirements.kv_cache_layer_count) *
                                        runtime_requirements.kv_cache_head_count * runtime_requirements.kv_cache_head_dim *
                                        2 * dtype_size(kv_cache_quantized ? DType::I8 : config.dtype);
        if (kv_cache_quantized)
        {
            kv_bytes_per_token += static_cast<size_t>(runtime_requirements.kv_cache_layer_count) *
                                  runtime_requirements.kv_cache_head_count * 2 * sizeof(float) / 16;
        }
        FIREFLY_LOG_INFO("startup",
                         "runtime configured architecture={} prefill_chunk_tokens={} attention_backend={} "
                         "kv_cache_dtype={} kv_cache_kib_per_token={:.2f}",
                         arch, engine_options.max_prefill_chunk_size,
                         kernels::attention_backend_name(kernels::get_attention_backend()),
                         kv_cache_format_name(engine_options.kv_cache_format), kv_bytes_per_token / 1024.0);

        FIREFLY_LOG_INFO("startup", "loading model weights directory={}", model_dir);
        model::ModelWeightPool<Device::CUDA> weight_pool;
        model::ModelLoader::Options          options;
        options.validate_checksums = false;
        options.alignment = 256;
        options.include_weight = [&base_model](std::string_view name) { return base_model->accepts_weight(name); };
        options.retain_weight = [&base_model](std::string_view name)
        { return base_model->retains_source_weight(name); };

        auto weights_map = FIREFLY_TRY_CONTEXT(model::ModelLoader::load_model(model_dir, weight_pool, options),
                                               "load target model weights");
        if (auto it = weights_map.find("model.embed_tokens.weight"); it != weights_map.end())
        {
            config.dtype = it->second.dtype();
        }

        FIREFLY_TRY_CONTEXT(base_model->load_weights(weights_map), "map target model weights");
        FIREFLY_TRY(device::check_cuda(cudaDeviceSynchronize(), "synchronize target model loading"));
        cudaMemPool_t memory_pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&memory_pool, device) == cudaSuccess)
            cudaMemPoolTrimTo(memory_pool, 0);
        FIREFLY_LOG_INFO("startup", "model loaded architecture={}", arch);

        std::unique_ptr<model::Model> draft_model;
        model::ModelConfig          draft_config{};
        std::unique_ptr<model::ModelWeightPool<Device::CUDA>> draft_weight_pool;
        if (draft_model_dir.empty())
        {
            const char* environment = std::getenv("FIREFLY_SPECULATIVE_MODEL");
            if (environment == nullptr) environment = std::getenv("FIREFLY_DSPARK_MODEL");
            if (environment != nullptr) draft_model_dir = environment;
        }
        if (!draft_model_dir.empty())
        {
            std::string draft_config_path = draft_model_dir + "/config.json";
            std::ifstream draft_config_file(draft_config_path);
            if (!draft_config_file.good()) draft_config_path = draft_model_dir + "/config_v03.json";
            FIREFLY_LOG_INFO("startup", "loading speculative draft config path={}", draft_config_path);
            auto draft_descriptor = FIREFLY_TRY(model::load_model_descriptor(draft_config_path));
            draft_config = draft_descriptor.config;
            draft_model = FIREFLY_TRY(model::ModelRegistry::get().create(draft_descriptor));
            draft_weight_pool = std::make_unique<model::ModelWeightPool<Device::CUDA>>();
            model::ModelLoader::Options draft_options;
            draft_options.validate_checksums = false;
            draft_options.alignment = 256;
            draft_options.include_weight = [&draft_model](std::string_view name)
            { return draft_model->accepts_weight(name); };
            draft_options.retain_weight = [&draft_model](std::string_view name)
            { return draft_model->retains_source_weight(name); };
            auto draft_weights = FIREFLY_TRY_CONTEXT(
                model::ModelLoader::load_model(draft_model_dir, *draft_weight_pool, draft_options),
                "load speculative draft weights");
            FIREFLY_TRY_CONTEXT(draft_model->load_weights(draft_weights), "map speculative draft weights");
            FIREFLY_TRY(device::check_cuda(cudaDeviceSynchronize(), "synchronize speculative draft loading"));
            FIREFLY_LOG_INFO("startup", "speculative draft loaded architecture={} tokens={}",
                             draft_descriptor.architecture, draft_config.vocab_size);
            auto* proposer = dynamic_cast<model::SpeculativeProposer*>(draft_model.get());
            if (proposer == nullptr)
                return unexpected(Error{ErrorCode::Model,
                                        "speculative draft model does not implement SpeculativeProposer"});
            engine_options.speculative.proposer = proposer;
            const char* environment = std::getenv("FIREFLY_SPECULATIVE_DRAFT_TOKENS");
            if (environment == nullptr) environment = std::getenv("FIREFLY_DSPARK_DRAFT_TOKENS");
            if (environment != nullptr)
                engine_options.speculative.max_draft_tokens =
                    FIREFLY_TRY(parse_positive_int(environment, "FIREFLY_SPECULATIVE_DRAFT_TOKENS"));
            const char* confidence = std::getenv("FIREFLY_SPECULATIVE_CONFIDENCE_THRESHOLD");
            if (confidence == nullptr) confidence = std::getenv("FIREFLY_DSPARK_CONFIDENCE_THRESHOLD");
            if (confidence != nullptr)
                engine_options.speculative.confidence_threshold = static_cast<float>(
                    FIREFLY_TRY(parse_fraction(confidence, "FIREFLY_SPECULATIVE_CONFIDENCE_THRESHOLD")));
        }

        model::Tokenizer tokenizer;
        std::string      tokenizer_path = model_dir + "/tokenizer.json";
        FIREFLY_TRY_CONTEXT(tokenizer.load(tokenizer_path), "load target tokenizer");

        execution::ResultQueue global_result_queue;

        std::unique_ptr<execution::Engine> engine = FIREFLY_TRY(execution::Engine::create(
            base_model.get(), config, tokenizer, &global_result_queue, engine_options));
        FIREFLY_TRY(engine->start());

        service::Server server(*engine, tokenizer, global_result_queue);
        FIREFLY_TRY(server.run(50051));

        engine->stop();
    return {};
}
}  // namespace

int main(int argc, char** argv)
{
    Status status = run_server(argc, argv);
    if (status) return 0;
    FIREFLY_LOG_CRITICAL("startup", "{}", status.error().describe());
    return 1;
}
