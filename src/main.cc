#include <cuda_runtime_api.h>

#include <memory>
#include <cstdlib>
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
#include "firefly/model/tokenizer.h"
#include "firefly/model/weight_pool.h"
#include "firefly/service/server.h"

using namespace firefly;

namespace
{
execution::KVCacheFormat kv_cache_format_from_environment()
{
    const char* environment = std::getenv("FIREFLY_KV_CACHE_DTYPE");
    if (environment == nullptr || std::string_view(environment).empty() || std::string_view(environment) == "model")
    {
        return execution::KVCacheFormat::Model;
    }
    if (std::string_view(environment) == "int8") return execution::KVCacheFormat::Int8;
    throw std::runtime_error("FIREFLY_KV_CACHE_DTYPE must be 'model' or 'int8'");
}

const char* kv_cache_format_name(execution::KVCacheFormat format)
{
    return format == execution::KVCacheFormat::Int8 ? "int8" : "model";
}
}  // namespace

#define CHECK_CUDA_LAST_ERROR(operation)                                                              \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t error = cudaGetLastError();                                                         \
        if (error != cudaSuccess)                                                                       \
        {                                                                                              \
            FIREFLY_LOG_ERROR("cuda", "CUDA error operation={} error={} code={}", operation,          \
                               cudaGetErrorString(error), static_cast<int>(error));                      \
        }                                                                                              \
    } while (0)

int main(int argc, char** argv)
{
    try
    {
        std::string model_dir = "qwen3";
        int         max_prefill_chunk_size = 16384;
        double      gpu_memory_utilization = 0.90;
        log::Options log_options = log::options_from_environment();
        if (const char* environment = std::getenv("FIREFLY_GPU_MEMORY_UTILIZATION"))
            gpu_memory_utilization = std::strtod(environment, nullptr);

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--max-prefill-chunk-size" && i + 1 < argc)
            {
                max_prefill_chunk_size = std::stoi(argv[++i]);
            }
            else if (arg == "--log-detail")
            {
                log_options.detailed = true;
            }
            else if (arg == "--log-level" && i + 1 < argc)
            {
                auto level = log::parse_level(argv[++i]);
                if (!level) throw std::invalid_argument("invalid --log-level value");
                log_options.level = *level;
            }
            else if (arg == "--log-color" && i + 1 < argc)
            {
                auto color = log::parse_color_mode(argv[++i]);
                if (!color) throw std::invalid_argument("invalid --log-color value");
                log_options.color = *color;
            }
            else if (arg[0] != '-')
            {
                model_dir = arg;
            }
        }
        log::configure(log_options);

        cudaSetDevice(0);
        int device = -1;
        cudaGetDevice(&device);
        FIREFLY_LOG_INFO("startup", "initializing Firefly server device={}", device);

        DeviceAllocator<Device::CUDA>::init(0, UINT64_MAX);

        std::string config_path = model_dir + "/config.json";
        FIREFLY_LOG_INFO("startup", "loading model config path={}", config_path);
        auto                     descriptor = model::load_model_descriptor(config_path);
        auto                     config = descriptor.config;
        const auto&              arch = descriptor.architecture;
        execution::EngineOptions engine_options{
            .max_prefill_chunk_size = max_prefill_chunk_size,
            .kv_cache_format = kv_cache_format_from_environment(),
            .gpu_memory_utilization = gpu_memory_utilization,
        };
        std::unique_ptr<model::Model> base_model = model::ModelRegistry::get().create(descriptor);
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

        auto result = model::ModelLoader::load_model(model_dir, weight_pool, options);
        if (!result)
        {
            FIREFLY_LOG_ERROR("startup", "model load failed error={}", result.error().description());
            return 1;
        }
        auto weights_map = std::move(result.value());
        if (auto it = weights_map.find("model.embed_tokens.weight"); it != weights_map.end())
        {
            config.dtype = it->second.dtype();
        }

        base_model->load_weights(weights_map);
        cudaDeviceSynchronize();
        cudaMemPool_t memory_pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&memory_pool, device) == cudaSuccess)
            cudaMemPoolTrimTo(memory_pool, 0);
        FIREFLY_LOG_INFO("startup", "model loaded architecture={}", arch);
        CHECK_CUDA_LAST_ERROR("After model load");

        model::Tokenizer tokenizer;
        std::string      tokenizer_path = model_dir + "/tokenizer.json";
        if (!tokenizer.load(tokenizer_path))
        {
            FIREFLY_LOG_ERROR("startup", "tokenizer load failed path={}", tokenizer_path);
            return 1;
        }

        execution::ResultQueue global_result_queue;

        execution::Engine engine(base_model.get(), config, tokenizer, &global_result_queue, engine_options);
        engine.start();

        service::Server server(engine, tokenizer, global_result_queue);
        server.run(50051);

        engine.stop();
    }
    catch (const std::exception& e)
    {
        FIREFLY_LOG_CRITICAL("startup", "unhandled exception error={}", e.what());
        return 1;
    }

    return 0;
}
