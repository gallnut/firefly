#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>
#include <string>

#include "firefly/execution/engine.h"
#include "firefly/execution/result_queue.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/device/allocator.h"
#include "firefly/model/weight_pool.h"
#include "firefly/model/config_loader.h"
#include "firefly/model/model.h"
#include "firefly/model/loader.h"
#include "firefly/model/registry.h"
#include "firefly/service/server.h"
#include "firefly/model/tokenizer.h"

using namespace firefly;

#define CHECK_CUDA_LAST_ERROR(msg)                                                                             \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t _err = cudaGetLastError();                                                                 \
        if (_err != cudaSuccess)                                                                               \
        {                                                                                                      \
            std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(_err) << " (" << _err << ")\n"; \
        }                                                                                                      \
    } while (0)

int main(int argc, char** argv)
{
    try
    {
        cudaSetDevice(0);
        int dev = -1;
        cudaGetDevice(&dev);
        std::cout << "Initializing Firefly Server..." << std::endl;
        std::cout << "  Device: " << dev << std::endl;

        DeviceAllocator<Device::CUDA>::init(0, UINT64_MAX);

        std::string model_dir = "qwen3";
        int         max_prefill_chunk_size = 4096;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--max-prefill-chunk-size" && i + 1 < argc)
            {
                max_prefill_chunk_size = std::stoi(argv[++i]);
            }
            else if (arg[0] != '-')
            {
                model_dir = arg;
            }
        }

        std::string config_path = model_dir + "/config.json";
        std::string weights_path = model_dir + "/model.safetensors";

        std::cout << "Loading config from " << config_path << "..." << std::endl;
        auto descriptor = model::load_model_descriptor(config_path);
        auto config = descriptor.config;
        const auto& arch = descriptor.architecture;
        std::cout << "Detected architecture: " << arch << std::endl;
        std::cout << "Runtime options:\n";
        std::cout << "  Max prefill chunk size: " << max_prefill_chunk_size << " tokens\n";
        std::cout << "  Attention backend: "
                  << firefly::kernels::attention_backend_name(firefly::kernels::get_attention_backend()) << "\n";
        size_t kv_bytes_per_token = static_cast<size_t>(config.num_hidden_layers) * config.num_key_value_heads *
                                    config.head_dim * 2 * dtype_size(config.dtype);
        std::cout << "  KV cache/token: " << kv_bytes_per_token / 1024.0 << " KiB\n";

        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        model::ModelWeightPool<Device::CUDA> weight_pool;
        model::ModelLoader::Options       options;
        options.validate_checksums = false;
        options.alignment = 256;

        auto result = model::ModelLoader::load_safetensors(weights_path, weight_pool, options);
        if (!result)
        {
            std::cerr << "Failed to load model: " << result.error().description() << std::endl;
            return 1;
        }
        auto weights_map = std::move(result.value());
        if (auto it = weights_map.find("model.embed_tokens.weight"); it != weights_map.end())
        {
            config.dtype = it->second.dtype();
        }

        std::unique_ptr<model::Model> base_model = model::ModelRegistry::get().create(arch, config);
        base_model->load_weights(weights_map);
        std::cout << "Model loaded successfully." << std::endl;
        CHECK_CUDA_LAST_ERROR("After model load");

        model::Tokenizer tokenizer;
        std::string tokenizer_path = model_dir + "/tokenizer.json";
        if (!tokenizer.load(tokenizer_path))
        {
            std::cerr << "Failed to load tokenizer from " << tokenizer_path << std::endl;
            return 1;
        }

        // Initialize and start global Engine Queue
        execution::ResultQueue global_result_queue;

        // Initialize and start Engine
        execution::Engine engine(base_model.get(), config, tokenizer, &global_result_queue, max_prefill_chunk_size);
        engine.start();

        service::Server server(engine, tokenizer, global_result_queue);
        server.run(50051);

        engine.stop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
