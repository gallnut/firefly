#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <unordered_map>

#include "firefly/engine.h"
#include "firefly/mm/allocator.h"
#include "firefly/mm/model_weight_pool.h"
#include "firefly/model/model.h"
#include "firefly/model_loader.h"
#include "firefly/model_registry.h"
#include "firefly/queue.h"
#include "firefly/tokenizer.h"
#include "grpc_adapter.h"

using namespace firefly;
using json = nlohmann::json;

#define CHECK_CUDA_LAST_ERROR(msg)                                                                             \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t _err = cudaGetLastError();                                                                 \
        if (_err != cudaSuccess)                                                                               \
        {                                                                                                      \
            std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(_err) << " (" << _err << ")\n"; \
        }                                                                                                      \
    } while (0)

model::ModelConfig load_config(const std::string& config_path, std::string& out_arch)
{
    std::ifstream f(config_path);
    if (!f.is_open())
    {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }
    json j = json::parse(f);

    model::ModelConfig config;
    config.hidden_size = j.value("hidden_size", 0);
    config.intermediate_size = j.value("intermediate_size", 0);
    config.num_hidden_layers = j.value("num_hidden_layers", 0);
    config.num_attention_heads = j.value("num_attention_heads", 0);
    config.num_key_value_heads = j.value("num_key_value_heads", config.num_attention_heads);
    config.head_dim = j.value("head_dim", config.hidden_size / config.num_attention_heads);
    config.vocab_size = j.value("vocab_size", 0);
    config.max_position_embeddings = j.value("max_position_embeddings", 32768);
    config.rms_norm_eps = j.value("rms_norm_eps", 1e-6);
    config.rope_theta = j.value("rope_theta", 10000.0f);

    if (j.contains("architectures") && j["architectures"].is_array() && !j["architectures"].empty())
    {
        out_arch = j["architectures"][0].get<std::string>();
    }
    else
    {
        out_arch = "Qwen2ForCausalLM";
    }

    return config;
}

// Global Map for routing ResultQueue to individual Sessions
// Moved inside main to avoid global variables

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
        int         max_prefill_chunk_size = 256;

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
        std::string arch;
        auto        config = load_config(config_path, arch);
        std::cout << "Detected architecture: " << arch << std::endl;

        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        mm::ModelWeightPool<Device::CUDA> weight_pool;
        ModelLoader::Options              options;
        options.validate_checksums = false;
        options.alignment = 256;

        auto result = ModelLoader::load_safetensors(weights_path, weight_pool, options);
        if (!result)
        {
            std::cerr << "Failed to load model: " << result.error().description() << std::endl;
            return 1;
        }
        auto weights_map = std::move(result.value());

        std::unique_ptr<model::Model> base_model = model::ModelRegistry::get().create(arch, config);
        base_model->load_weights(weights_map);
        std::cout << "Model loaded successfully." << std::endl;
        CHECK_CUDA_LAST_ERROR("After model load");

        Tokenizer   tokenizer;
        std::string tokenizer_path = model_dir + "/tokenizer.json";
        if (!tokenizer.load(tokenizer_path))
        {
            std::cerr << "Failed to load tokenizer from " << tokenizer_path << std::endl;
            return 1;
        }

        // Initialize and start global Engine Queue
        ResultQueue global_result_queue;

        // Initialize and start Engine
        Engine engine(base_model.get(), config, tokenizer, &global_result_queue, max_prefill_chunk_size);
        engine.start();

        std::mutex                                                sessions_mtx;
        std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
        std::atomic<uint64_t>                                     global_request_id_counter{0};

        // Spawn central Dispatcher thread
        std::atomic<bool> dispatcher_running{true};
        std::thread       dispatcher_thread(
            [&global_result_queue, &dispatcher_running, &sessions_mtx, &sessions_]()
            {
                while (dispatcher_running)
                {
                    auto item = global_result_queue.pop();
                    if (item.req_id.empty() && item.is_finished)
                    {
                        // Stop signal
                        break;
                    }

                    std::shared_ptr<Session> session_ptr;
                    {
                        std::lock_guard<std::mutex> lock(sessions_mtx);
                        auto                        it = sessions_.find(item.req_id);
                        if (it != sessions_.end())
                        {
                            session_ptr = it->second;
                        }
                    }

                    if (session_ptr)
                    {
                        session_ptr->push(item.text, item.is_finished);
                        if (item.is_finished)
                        {
                            std::lock_guard<std::mutex> lock(sessions_mtx);
                            sessions_.erase(item.req_id);
                        }
                    }
                }
            });

        auto generate_handler = [&engine, &config, &tokenizer, &sessions_mtx, &sessions_, &global_request_id_counter](
                                    const std::string& req_body, bool is_stream, std::shared_ptr<Session> session)
        {
            try
            {
                std::vector<int> input_ids;
                int              im_start_id = 151644;
                int              im_end_id = 151645;
                int              max_tokens = 512;
                std::string      request_model = "qwen3";

                if (!req_body.empty())
                {
                    auto j = json::parse(req_body);
                    if (j.contains("max_tokens")) max_tokens = j["max_tokens"].get<int>();
                    if (j.contains("model")) request_model = j["model"].get<std::string>();

                    if (j.contains("messages") && j["messages"].is_array())
                    {
                        for (const auto& msg : j["messages"])
                        {
                            std::string role = msg.value("role", "user");
                            std::string content = msg.value("content", "");

                            // <|im_start|>role\n
                            input_ids.push_back(im_start_id);
                            std::vector<int> role_ids = tokenizer.encode(role + "\n");
                            input_ids.insert(input_ids.end(), role_ids.begin(), role_ids.end());

                            // content<|im_end|>\n
                            std::vector<int> content_ids = tokenizer.encode(content);
                            input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());
                            input_ids.push_back(im_end_id);

                            std::vector<int> nl_id = tokenizer.encode("\n");
                            input_ids.insert(input_ids.end(), nl_id.begin(), nl_id.end());
                        }
                    }
                    else if (j.contains("prompt"))
                    {
                        input_ids.push_back(im_start_id);
                        std::vector<int> role_ids = tokenizer.encode("user\n");
                        input_ids.insert(input_ids.end(), role_ids.begin(), role_ids.end());

                        std::vector<int> content_ids = tokenizer.encode(j["prompt"].get<std::string>());
                        input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());

                        input_ids.push_back(im_end_id);
                        std::vector<int> nl_id = tokenizer.encode("\n");
                        input_ids.insert(input_ids.end(), nl_id.begin(), nl_id.end());
                    }
                }

                // Prompt generation cue
                input_ids.push_back(im_start_id);
                std::vector<int> asst_ids = tokenizer.encode("assistant\n");
                input_ids.insert(input_ids.end(), asst_ids.begin(), asst_ids.end());

                std::cout << "[Server] Request chat completions limit: " << max_tokens << std::endl;

                uint64_t    current_id = global_request_id_counter.fetch_add(1);
                std::string req_id = "req-" + std::to_string(current_id);

                // Register the session in the global map
                {
                    std::lock_guard<std::mutex> lock(sessions_mtx);
                    sessions_[req_id] = session;
                }

                // Push the async inference job to the Engine. It no longer blocks this thread!
                engine.async_generate(req_id, input_ids, max_tokens);

                // The parsing logic (CoT tag hiding, json building) previously done here inline
                // is now the responsibility of `grpc_adapter.cc`'s Consumer loop, reading from `session`.
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Server Context] Error: " << e.what() << std::endl;
                json response;
                response["error"] = {{"message", e.what()}, {"type", "server_error"}};
                session->push(response.dump(), true);
            }
        };

        server::GrpcAdapter grpc_adapter;
        grpc_adapter.start_server(50051, generate_handler);

        // Cleanup
        dispatcher_running = false;
        global_result_queue.push("", "", true);  // Wake up dispatcher
        if (dispatcher_thread.joinable()) dispatcher_thread.join();

        engine.stop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
