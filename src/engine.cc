#include "firefly/engine.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstring>

#include "firefly/kernels.h"
#include "firefly/mm/allocator.h"

namespace firefly
{

Engine::Engine(model::Model* model, const model::ModelConfig& config, const Tokenizer& tokenizer,
               ResultQueue* result_queue, int max_prefill_chunk_size)
    : model_(model),
      config_(config),
      tokenizer_(tokenizer),
      result_queue_(result_queue),
      scheduler_(),
      max_prefill_chunk_size_(max_prefill_chunk_size)
{
}

Engine::~Engine() { stop(); }

void Engine::async_generate(const std::string& id, const std::vector<int>& input_ids, int max_tokens)
{
    auto req = std::make_shared<GenerationRequest>(id, input_ids, max_tokens);
    scheduler_.add_request(req);
}

void Engine::start()
{
    if (running_) return;

    // --- Memory Profiling Phase ---
    cudaSetDevice(0);
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);

    // Dummy Forward Pass
    int    max_batched_tokens = std::max(2048, max_prefill_chunk_size_);
    Tensor dummy_input({1, max_batched_tokens}, DType::I32, Device::CUDA);
    cudaMemset(dummy_input.data(), 0, max_batched_tokens * sizeof(int));

    // Create dummy cache and block table for the profile run
    int                 dummy_num_blocks = (max_batched_tokens + 15) / 16;
    std::vector<Tensor> dummy_k, dummy_v;
    for (int i = 0; i < config_.num_hidden_layers; ++i)
    {
        dummy_k.emplace_back(
            Tensor({(long)dummy_num_blocks, 16, (long)config_.num_key_value_heads, (long)config_.head_dim}, DType::F16,
                   Device::CUDA));
        dummy_v.emplace_back(
            Tensor({(long)dummy_num_blocks, 16, (long)config_.num_key_value_heads, (long)config_.head_dim}, DType::F16,
                   Device::CUDA));
    }
    std::vector<int> h_dummy_table(dummy_num_blocks);
    for (int i = 0; i < dummy_num_blocks; ++i) h_dummy_table[i] = i;

    int* dummy_block_table;
    cudaMalloc(&dummy_block_table, dummy_num_blocks * sizeof(int));
    cudaMemcpy(dummy_block_table, h_dummy_table.data(), dummy_num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // Create a dummy context lengths tensor
    int    h_dummy_context = 0;
    Tensor dummy_context_lens({1}, DType::I32, Device::CUDA);
    cudaMemcpy(dummy_context_lens.data(), &h_dummy_context, sizeof(int), cudaMemcpyHostToDevice);

    // Run dummy forward
    model_->forward(dummy_input, dummy_context_lens, dummy_k, dummy_v, dummy_block_table, dummy_num_blocks);
    cudaDeviceSynchronize();

    cudaError_t dummy_err = cudaGetLastError();
    if (dummy_err != cudaSuccess)
    {
        std::cerr << "CRITICAL CAUSE: Dummy Forward Failed! " << cudaGetErrorString(dummy_err) << std::endl;
        std::abort();
    }

    // Measure peak memory overhead after activations are allocated
    size_t peak_free_byte, peak_total_byte;
    cudaMemGetInfo(&peak_free_byte, &peak_total_byte);

    // Cleanup dummy data
    cudaFree(dummy_block_table);

    // Calculate available KV cache memory
    double gpu_memory_utilization = 0.90;
    size_t usable_memory = static_cast<size_t>(peak_free_byte * gpu_memory_utilization);

    size_t block_bytes = config_.num_hidden_layers * 16 * config_.num_key_value_heads * config_.head_dim * 2 * 2;
    int    max_num_blocks = usable_memory / block_bytes;
    max_context_blocks_ = max_num_blocks;

    std::cout << "Memory Profiling:\n";
    std::cout << "  Free VRAM before profile: " << free_byte / (1024 * 1024) << " MB\n";
    std::cout << "  Free VRAM after dummy pass: " << peak_free_byte / (1024 * 1024) << " MB\n";
    std::cout << "  Max KV Blocks allocated: " << max_num_blocks << " ("
              << (max_num_blocks * block_bytes) / (1024 * 1024) << " MB)\n";

    // --- Allocate True KV Caches ---
    for (int i = 0; i < config_.num_hidden_layers; ++i)
    {
        k_caches_.emplace_back(
            Tensor({(long)max_num_blocks, 16, (long)config_.num_key_value_heads, (long)config_.head_dim}, DType::F16,
                   Device::CUDA));
        v_caches_.emplace_back(
            Tensor({(long)max_num_blocks, 16, (long)config_.num_key_value_heads, (long)config_.head_dim}, DType::F16,
                   Device::CUDA));
    }

    scheduler_.init(max_num_blocks, supported_batch_sizes_.back(), max_prefill_chunk_size_);

    // --- Graph Capture ---
    cudaStream_t capture_stream;
    cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);

    for (int bs : supported_batch_sizes_)
    {
        GraphData gd;
        gd.input_ids = Tensor({(long)bs, 1}, DType::I32, Device::CUDA);
        gd.context_lens = Tensor({(long)bs}, DType::I32, Device::CUDA);
        gd.block_table = Tensor({(long)(bs * max_context_blocks_)}, DType::I32, Device::CUDA);
        gd.next_tokens = Tensor({(long)bs}, DType::I32, Device::CUDA);

        // Pre-warm memory synchronously to avoid cross-stream page faults
        cudaMemset(gd.input_ids.data(), 0, bs * sizeof(int));
        cudaMemset(gd.context_lens.data(), 0, bs * sizeof(int));
        cudaMemset(gd.block_table.data(), 0, bs * max_context_blocks_ * sizeof(int));

        // Capture graph
        DeviceAllocator<Device::CUDA>::set_default_stream(capture_stream);
        firefly::kernels::set_default_stream(capture_stream);
        auto result = gd.graph.capture(capture_stream,
                                       [&]()
                                       {
                                           int*   d_block_table = (int*)gd.block_table.data();
                                           Tensor logits =
                                               model_->forward(gd.input_ids, gd.context_lens, k_caches_, v_caches_,
                                                               d_block_table, max_context_blocks_);

                                           kernels::argmax(logits, gd.next_tokens);
                                       });
        DeviceAllocator<Device::CUDA>::set_default_stream(nullptr);
        firefly::kernels::set_default_stream(nullptr);

        cudaMallocHost((void**)&gd.h_input_ids, bs * sizeof(int));
        cudaMallocHost((void**)&gd.h_context_lens, bs * sizeof(int));
        cudaMallocHost((void**)&gd.h_block_table, bs * max_context_blocks_ * sizeof(int));
        cudaMallocHost((void**)&gd.h_next_tokens, bs * sizeof(int));

        if (!result)
        {
            std::cerr << "Failed to capture graph for batch size " << bs << ": " << result.error().description()
                      << std::endl;
        }
        else
        {
            std::cout << "Successfully captured execution graph for decode batch size " << bs << std::endl;
            // Pre-warm the graph to pay the 5s+ driver instantiation/compilation cost immediately at startup
            // instead of stalling the first generation request at runtime.
            [[maybe_unused]] auto _ = gd.graph.launch(capture_stream);
            cudaStreamSynchronize(capture_stream);
            dec_graphs_[bs] = std::move(gd);
        }
    }

    cudaStreamDestroy(capture_stream);

    running_ = true;
    background_thread_ = std::thread(&Engine::loop, this);
}

void Engine::stop()
{
    running_ = false;
    if (background_thread_.joinable())
    {
        background_thread_.join();
    }
    for (auto& [bs, gd] : dec_graphs_)
    {
        if (gd.h_input_ids) cudaFreeHost(gd.h_input_ids);
        if (gd.h_context_lens) cudaFreeHost(gd.h_context_lens);
        if (gd.h_block_table) cudaFreeHost(gd.h_block_table);
        if (gd.h_next_tokens) cudaFreeHost(gd.h_next_tokens);
    }
    dec_graphs_.clear();
}

void Engine::loop()
{
    cudaSetDevice(0);
    cudaStream_t decode_stream;
    cudaStreamCreateWithFlags(&decode_stream, cudaStreamNonBlocking);

    while (running_)
    {
        if (!scheduler_.has_unfinished_requests())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        nvtxRangePush("Engine_Schedule");
        auto batch = scheduler_.step();
        nvtxRangePop();
        auto t1 = std::chrono::high_resolution_clock::now();
        if (batch.requests.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        std::vector<std::shared_ptr<GenerationRequest>> prefill_requests;
        std::vector<std::shared_ptr<GenerationRequest>> decode_requests;

        for (auto req : batch.requests)
        {
            if (req->generated_tokens.empty())
            {
                prefill_requests.push_back(req);
            }
            else
            {
                decode_requests.push_back(req);
            }
        }

        // --- PREFILL (Chunked Processing) ---
        if (!prefill_requests.empty())
        {
            nvtxRangePush("Engine_Prefill");
            for (auto req : prefill_requests)
            {
                int unmatched_tokens = req->prompt_tokens.size() - req->context_len;
                int seq_len = std::min(unmatched_tokens, max_prefill_chunk_size_);
                int context_len = req->context_len;

                std::vector<int> current_input_ids(req->prompt_tokens.begin() + req->context_len,
                                                   req->prompt_tokens.begin() + req->context_len + seq_len);

                Tensor d_input({1, seq_len}, DType::I32, Device::CUDA);
                cudaMemcpy(d_input.data(), current_input_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

                Tensor d_context_lens({1}, DType::I32, Device::CUDA);
                cudaMemcpy(d_context_lens.data(), &context_len, sizeof(int), cudaMemcpyHostToDevice);

                int    num_blocks = req->block_table.size();
                Tensor d_block_table_tensor({(long)num_blocks}, DType::I32, Device::CUDA);
                int*   d_block_table = (int*)d_block_table_tensor.data();
                cudaMemcpy(d_block_table, req->block_table.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);

                Tensor logits =
                    model_->forward(d_input, d_context_lens, k_caches_, v_caches_, d_block_table, num_blocks);

                req->context_len += seq_len;

                // Only generate a token if we have reached the end of the prompt
                if (req->context_len == req->prompt_tokens.size())
                {
                    Tensor next_token({1}, DType::I32, Device::CUDA);
                    kernels::argmax(logits, next_token);
                    int next_token_id;
                    cudaMemcpy(&next_token_id, next_token.data(), sizeof(int), cudaMemcpyDeviceToHost);

                    req->generated_tokens.push_back(next_token_id);
                }
            }
            nvtxRangePop();  // Engine_Prefill
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        // --- DECODE (True Continuous Batching with CUDA Graph) ---
        if (!decode_requests.empty())
        {
            nvtxRangePush("Engine_Decode");
            int current_batch_size = decode_requests.size();

            // Find the appropriate graph batch size
            int target_bs = supported_batch_sizes_.back();
            for (int bs : supported_batch_sizes_)
            {
                if (bs >= current_batch_size)
                {
                    target_bs = bs;
                    break;
                }
            }

            auto& gd = dec_graphs_[target_bs];

            std::fill(gd.h_input_ids, gd.h_input_ids + target_bs, 0);
            std::fill(gd.h_context_lens, gd.h_context_lens + target_bs, 0);
            std::fill(gd.h_block_table, gd.h_block_table + target_bs * max_context_blocks_, -1);

            for (int i = 0; i < current_batch_size; ++i)
            {
                auto req = decode_requests[i];
                gd.h_input_ids[i] = req->generated_tokens.back();
                gd.h_context_lens[i] = req->context_len;

                for (size_t b = 0; b < req->block_table.size(); ++b)
                {
                    gd.h_block_table[i * max_context_blocks_ + b] = req->block_table[b];
                }
            }

            // Copy to pre-allocated static buffers (using explicit stream)
            nvtxRangePush("Engine_Decode_Graph_Copy");
            auto d0 = std::chrono::high_resolution_clock::now();
            cudaMemcpyAsync(gd.input_ids.data(), gd.h_input_ids, target_bs * sizeof(int), cudaMemcpyHostToDevice,
                            decode_stream);
            cudaMemcpyAsync(gd.context_lens.data(), gd.h_context_lens, target_bs * sizeof(int), cudaMemcpyHostToDevice,
                            decode_stream);
            cudaMemcpyAsync(gd.block_table.data(), gd.h_block_table, target_bs * max_context_blocks_ * sizeof(int),
                            cudaMemcpyHostToDevice, decode_stream);

            auto d1 = std::chrono::high_resolution_clock::now();
            nvtxRangePop();  // Engine_Decode_Graph_Copy

            // Launch the static graph
            nvtxRangePush("Engine_Decode_Graph_Launch");
            auto result = gd.graph.launch(decode_stream);
            if (!result)
            {
                std::cerr << "Graph Launch failed: " << result.error().description() << std::endl;
            }
            auto d2 = std::chrono::high_resolution_clock::now();

            // Extract tokens back to host
            cudaMemcpyAsync(gd.h_next_tokens, gd.next_tokens.data(), target_bs * sizeof(int), cudaMemcpyDeviceToHost,
                            decode_stream);
            auto d3 = std::chrono::high_resolution_clock::now();

            // Synchronize before reading host elements
            cudaStreamSynchronize(decode_stream);
            nvtxRangePop();  // Engine_Decode_Graph_Launch
            auto d4 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < current_batch_size; ++i)
            {
                auto req = decode_requests[i];
                req->generated_tokens.push_back(gd.h_next_tokens[i]);
                req->context_len += 1;
            }
            auto d5 = std::chrono::high_resolution_clock::now();

            static double td_h2d = 0, td_launch = 0, td_d2h = 0, td_sync = 0, td_upd = 0;
            td_h2d += std::chrono::duration<double, std::milli>(d1 - d0).count();
            td_launch += std::chrono::duration<double, std::milli>(d2 - d1).count();
            td_d2h += std::chrono::duration<double, std::milli>(d3 - d2).count();
            td_sync += std::chrono::duration<double, std::milli>(d4 - d3).count();
            td_upd += std::chrono::duration<double, std::milli>(d5 - d4).count();
            static int dec_steps = 0;
            dec_steps++;
            if (dec_steps % 50 == 0)
            {
                std::cout << "DECODE [50]: H2D:" << td_h2d << "ms, Lch:" << td_launch << "ms, D2H:" << td_d2h
                          << "ms, Sync:" << td_sync << "ms, Upd:" << td_upd << "ms\n";
                td_h2d = td_launch = td_d2h = td_sync = td_upd = 0;
            }
            nvtxRangePop();  // Engine_Decode
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        // --- Process Results & Check EOS ---
        nvtxRangePush("Engine_Process_Results");
        for (auto req : batch.requests)
        {
            int         next_token_id = req->generated_tokens.back();
            std::string text = tokenizer_.decode(next_token_id);

            bool is_finished = false;
            if (next_token_id == 151645 || next_token_id == 151643 ||
                req->generated_tokens.size() >= (size_t)req->max_tokens)
            {
                is_finished = true;
            }

            req->utf8_buffer += text;

            size_t valid_len = 0;
            size_t len = req->utf8_buffer.length();
            for (size_t k = 1; k <= std::min<size_t>(4, len); ++k)
            {
                unsigned char c = req->utf8_buffer[len - k];
                if ((c & 0x80) == 0)
                {
                    valid_len = len;
                    break;
                }
                if ((c & 0xC0) == 0x80) continue;
                if ((c & 0xE0) == 0xC0)
                {
                    valid_len = (k >= 2) ? len : (len - k);
                    break;
                }
                if ((c & 0xF0) == 0xE0)
                {
                    valid_len = (k >= 3) ? len : (len - k);
                    break;
                }
                if ((c & 0xF8) == 0xF0)
                {
                    valid_len = (k >= 4) ? len : (len - k);
                    break;
                }
                valid_len = len;
                break;
            }
            if (valid_len == 0 && len >= 4) valid_len = len;
            if (is_finished) valid_len = len;

            std::string valid_text = req->utf8_buffer.substr(0, valid_len);
            req->utf8_buffer.erase(0, valid_len);

            if (result_queue_ && (!valid_text.empty() || is_finished))
            {
                result_queue_->push(req->id, valid_text, is_finished);
            }

            if (is_finished)
            {
                scheduler_.finish_request(req);
            }
        }
        nvtxRangePop();  // Engine_Process_Results

        auto t4 = std::chrono::high_resolution_clock::now();

        if (!batch.requests.empty())
        {
            static int    steps = 0;
            static double d_sched = 0, d_pref = 0, d_dec = 0, d_proc = 0;
            d_sched += std::chrono::duration<double, std::milli>(t1 - t0).count();
            d_pref += std::chrono::duration<double, std::milli>(t2 - t1).count();
            d_dec += std::chrono::duration<double, std::milli>(t3 - t2).count();
            d_proc += std::chrono::duration<double, std::milli>(t4 - t3).count();
            steps++;
            if (steps % 50 == 0)
            {
                std::cout << "Perf [50 steps]: Sched: " << d_sched << "ms, Prefill: " << d_pref
                          << "ms, Decode: " << d_dec << "ms, Process: " << d_proc << "ms\n";
                d_sched = d_pref = d_dec = d_proc = 0;
            }
        }
    }

    cudaStreamDestroy(decode_stream);
}

}  // namespace firefly
