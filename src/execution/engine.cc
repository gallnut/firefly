#include "firefly/execution/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/sampling/argmax.h"
#include "firefly/device/allocator.h"
#include "firefly/device/stream.h"
#include "firefly/execution/trace.h"

namespace firefly::execution
{
namespace
{
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
bool profile_prefill_enabled()
{
    static bool enabled = []()
    {
        const char* value = std::getenv("FIREFLY_PROFILE_PREFILL");
        return value && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool profile_decode_enabled()
{
    static bool enabled = []()
    {
        const char* value = std::getenv("FIREFLY_PROFILE_DECODE");
        return value && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

struct PrefillProfileEvents
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~PrefillProfileEvents()
    {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

PrefillProfileEvents& prefill_profile_events()
{
    static thread_local PrefillProfileEvents events;
    if (!events.start)
    {
        cudaEventCreate(&events.start);
        cudaEventCreate(&events.stop);
    }
    return events;
}

struct DecodeProfileEvents
{
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t fwd_stop = nullptr;
    cudaEvent_t argmax_stop = nullptr;
    cudaEvent_t d2h_stop = nullptr;

    ~DecodeProfileEvents()
    {
        if (h2d_start) cudaEventDestroy(h2d_start);
        if (h2d_stop) cudaEventDestroy(h2d_stop);
        if (fwd_stop) cudaEventDestroy(fwd_stop);
        if (argmax_stop) cudaEventDestroy(argmax_stop);
        if (d2h_stop) cudaEventDestroy(d2h_stop);
    }
};

DecodeProfileEvents& decode_profile_events()
{
    static thread_local DecodeProfileEvents events;
    if (!events.h2d_start)
    {
        cudaEventCreate(&events.h2d_start);
        cudaEventCreate(&events.h2d_stop);
        cudaEventCreate(&events.fwd_stop);
        cudaEventCreate(&events.argmax_stop);
        cudaEventCreate(&events.d2h_stop);
    }
    return events;
}
#endif

struct DecodeHostScratch
{
    std::vector<int> input_ids;
    std::vector<int> context_lens;
    std::vector<int> block_table;
    std::vector<int> next_tokens;
};

DecodeHostScratch& decode_host_scratch()
{
    static thread_local DecodeHostScratch scratch;
    return scratch;
}
}  // namespace

Engine::Engine(model::Model* model, const model::ModelConfig& config, const model::Tokenizer& tokenizer,
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

void Engine::async_generate(const std::string& id, const std::vector<int>& input_ids, int max_tokens,
                            std::shared_ptr<std::atomic_bool> cancel_flag)
{
    auto sequence = std::make_shared<scheduler::Sequence>(id, input_ids, max_tokens, std::move(cancel_flag));
    scheduler_.add_sequence(sequence);
}

void Engine::start()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
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
            Tensor({(long)dummy_num_blocks, 16, (long)config_.num_key_value_heads, (long)config_.head_dim},
                   config_.dtype, Device::CUDA));
        dummy_v.emplace_back(
            Tensor({(long)dummy_num_blocks, 16, (long)config_.num_key_value_heads, (long)config_.head_dim},
                   config_.dtype, Device::CUDA));
    }
    std::vector<int> h_dummy_table(dummy_num_blocks);
    for (int i = 0; i < dummy_num_blocks; ++i) h_dummy_table[i] = i;

    int* dummy_block_table;
    cudaMalloc(&dummy_block_table, dummy_num_blocks * sizeof(int));
    cudaMemcpy(dummy_block_table, h_dummy_table.data(), dummy_num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    Tensor dummy_context_lens({1}, DType::I32, Device::CUDA);
    cudaMemset(dummy_context_lens.data(), 0, sizeof(int));

    // Run dummy forward
    model::ModelInput dummy_model_input{
        dummy_input, dummy_context_lens, {dummy_k, dummy_v, dummy_block_table, dummy_num_blocks}};
    model_->forward(dummy_model_input, {.compute_logits = false});
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

    size_t block_bytes =
        config_.num_hidden_layers * 16 * config_.num_key_value_heads * config_.head_dim * 2 * dtype_size(config_.dtype);
    int max_num_blocks = usable_memory / block_bytes;
    decode_scratch_blocks_ = supported_batch_sizes_.back();
    if (max_num_blocks <= decode_scratch_blocks_)
    {
        throw std::runtime_error("Not enough KV cache blocks for decode graph scratch rows");
    }
    max_num_blocks -= decode_scratch_blocks_;
    decode_scratch_first_block_ = max_num_blocks;
    max_context_blocks_ = max_num_blocks;

    std::cout << "Memory Profiling:\n";
    std::cout << "  Free VRAM before profile: " << free_byte / (1024 * 1024) << " MB\n";
    std::cout << "  Free VRAM after dummy pass: " << peak_free_byte / (1024 * 1024) << " MB\n";
    std::cout << "  Max KV Blocks allocated: " << max_num_blocks << " ("
              << (max_num_blocks * block_bytes) / (1024 * 1024) << " MB)\n";
    std::cout << "  Max KV tokens allocated: " << max_num_blocks * 16 << "\n";
    std::cout << "  Approx 8K request capacity: " << (max_num_blocks * 16) / 8192 << "\n";
    std::cout << "  Approx 16K request capacity: " << (max_num_blocks * 16) / 16384 << "\n";
    std::cout << "  Decode scratch KV blocks: " << decode_scratch_blocks_ << "\n";

    // --- Allocate True KV Caches ---
    for (int i = 0; i < config_.num_hidden_layers; ++i)
    {
        k_caches_.emplace_back(
            Tensor({(long)(max_num_blocks + decode_scratch_blocks_), 16, (long)config_.num_key_value_heads,
                    (long)config_.head_dim},
                   config_.dtype, Device::CUDA));
        v_caches_.emplace_back(
            Tensor({(long)(max_num_blocks + decode_scratch_blocks_), 16, (long)config_.num_key_value_heads,
                    (long)config_.head_dim},
                   config_.dtype, Device::CUDA));
    }

    scheduler_.init(max_num_blocks, supported_batch_sizes_.back(), max_prefill_chunk_size_);

    // --- Graph Capture ---
    auto capture_stream_result = device::Stream::create();
    if (!capture_stream_result) throw std::runtime_error(capture_stream_result.error().description());
    device::Stream capture_stream_owner = std::move(capture_stream_result.value());
    cudaStream_t capture_stream = capture_stream_owner.get();
    device::Context capture_context = capture_stream_owner.context();

    auto capture_decode_graph = [&](int bs) -> std::optional<GraphData>
    {
        GraphData gd;
        gd.input_ids = Tensor({(long)bs, 1}, DType::I32, Device::CUDA, capture_context);
        gd.context_lens = Tensor({(long)bs}, DType::I32, Device::CUDA, capture_context);
        gd.block_table = Tensor({(long)(bs * max_context_blocks_)}, DType::I32, Device::CUDA, capture_context);
        gd.next_tokens = Tensor({(long)bs}, DType::I32, Device::CUDA, capture_context);

        cudaMemset(gd.input_ids.data(), 0, bs * sizeof(int));
        cudaMemset(gd.context_lens.data(), 0, bs * sizeof(int));
        std::vector<int> h_capture_block_table(bs * max_context_blocks_, decode_scratch_first_block_);
        for (int i = 0; i < bs; ++i)
        {
            h_capture_block_table[i * max_context_blocks_] = decode_scratch_first_block_ + i;
        }
        cudaMemcpy(gd.block_table.data(), h_capture_block_table.data(), h_capture_block_table.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        auto result = gd.graph.capture(capture_stream,
                                       [&]()
                                       {
                                           auto* d_block_table = static_cast<int*>(gd.block_table.data());
                                           model::ModelInput model_input{
                                               gd.input_ids,
                                               gd.context_lens,
                                               {k_caches_, v_caches_, d_block_table, max_context_blocks_}};
                                           Tensor logits = model_->forward(model_input, {.context = capture_context});

                                           kernels::argmax(logits, gd.next_tokens, capture_context);
                                       });

        if (!result)
        {
            std::cerr << "Failed to capture graph for batch size " << bs << ": " << result.error().description()
                      << std::endl;
            return std::nullopt;
        }

        cudaMallocHost((void**)&gd.h_input_ids, bs * sizeof(int));
        cudaMallocHost((void**)&gd.h_context_lens, bs * sizeof(int));
        cudaMallocHost((void**)&gd.h_block_table, bs * max_context_blocks_ * sizeof(int));
        cudaMallocHost((void**)&gd.h_next_tokens, bs * sizeof(int));

        std::cout << "Successfully captured execution graph for decode batch size " << bs << std::endl;
        [[maybe_unused]] auto _ = gd.graph.launch(capture_stream);
        cudaStreamSynchronize(capture_stream);
        return gd;
    };

    for (int bs : supported_batch_sizes_)
    {
        auto gd = capture_decode_graph(bs);
        if (gd)
        {
            dec_graphs_[bs] = std::move(*gd);
        }
    }

    running_ = true;
    background_thread_ = std::thread(&Engine::loop, this);
}

void Engine::stop()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
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

void Engine::copy_prefix_cow_blocks(const std::vector<scheduler::SequencePtr>& requests,
                                    const device::Context& context)
{
    size_t block_bytes = 16 * static_cast<size_t>(config_.num_key_value_heads) * static_cast<size_t>(config_.head_dim) *
                         dtype_size(config_.dtype);
    cudaStream_t stream = context.stream();

    for (const auto& req : requests)
    {
        if (req->prefix_cow_copied || req->prefix_cow_source_block < 0 || req->prefix_cow_private_block < 0)
        {
            continue;
        }

        size_t src_offset = static_cast<size_t>(req->prefix_cow_source_block) * block_bytes;
        size_t dst_offset = static_cast<size_t>(req->prefix_cow_private_block) * block_bytes;
        for (size_t layer = 0; layer < k_caches_.size(); ++layer)
        {
            auto* k_base = static_cast<std::byte*>(k_caches_[layer].data());
            auto* v_base = static_cast<std::byte*>(v_caches_[layer].data());
            cudaMemcpyAsync(k_base + dst_offset, k_base + src_offset, block_bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_base + dst_offset, v_base + src_offset, block_bytes, cudaMemcpyDeviceToDevice, stream);
        }
        req->prefix_cow_copied = true;
    }
}

void Engine::loop()
{
    cudaSetDevice(0);
    auto decode_stream_result = device::Stream::create();
    if (!decode_stream_result) throw std::runtime_error(decode_stream_result.error().description());
    device::Stream decode_stream_owner = std::move(decode_stream_result.value());
    cudaStream_t decode_stream = decode_stream_owner.get();
    device::Context decode_context = decode_stream_owner.context();

    auto ensure_decode_fallback_storage = [this, &decode_context](int current_batch_size, int block_table_elems)
    {
        if (decode_fb_input_capacity_ < current_batch_size)
        {
            decode_fb_input_storage_ = Tensor({current_batch_size, 1}, DType::I32, Device::CUDA, decode_context);
            decode_fb_input_capacity_ = current_batch_size;
        }
        if (decode_fb_context_capacity_ < current_batch_size)
        {
            decode_fb_context_storage_ = Tensor({current_batch_size}, DType::I32, Device::CUDA, decode_context);
            decode_fb_context_capacity_ = current_batch_size;
        }
        if (decode_fb_next_token_capacity_ < current_batch_size)
        {
            decode_fb_next_token_storage_ = Tensor({current_batch_size}, DType::I32, Device::CUDA, decode_context);
            decode_fb_next_token_capacity_ = current_batch_size;
        }
        if (decode_fb_block_table_capacity_ < block_table_elems)
        {
            decode_fb_block_table_storage_ = Tensor({block_table_elems}, DType::I32, Device::CUDA, decode_context);
            decode_fb_block_table_capacity_ = block_table_elems;
        }
    };

    while (running_)
    {
        if (!scheduler_.has_unfinished_sequences())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        FIREFLY_NVTX_PUSH("Engine_Schedule");
        auto batch = scheduler_.step();
        FIREFLY_NVTX_POP();
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto t1 = std::chrono::high_resolution_clock::now();
#endif
        for (const auto& req : batch.failed_sequences)
        {
            if (result_queue_)
            {
                result_queue_->push(req->id, req->error_message.empty() ? "request failed" : req->error_message, true,
                                    static_cast<int>(req->prompt_tokens.size()),
                                    static_cast<int>(req->generated_tokens.size()));
            }
        }
        if (batch.sequences.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        std::vector<scheduler::SequencePtr> prefill_requests;
        std::vector<scheduler::SequencePtr> decode_requests;

        for (auto req : batch.sequences)
        {
            if (req->is_cancelled())
            {
                scheduler_.abort_sequence(req);
                if (result_queue_)
                {
                    result_queue_->push(req->id, "", true, static_cast<int>(req->prompt_tokens.size()),
                                        static_cast<int>(req->generated_tokens.size()));
                }
                continue;
            }
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
            FIREFLY_NVTX_PUSH("Engine_Prefill");
            std::map<int, std::vector<scheduler::SequencePtr>, std::greater<int>> prefill_groups;
            for (auto req : prefill_requests)
            {
                int unmatched_tokens = req->prompt_tokens.size() - req->context_len;
                int seq_len = std::min(unmatched_tokens, max_prefill_chunk_size_);
                if (seq_len > 0)
                {
                    prefill_groups[seq_len].push_back(req);
                }
            }

            for (auto& [seq_len, group] : prefill_groups)
            {
                int group_size = group.size();
                int max_blocks = 0;
                int min_context_len = std::numeric_limits<int>::max();
                bool any_prompt_finished = false;

                std::vector<int> h_input_ids(group_size * seq_len);
                std::vector<int> h_context_lens(group_size);
                for (int i = 0; i < group_size; ++i)
                {
                    auto req = group[i];
                    std::copy_n(req->prompt_tokens.begin() + req->context_len, seq_len,
                                h_input_ids.begin() + i * seq_len);
                    h_context_lens[i] = req->context_len;
                    min_context_len = std::min(min_context_len, req->context_len);
                    any_prompt_finished =
                        any_prompt_finished || (req->context_len + seq_len == (int)req->prompt_tokens.size());
                    max_blocks = std::max<int>(max_blocks, req->block_table.size());
                }

                Tensor d_input({(long)group_size, seq_len}, DType::I32, Device::CUDA, decode_context);
                cudaMemcpyAsync(d_input.data(), h_input_ids.data(), h_input_ids.size() * sizeof(int),
                                cudaMemcpyHostToDevice, decode_stream);

                Tensor d_context_lens({(long)group_size}, DType::I32, Device::CUDA, decode_context);
                cudaMemcpyAsync(d_context_lens.data(), h_context_lens.data(), h_context_lens.size() * sizeof(int),
                                cudaMemcpyHostToDevice, decode_stream);

                std::vector<int> h_block_table(group_size * max_blocks, decode_scratch_first_block_);
                for (int i = 0; i < group_size; ++i)
                {
                    auto& blocks = group[i]->block_table;
                    std::copy(blocks.begin(), blocks.end(), h_block_table.begin() + i * max_blocks);
                }

                Tensor d_block_table_tensor({(long)(group_size * max_blocks)}, DType::I32, Device::CUDA,
                                            decode_context);
                int* d_block_table = static_cast<int*>(d_block_table_tensor.data());
                cudaMemcpyAsync(d_block_table, h_block_table.data(), h_block_table.size() * sizeof(int),
                                cudaMemcpyHostToDevice, decode_stream);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                if (profile_prefill_enabled())
                {
                    auto& events = prefill_profile_events();
                    cudaEventRecord(events.start, decode_stream);
                }
#endif

                copy_prefix_cow_blocks(group, decode_context);

                model::ModelInput model_input{
                    d_input, d_context_lens, {k_caches_, v_caches_, d_block_table, max_blocks}};
                model::ForwardOptions forward_options{.context = decode_context,
                                                       .compute_logits = any_prompt_finished,
                                                       .min_context_len = min_context_len};
                Tensor logits = model_->forward(model_input, forward_options);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                if (profile_prefill_enabled())
                {
                    auto& events = prefill_profile_events();
                    cudaEventRecord(events.stop, decode_stream);
                    cudaEventSynchronize(events.stop);
                    float elapsed_ms = 0.0f;
                    cudaEventElapsedTime(&elapsed_ms, events.start, events.stop);
                    std::cout << "PREFILL_GPU chunk: bs=" << group_size << " seq=" << seq_len
                              << " prompt_finished=" << any_prompt_finished << " time=" << elapsed_ms
                              << "ms tok/s=" << (group_size * seq_len * 1000.0f / std::max(elapsed_ms, 1e-3f))
                              << std::endl;
                }
#endif

                for (auto& req : group)
                {
                    req->context_len += seq_len;
                }

                if (any_prompt_finished)
                {
                    Tensor next_tokens({(long)group_size}, DType::I32, Device::CUDA, decode_context);
                    kernels::argmax(logits, next_tokens, decode_context);
                    std::vector<int> h_next_tokens(group_size);
                    cudaMemcpyAsync(h_next_tokens.data(), next_tokens.data(), h_next_tokens.size() * sizeof(int),
                                    cudaMemcpyDeviceToHost, decode_stream);
                    cudaStreamSynchronize(decode_stream);

                    for (int i = 0; i < group_size; ++i)
                    {
                        auto req = group[i];
                        if (req->context_len == (int)req->prompt_tokens.size())
                        {
                            req->generated_tokens.push_back(h_next_tokens[i]);
                        }
                    }
                }
            }
            FIREFLY_NVTX_POP();  // Engine_Prefill
        }

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto t2 = std::chrono::high_resolution_clock::now();
#endif

        // --- DECODE (True Continuous Batching with CUDA Graph) ---
        if (!decode_requests.empty())
        {
            FIREFLY_NVTX_PUSH("Engine_Decode");
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

            int max_decode_context_len = 0;
            for (const auto& req : decode_requests)
            {
                max_decode_context_len = std::max(max_decode_context_len, req->context_len);
            }
            bool use_split_decode = max_decode_context_len >= 4096;

            auto run_decode_fallback = [&](bool prefer_split_decode)
            {
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                auto f0 = std::chrono::high_resolution_clock::now();
#endif
                int max_blocks = 0;
                auto& scratch = decode_host_scratch();
                scratch.input_ids.resize(current_batch_size);
                scratch.context_lens.resize(current_batch_size);
                for (int i = 0; i < current_batch_size; ++i)
                {
                    auto req = decode_requests[i];
                    scratch.input_ids[i] = req->generated_tokens.back();
                    scratch.context_lens[i] = req->context_len;
                    max_blocks = std::max<int>(max_blocks, req->block_table.size());
                }

                scratch.block_table.assign(current_batch_size * max_blocks, decode_scratch_first_block_);
                for (int i = 0; i < current_batch_size; ++i)
                {
                    const auto& blocks = decode_requests[i]->block_table;
                    std::copy(blocks.begin(), blocks.end(), scratch.block_table.begin() + i * max_blocks);
                }

                int block_table_elems = current_batch_size * max_blocks;
                ensure_decode_fallback_storage(current_batch_size, block_table_elems);
                Tensor d_input =
                    Tensor::from_external(decode_fb_input_storage_.data(), {current_batch_size, 1}, DType::I32,
                                          Device::CUDA);
                Tensor d_context_lens =
                    Tensor::from_external(decode_fb_context_storage_.data(), {current_batch_size}, DType::I32,
                                          Device::CUDA);
                Tensor d_block_table_tensor = Tensor::from_external(decode_fb_block_table_storage_.data(),
                                                                    {block_table_elems}, DType::I32, Device::CUDA);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                DecodeProfileEvents* gpu_events = nullptr;
                cudaStream_t profile_stream = decode_stream;
                if (profile_decode_enabled())
                {
                    gpu_events = &decode_profile_events();
                    cudaEventRecord(gpu_events->h2d_start, profile_stream);
                }
#endif

                cudaMemcpyAsync(d_input.data(), scratch.input_ids.data(), scratch.input_ids.size() * sizeof(int),
                                cudaMemcpyHostToDevice, decode_stream);
                cudaMemcpyAsync(d_context_lens.data(), scratch.context_lens.data(),
                                scratch.context_lens.size() * sizeof(int), cudaMemcpyHostToDevice, decode_stream);
                cudaMemcpyAsync(d_block_table_tensor.data(), scratch.block_table.data(),
                                scratch.block_table.size() * sizeof(int), cudaMemcpyHostToDevice, decode_stream);
                if (kernels::get_attention_backend() == kernels::AttentionBackend::FlashInfer)
                {
                    kernels::prepare_attention_decode(scratch.context_lens.data(), scratch.block_table.data(),
                                                      current_batch_size, max_blocks, config_.num_attention_heads,
                                                      decode_context);
                }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                if (gpu_events) cudaEventRecord(gpu_events->h2d_stop, profile_stream);
                auto f1 = std::chrono::high_resolution_clock::now();
#endif

                model::ModelInput model_input{
                    d_input,
                    d_context_lens,
                    {k_caches_, v_caches_, static_cast<int*>(d_block_table_tensor.data()), max_blocks}};
                model::ForwardOptions forward_options{.context = decode_context,
                                                        .prefer_split_decode = prefer_split_decode,
                                                        .max_decode_context_len = max_decode_context_len};
                Tensor logits = model_->forward(model_input, forward_options);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                if (gpu_events) cudaEventRecord(gpu_events->fwd_stop, profile_stream);
                auto f2 = std::chrono::high_resolution_clock::now();
#endif
                Tensor next_tokens = Tensor::from_external(decode_fb_next_token_storage_.data(), {current_batch_size},
                                                           DType::I32, Device::CUDA);
                kernels::argmax(logits, next_tokens, decode_context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                if (gpu_events) cudaEventRecord(gpu_events->argmax_stop, profile_stream);
                auto f3 = std::chrono::high_resolution_clock::now();
#endif

                scratch.next_tokens.resize(current_batch_size);
                cudaMemcpyAsync(scratch.next_tokens.data(), next_tokens.data(), scratch.next_tokens.size() * sizeof(int),
                                cudaMemcpyDeviceToHost, decode_stream);
                cudaStreamSynchronize(decode_stream);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                if (gpu_events) cudaEventRecord(gpu_events->d2h_stop, profile_stream);
                auto f4 = std::chrono::high_resolution_clock::now();
#endif

                for (int i = 0; i < current_batch_size; ++i)
                {
                    auto req = decode_requests[i];
                    req->generated_tokens.push_back(scratch.next_tokens[i]);
                    req->context_len += 1;
                }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                auto f5 = std::chrono::high_resolution_clock::now();

                if (profile_decode_enabled())
                {
                    static double fb_prep = 0, fb_forward = 0, fb_argmax = 0, fb_d2h = 0, fb_update = 0;
                    static double fb_gpu_h2d = 0, fb_gpu_fwd = 0, fb_gpu_argmax = 0, fb_gpu_d2h = 0;
                    static int    fb_steps = 0;
                    fb_prep += std::chrono::duration<double, std::milli>(f1 - f0).count();
                    fb_forward += std::chrono::duration<double, std::milli>(f2 - f1).count();
                    fb_argmax += std::chrono::duration<double, std::milli>(f3 - f2).count();
                    fb_d2h += std::chrono::duration<double, std::milli>(f4 - f3).count();
                    fb_update += std::chrono::duration<double, std::milli>(f5 - f4).count();
                    if (gpu_events)
                    {
                        cudaEventSynchronize(gpu_events->d2h_stop);
                        float elapsed_ms = 0.0f;
                        cudaEventElapsedTime(&elapsed_ms, gpu_events->h2d_start, gpu_events->h2d_stop);
                        fb_gpu_h2d += elapsed_ms;
                        cudaEventElapsedTime(&elapsed_ms, gpu_events->h2d_stop, gpu_events->fwd_stop);
                        fb_gpu_fwd += elapsed_ms;
                        cudaEventElapsedTime(&elapsed_ms, gpu_events->fwd_stop, gpu_events->argmax_stop);
                        fb_gpu_argmax += elapsed_ms;
                        cudaEventElapsedTime(&elapsed_ms, gpu_events->argmax_stop, gpu_events->d2h_stop);
                        fb_gpu_d2h += elapsed_ms;
                    }
                    fb_steps++;
                    if (fb_steps % 50 == 0)
                    {
                        std::cout << "DECODE_FALLBACK [50]: Prep:" << fb_prep << "ms, Forward:" << fb_forward
                                  << "ms, Argmax:" << fb_argmax << "ms, D2H:" << fb_d2h
                                  << "ms, Update:" << fb_update << "ms" << std::endl;
                        std::cout << "DECODE_FALLBACK_GPU [50]: H2D:" << fb_gpu_h2d << "ms, Forward:"
                                  << fb_gpu_fwd << "ms, Argmax:" << fb_gpu_argmax << "ms, D2H:" << fb_gpu_d2h
                                  << "ms" << std::endl;
                        fb_prep = fb_forward = fb_argmax = fb_d2h = fb_update = 0;
                        fb_gpu_h2d = fb_gpu_fwd = fb_gpu_argmax = fb_gpu_d2h = 0;
                    }
                }
#endif
            };

            auto gd_it = dec_graphs_.find(target_bs);
            const bool use_flashinfer = kernels::get_attention_backend() == kernels::AttentionBackend::FlashInfer;
            if (gd_it == dec_graphs_.end() || use_split_decode || use_flashinfer)
            {
                FIREFLY_NVTX_PUSH("Engine_Decode_Fallback");
                run_decode_fallback(use_split_decode);
                FIREFLY_NVTX_POP();  // Engine_Decode_Fallback
                FIREFLY_NVTX_POP();  // Engine_Decode
            }
            else
            {
                auto& gd = gd_it->second;

                std::fill(gd.h_input_ids, gd.h_input_ids + target_bs, 0);
                std::fill(gd.h_context_lens, gd.h_context_lens + target_bs, 0);
                std::fill(gd.h_block_table, gd.h_block_table + target_bs * max_context_blocks_,
                          decode_scratch_first_block_);
                for (int i = current_batch_size; i < target_bs; ++i)
                {
                    gd.h_block_table[i * max_context_blocks_] = decode_scratch_first_block_ + i;
                }

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
                FIREFLY_NVTX_PUSH("Engine_Decode_Graph_Copy");
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                auto d0 = std::chrono::high_resolution_clock::now();
#endif
                cudaMemcpyAsync(gd.input_ids.data(), gd.h_input_ids, target_bs * sizeof(int), cudaMemcpyHostToDevice,
                                decode_stream);
                cudaMemcpyAsync(gd.context_lens.data(), gd.h_context_lens, target_bs * sizeof(int),
                                cudaMemcpyHostToDevice, decode_stream);
                cudaMemcpyAsync(gd.block_table.data(), gd.h_block_table, target_bs * max_context_blocks_ * sizeof(int),
                                cudaMemcpyHostToDevice, decode_stream);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                auto d1 = std::chrono::high_resolution_clock::now();
#endif
                FIREFLY_NVTX_POP();  // Engine_Decode_Graph_Copy

                // Launch the static graph
                FIREFLY_NVTX_PUSH("Engine_Decode_Graph_Launch");
                auto result = gd.graph.launch(decode_stream);
                if (!result)
                {
                    std::cerr << "Graph Launch failed: " << result.error().description() << std::endl;
                    FIREFLY_NVTX_POP();  // Engine_Decode_Graph_Launch

                    FIREFLY_NVTX_PUSH("Engine_Decode_Fallback");
                    run_decode_fallback(false);
                    FIREFLY_NVTX_POP();  // Engine_Decode_Fallback
                    FIREFLY_NVTX_POP();  // Engine_Decode
                }
                else
                {
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                    auto d2 = std::chrono::high_resolution_clock::now();
#endif

                    // Extract tokens back to host
                    cudaMemcpyAsync(gd.h_next_tokens, gd.next_tokens.data(), target_bs * sizeof(int),
                                    cudaMemcpyDeviceToHost, decode_stream);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                    auto d3 = std::chrono::high_resolution_clock::now();
#endif

                    // Synchronize before reading host elements
                    cudaStreamSynchronize(decode_stream);
                    FIREFLY_NVTX_POP();  // Engine_Decode_Graph_Launch
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
                    auto d4 = std::chrono::high_resolution_clock::now();
#endif

                    for (int i = 0; i < current_batch_size; ++i)
                    {
                        auto req = decode_requests[i];
                        req->generated_tokens.push_back(gd.h_next_tokens[i]);
                        req->context_len += 1;
                    }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
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
#endif
                    FIREFLY_NVTX_POP();  // Engine_Decode
                }
            }
        }

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto t3 = std::chrono::high_resolution_clock::now();
#endif

        // --- Process Results & Check EOS ---
        FIREFLY_NVTX_PUSH("Engine_Process_Results");
        process_outputs(batch.sequences);
        FIREFLY_NVTX_POP();  // Engine_Process_Results

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto t4 = std::chrono::high_resolution_clock::now();

        if (!batch.sequences.empty())
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
#endif
    }

}

}  // namespace firefly::execution
