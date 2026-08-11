#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "firefly/core/logging.h"
#include "firefly/device/stream.h"
#include "firefly/execution/engine.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/sampling/argmax.h"

namespace firefly::execution
{
void Engine::start()
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (running_) return;
    if (options_.gpu_memory_utilization <= 0.0 || options_.gpu_memory_utilization > 1.0)
    {
        throw std::runtime_error("GPU memory utilization must be in (0, 1]");
    }

    cudaSetDevice(0);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    const bool kv_cache_quantized = options_.kv_cache_format == KVCacheFormat::Int8;
    if (runtime_requirements_.sequence_state && kv_cache_quantized)
    {
        throw std::runtime_error("quantized KV cache is not supported for models with recurrent sequence state");
    }
    model_->initialize_runtime(supported_batch_sizes_.back(), {});
    int        profile_token_count = runtime_requirements_.sequence_state
                                         ? options_.max_prefill_chunk_size
                                         : std::max(2048, options_.max_prefill_chunk_size);
    Tensor     profile_input({1, profile_token_count}, DType::I32, Device::CUDA);
    cudaMemset(profile_input.data(), 0, profile_token_count * sizeof(int));

    int                 profile_block_count = (profile_token_count + 15) / 16;
    std::vector<Tensor> profile_keys;
    std::vector<Tensor> profile_values;
    std::vector<Tensor> profile_scales;
    for (int i = 0; i < runtime_requirements_.kv_cache_layer_count; ++i)
    {
        profile_keys.emplace_back(
            Tensor({(long)profile_block_count, 16, (long)runtime_requirements_.kv_cache_head_count,
                    (long)runtime_requirements_.kv_cache_head_dim},
                   kv_cache_quantized ? DType::I8 : config_.dtype, Device::CUDA));
        profile_values.emplace_back(
            Tensor({(long)profile_block_count, 16, (long)runtime_requirements_.kv_cache_head_count,
                    (long)runtime_requirements_.kv_cache_head_dim},
                   kv_cache_quantized ? DType::I8 : config_.dtype, Device::CUDA));
        if (kv_cache_quantized)
        {
            profile_scales.emplace_back(
                Tensor({(long)profile_block_count, 16, (long)runtime_requirements_.kv_cache_head_count, 2}, DType::F32,
                       Device::CUDA));
        }
    }
    std::vector<int> profile_block_table_host(profile_block_count);
    for (int i = 0; i < profile_block_count; ++i) profile_block_table_host[i] = i;
    Tensor profile_block_table({profile_block_count}, DType::I32, Device::CUDA);
    cudaMemcpy(profile_block_table.data(), profile_block_table_host.data(), profile_block_count * sizeof(int),
               cudaMemcpyHostToDevice);

    Tensor profile_context_lengths({1}, DType::I32, Device::CUDA);
    cudaMemset(profile_context_lengths.data(), 0, sizeof(int));
    Tensor profile_state_slot({1}, DType::I32, Device::CUDA);
    cudaMemset(profile_state_slot.data(), 0, sizeof(int));

    model::ModelInput profile_model_input{profile_input,
                                          profile_context_lengths,
                                          {profile_keys, profile_values, profile_scales,
                                           static_cast<int*>(profile_block_table.data()), profile_block_count},
                                          static_cast<int*>(profile_state_slot.data())};
    model_->forward(profile_model_input, {.compute_logits = false, .min_context_len = 0});
    cudaError_t profile_error = cudaDeviceSynchronize();
    if (profile_error != cudaSuccess)
    {
        throw std::runtime_error(std::string("memory profile forward failed: ") + cudaGetErrorString(profile_error));
    }
    model_->reset_runtime({});

    size_t profile_free_bytes = 0;
    size_t profile_total_bytes = 0;
    cudaMemGetInfo(&profile_free_bytes, &profile_total_bytes);

    size_t usable_memory = static_cast<size_t>(profile_free_bytes * options_.gpu_memory_utilization);

    size_t block_bytes = runtime_requirements_.kv_cache_layer_count * 16 * runtime_requirements_.kv_cache_head_count *
                         runtime_requirements_.kv_cache_head_dim * 2 *
                         dtype_size(kv_cache_quantized ? DType::I8 : config_.dtype);
    if (kv_cache_quantized)
    {
        block_bytes += runtime_requirements_.kv_cache_layer_count * 16 * runtime_requirements_.kv_cache_head_count * 2 *
                       sizeof(float);
    }
    int max_num_blocks = usable_memory / block_bytes;
    decode_scratch_blocks_ = supported_batch_sizes_.back();
    if (max_num_blocks <= decode_scratch_blocks_)
    {
        throw std::runtime_error("Not enough KV cache blocks for decode graph scratch rows");
    }
    max_num_blocks -= decode_scratch_blocks_;
    decode_scratch_first_block_ = max_num_blocks;
    max_context_blocks_ = max_num_blocks;

    FIREFLY_LOG_INFO("runtime",
                     "KV cache ready format={} blocks={} tokens={} memory_mb={} gpu_memory_mb={} "
                     "free_before_mb={} free_after_mb={} capacity_8k={} capacity_16k={} scratch_blocks={}",
                     kv_cache_quantized ? "int8" : "model", max_num_blocks, max_num_blocks * 16,
                     (max_num_blocks * block_bytes) / (1024 * 1024), total_bytes / (1024 * 1024),
                     free_bytes / (1024 * 1024), profile_free_bytes / (1024 * 1024),
                     (max_num_blocks * 16) / 8192, (max_num_blocks * 16) / 16384, decode_scratch_blocks_);

    for (int i = 0; i < runtime_requirements_.kv_cache_layer_count; ++i)
    {
        k_caches_.emplace_back(Tensor({(long)(max_num_blocks + decode_scratch_blocks_), 16,
                                       (long)runtime_requirements_.kv_cache_head_count,
                                       (long)runtime_requirements_.kv_cache_head_dim},
                                      kv_cache_quantized ? DType::I8 : config_.dtype, Device::CUDA));
        v_caches_.emplace_back(Tensor({(long)(max_num_blocks + decode_scratch_blocks_), 16,
                                       (long)runtime_requirements_.kv_cache_head_count,
                                       (long)runtime_requirements_.kv_cache_head_dim},
                                      kv_cache_quantized ? DType::I8 : config_.dtype, Device::CUDA));
        if (kv_cache_quantized)
        {
            kv_scale_caches_.emplace_back(
                Tensor({(long)(max_num_blocks + decode_scratch_blocks_),
                        (long)runtime_requirements_.kv_cache_head_count, 16, 2},
                       DType::F32, Device::CUDA));
        }
    }

    scheduler_.init(max_num_blocks, supported_batch_sizes_.back(), options_.max_prefill_chunk_size,
                    runtime_requirements_.prefix_cache);

    if (kv_cache_quantized)
    {
        kernels::reserve_paged_decode_scratch(supported_batch_sizes_.back(), config_.num_attention_heads,
                                              max_context_blocks_, runtime_requirements_.kv_cache_head_dim);
    }

    auto capture_stream_result = device::Stream::create();
    if (!capture_stream_result) throw std::runtime_error(capture_stream_result.error().description());
    device::Stream  capture_stream_owner = std::move(capture_stream_result.value());
    cudaStream_t    capture_stream = capture_stream_owner.get();
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
                                           auto*             d_block_table = static_cast<int*>(gd.block_table.data());
                                           model::ModelInput model_input{gd.input_ids,
                                                                         gd.context_lens,
                                                                         {k_caches_, v_caches_, kv_scale_caches_,
                                                                          d_block_table, max_context_blocks_}};
                                           Tensor logits = model_->forward(model_input, {.context = capture_context});

                                           kernels::argmax(logits, gd.next_tokens, capture_context);
                                       });

        if (!result)
        {
            FIREFLY_LOG_ERROR("runtime", "decode graph capture failed batch_size={} error={}", bs,
                              result.error().description());
            return std::nullopt;
        }

        cudaMallocHost((void**)&gd.h_input_ids, bs * sizeof(int));
        cudaMallocHost((void**)&gd.h_context_lens, bs * sizeof(int));
        cudaMallocHost((void**)&gd.h_block_table, bs * max_context_blocks_ * sizeof(int));
        cudaMallocHost((void**)&gd.h_next_tokens, bs * sizeof(int));

        FIREFLY_LOG_DEBUG("runtime", "decode graph captured batch_size={}", bs);
        [[maybe_unused]] auto _ = gd.graph.launch(capture_stream);
        cudaStreamSynchronize(capture_stream);
        return gd;
    };

    for (int bs : runtime_requirements_.cuda_graph ? supported_batch_sizes_ : std::vector<int>{})
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
}  // namespace firefly::execution
