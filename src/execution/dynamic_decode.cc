#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "firefly/core/logging.h"
#include "firefly/device/error.h"
#include "firefly/execution/engine.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/sampling/argmax.h"

namespace firefly::execution
{
namespace
{
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
bool profile_decode_enabled()
{
    static bool enabled = []()
    {
        const char* value = std::getenv("FIREFLY_PROFILE_DECODE");
        return value && std::strcmp(value, "0") != 0;
    }();
    return enabled;
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
    std::vector<int> state_slots;
};

DecodeHostScratch& decode_host_scratch()
{
    static thread_local DecodeHostScratch scratch;
    return scratch;
}
}  // namespace

Status Engine::process_dynamic_decode(const std::vector<scheduler::SequencePtr>& requests, int max_context_length,
                                      bool prefer_split_decode, const device::Context& context)
{
    if (requests.empty()) return {};
    cudaStream_t decode_stream = context.stream();
    int          batch_size = requests.size();

    auto ensure_storage = [this, &context](int required_batch_size, int block_table_elements) -> Status
    {
        if (decode_fb_input_capacity_ < required_batch_size)
        {
            decode_fb_input_storage_ = FIREFLY_TRY(
                Tensor::create({required_batch_size, 1}, DType::I32, Device::CUDA, context));
            decode_fb_input_capacity_ = required_batch_size;
        }
        if (decode_fb_context_capacity_ < required_batch_size)
        {
            decode_fb_context_storage_ = FIREFLY_TRY(
                Tensor::create({required_batch_size}, DType::I32, Device::CUDA, context));
            decode_fb_context_capacity_ = required_batch_size;
        }
        if (decode_fb_next_token_capacity_ < required_batch_size)
        {
            decode_fb_next_token_storage_ = FIREFLY_TRY(
                Tensor::create({required_batch_size}, DType::I32, Device::CUDA, context));
            decode_fb_next_token_capacity_ = required_batch_size;
        }
        if (decode_fb_state_slot_capacity_ < required_batch_size)
        {
            decode_fb_state_slot_storage_ = FIREFLY_TRY(
                Tensor::create({required_batch_size}, DType::I32, Device::CUDA, context));
            decode_fb_state_slot_capacity_ = required_batch_size;
        }
        if (decode_fb_block_table_capacity_ < block_table_elements)
        {
            decode_fb_block_table_storage_ = FIREFLY_TRY(
                Tensor::create({block_table_elements}, DType::I32, Device::CUDA, context));
            decode_fb_block_table_capacity_ = block_table_elements;
        }
        return {};
    };

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto f0 = std::chrono::high_resolution_clock::now();
#endif
    int   max_blocks = 0;
    auto& scratch = decode_host_scratch();
    scratch.input_ids.resize(batch_size);
    scratch.context_lens.resize(batch_size);
    scratch.state_slots.resize(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        auto req = requests[i];
        scratch.input_ids[i] = req->generated_tokens.back();
        scratch.context_lens[i] = req->context_len;
        scratch.state_slots[i] = req->state_slot;
        max_blocks = std::max<int>(max_blocks, req->block_table.size());
    }

    FIREFLY_LOG_DEBUG("decode", "begin batch={} max_context={} split={} max_blocks={}", batch_size,
                      max_context_length, prefer_split_decode, max_blocks);
    for (const auto& req : requests)
    {
        FIREFLY_LOG_DEBUG("decode", "request_id={} context={} prompt={} generated={} state_slot={} blocks={}",
                          req->id, req->context_len, req->prompt_tokens.size(), req->generated_tokens.size(),
                          req->state_slot, req->block_table.size());
    }

    constexpr int quantized_split_bucket_size = 128;
    int quantized_context_bucket = (max_context_length + quantized_split_bucket_size) / quantized_split_bucket_size;
    if (options_.kv_cache_format == KVCacheFormat::Int8 && prefer_split_decode)
    {
        max_blocks = quantized_context_bucket * (quantized_split_bucket_size / 16);
    }

    scratch.block_table.assign(batch_size * max_blocks, decode_scratch_first_block_);
    for (int i = 0; i < batch_size; ++i)
    {
        const auto& blocks = requests[i]->block_table;
        std::copy(blocks.begin(), blocks.end(), scratch.block_table.begin() + i * max_blocks);
    }

    int block_table_elems = batch_size * max_blocks;
    FIREFLY_TRY(ensure_storage(batch_size, block_table_elems));
    Tensor d_input = Tensor::from_external(decode_fb_input_storage_.data(), {batch_size, 1}, DType::I32, Device::CUDA);
    Tensor d_context_lens =
        Tensor::from_external(decode_fb_context_storage_.data(), {batch_size}, DType::I32, Device::CUDA);
    Tensor d_state_slots =
        Tensor::from_external(decode_fb_state_slot_storage_.data(), {batch_size}, DType::I32, Device::CUDA);
    Tensor d_block_table_tensor =
        Tensor::from_external(decode_fb_block_table_storage_.data(), {block_table_elems}, DType::I32, Device::CUDA);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    DecodeProfileEvents* gpu_events = nullptr;
    cudaStream_t         profile_stream = decode_stream;
    if (profile_decode_enabled())
    {
        gpu_events = &decode_profile_events();
        cudaEventRecord(gpu_events->h2d_start, profile_stream);
    }
#endif

    FIREFLY_TRY(device::check_cuda(cudaMemcpyAsync(d_input.data(), scratch.input_ids.data(),
                                                   scratch.input_ids.size() * sizeof(int), cudaMemcpyHostToDevice,
                                                   decode_stream), "copy decode token IDs"));
    FIREFLY_TRY(device::check_cuda(cudaMemcpyAsync(d_context_lens.data(), scratch.context_lens.data(),
                                                   scratch.context_lens.size() * sizeof(int), cudaMemcpyHostToDevice,
                                                   decode_stream), "copy decode context lengths"));
    FIREFLY_TRY(device::check_cuda(cudaMemcpyAsync(d_state_slots.data(), scratch.state_slots.data(),
                                                   scratch.state_slots.size() * sizeof(int), cudaMemcpyHostToDevice,
                                                   decode_stream), "copy decode state slots"));
    FIREFLY_TRY(device::check_cuda(cudaMemcpyAsync(d_block_table_tensor.data(), scratch.block_table.data(),
                                                   scratch.block_table.size() * sizeof(int), cudaMemcpyHostToDevice,
                                                   decode_stream), "copy decode block table"));
    if (options_.kv_cache_format != KVCacheFormat::Int8 &&
        kernels::get_attention_backend() == kernels::AttentionBackend::FlashInfer)
    {
        FIREFLY_TRY(kernels::prepare_attention_decode(
            scratch.context_lens.data(), scratch.block_table.data(), batch_size, max_blocks,
            config_.num_attention_heads, runtime_requirements_.kv_cache_head_count,
            runtime_requirements_.kv_cache_head_dim, context));
    }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    if (gpu_events) cudaEventRecord(gpu_events->h2d_stop, profile_stream);
    auto f1 = std::chrono::high_resolution_clock::now();
#endif

    model::ModelInput model_input{d_input,
                                  d_context_lens,
                                  {k_caches_, v_caches_, kv_scale_caches_,
                                   static_cast<int*>(d_block_table_tensor.data()), max_blocks},
                                  static_cast<int*>(d_state_slots.data())};
    Tensor layered_hidden_states;
    model::ForwardOptions forward_options{
        .context = context,
        .prefer_split_decode = prefer_split_decode, .max_decode_context_len = max_context_length};
    if (speculative_decoder_ != nullptr)
        FIREFLY_TRY(speculative_decoder_->configure_target_forward(forward_options, layered_hidden_states,
                                                                   batch_size, 1));
    Tensor next_tokens =
        Tensor::from_external(decode_fb_next_token_storage_.data(), {batch_size}, DType::I32, Device::CUDA);
    const bool use_flashinfer_graph = runtime_requirements_.cuda_graph &&
                                      options_.kv_cache_format != KVCacheFormat::Int8 &&
                                      kernels::get_attention_backend() == kernels::AttentionBackend::FlashInfer &&
                                      !prefer_split_decode;
    bool       graph_launched = false;
    if (runtime_requirements_.cuda_graph && options_.kv_cache_format == KVCacheFormat::Int8 && prefer_split_decode)
    {
        const bool graph_shape_changed =
            quantized_decode_graph_.empty() || quantized_decode_batch_size_ != batch_size ||
            quantized_decode_max_blocks_ != max_blocks || quantized_decode_context_bucket_ != quantized_context_bucket;
        if (graph_shape_changed)
        {
            FIREFLY_TRY(kernels::reserve_paged_decode_scratch(batch_size, config_.num_attention_heads, max_blocks,
                                                              config_.head_dim));
            quantized_decode_graph_ = device::Graph{};
            quantized_decode_layered_hidden_states_ = std::move(layered_hidden_states);
            if (speculative_decoder_ != nullptr)
            {
                forward_options.hidden_state_layers = speculative_decoder_->target_hidden_layers();
                forward_options.layered_hidden_state_output = &quantized_decode_layered_hidden_states_;
                layered_hidden_states = Tensor::from_external(
                    quantized_decode_layered_hidden_states_.data(), quantized_decode_layered_hidden_states_.shape(),
                    quantized_decode_layered_hidden_states_.dtype(), Device::CUDA);
            }
            auto capture_result =
                quantized_decode_graph_.capture(decode_stream,
                                                [&]() -> Status
                                                {
                                                    Tensor graph_logits = FIREFLY_TRY(
                                                        model_->forward(model_input, forward_options));
                                                    return kernels::argmax(graph_logits, next_tokens, context);
                                                });
            if (capture_result)
            {
                quantized_decode_batch_size_ = batch_size;
                quantized_decode_max_blocks_ = max_blocks;
                quantized_decode_context_bucket_ = quantized_context_bucket;
            }
            else
            {
                quantized_decode_batch_size_ = 0;
                quantized_decode_max_blocks_ = 0;
                quantized_decode_context_bucket_ = 0;
            }
        }
        if (!quantized_decode_graph_.empty())
        {
            auto launch_result = quantized_decode_graph_.launch(decode_stream);
            graph_launched = launch_result.has_value();
            if (graph_launched && speculative_decoder_ != nullptr)
                layered_hidden_states = Tensor::from_external(
                    quantized_decode_layered_hidden_states_.data(), quantized_decode_layered_hidden_states_.shape(),
                    quantized_decode_layered_hidden_states_.dtype(), Device::CUDA);
        }
    }
    else if (use_flashinfer_graph)
    {
        std::vector<int> page_counts(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            page_counts[i] = (scratch.context_lens[i] + 16) / 16;
        }
        const bool graph_shape_changed =
            flashinfer_decode_graph_.empty() || flashinfer_decode_batch_size_ != batch_size ||
            flashinfer_decode_max_blocks_ != max_blocks || flashinfer_decode_pages_ != page_counts;
        if (graph_shape_changed)
        {
            flashinfer_decode_graph_ = device::Graph{};
            flashinfer_decode_layered_hidden_states_ = std::move(layered_hidden_states);
            if (speculative_decoder_ != nullptr)
            {
                forward_options.hidden_state_layers = speculative_decoder_->target_hidden_layers();
                forward_options.layered_hidden_state_output = &flashinfer_decode_layered_hidden_states_;
                layered_hidden_states = Tensor::from_external(
                    flashinfer_decode_layered_hidden_states_.data(), flashinfer_decode_layered_hidden_states_.shape(),
                    flashinfer_decode_layered_hidden_states_.dtype(), Device::CUDA);
            }
            auto capture_result = flashinfer_decode_graph_.capture(
                decode_stream,
                [&]() -> Status
                {
                    Tensor graph_logits = FIREFLY_TRY(model_->forward(model_input, forward_options));
                    return kernels::argmax(graph_logits, next_tokens, context);
                });
            if (capture_result)
            {
                flashinfer_decode_batch_size_ = batch_size;
                flashinfer_decode_max_blocks_ = max_blocks;
                flashinfer_decode_pages_ = std::move(page_counts);
            }
            else
            {
                flashinfer_decode_batch_size_ = 0;
                flashinfer_decode_max_blocks_ = 0;
                flashinfer_decode_pages_.clear();
            }
        }
        if (!flashinfer_decode_graph_.empty())
        {
            auto launch_result = flashinfer_decode_graph_.launch(decode_stream);
            graph_launched = launch_result.has_value();
            if (graph_launched && speculative_decoder_ != nullptr)
                layered_hidden_states = Tensor::from_external(
                    flashinfer_decode_layered_hidden_states_.data(), flashinfer_decode_layered_hidden_states_.shape(),
                    flashinfer_decode_layered_hidden_states_.dtype(), Device::CUDA);
        }
    }
    if (!graph_launched)
    {
        Tensor logits = FIREFLY_TRY_CONTEXT(model_->forward(model_input, forward_options),
                                            "execute dynamic decode forward");
        FIREFLY_TRY(kernels::argmax(logits, next_tokens, context));
    }
    if (speculative_decoder_ != nullptr)
        FIREFLY_TRY(speculative_decoder_->capture_target_context(layered_hidden_states, requests, 1, context));
    FIREFLY_LOG_DEBUG("decode", "forward complete batch={} graph={}", batch_size, graph_launched);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    if (gpu_events) cudaEventRecord(gpu_events->fwd_stop, profile_stream);
    auto f2 = std::chrono::high_resolution_clock::now();
#endif
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    if (gpu_events) cudaEventRecord(gpu_events->argmax_stop, profile_stream);
    auto f3 = std::chrono::high_resolution_clock::now();
#endif

    scratch.next_tokens.resize(batch_size);
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(scratch.next_tokens.data(), next_tokens.data(), scratch.next_tokens.size() * sizeof(int),
                        cudaMemcpyDeviceToHost, decode_stream),
        "copy dynamic decode output tokens"));
    FIREFLY_LOG_DEBUG("decode", "synchronize begin batch={} max_context={}", batch_size, max_context_length);
    const cudaError_t sync_error = cudaStreamSynchronize(decode_stream);
    FIREFLY_TRY(device::check_cuda(sync_error, "synchronize dynamic decode output"));
    FIREFLY_LOG_DEBUG("decode", "synchronize complete batch={} max_context={}", batch_size, max_context_length);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    if (gpu_events) cudaEventRecord(gpu_events->d2h_stop, profile_stream);
    auto f4 = std::chrono::high_resolution_clock::now();
#endif

    for (int i = 0; i < batch_size; ++i)
    {
        auto req = requests[i];
        req->generated_tokens.push_back(scratch.next_tokens[i]);
        req->context_len += 1;
        FIREFLY_LOG_DEBUG("decode", "complete request_id={} context={} generated={} token={}", req->id,
                          req->context_len, req->generated_tokens.size(), scratch.next_tokens[i]);
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
            FIREFLY_LOG_INFO("performance",
                             "dynamic decode profile steps=50 prepare_ms={:.3f} forward_ms={:.3f} "
                             "argmax_ms={:.3f} d2h_ms={:.3f} update_ms={:.3f}",
                             fb_prep, fb_forward, fb_argmax, fb_d2h, fb_update);
            FIREFLY_LOG_INFO("performance",
                             "dynamic decode GPU profile steps=50 h2d_ms={:.3f} forward_ms={:.3f} "
                             "argmax_ms={:.3f} d2h_ms={:.3f}",
                             fb_gpu_h2d, fb_gpu_fwd, fb_gpu_argmax, fb_gpu_d2h);
            fb_prep = fb_forward = fb_argmax = fb_d2h = fb_update = 0;
            fb_gpu_h2d = fb_gpu_fwd = fb_gpu_argmax = fb_gpu_d2h = 0;
        }
    }
#endif
    return {};
}
}  // namespace firefly::execution
