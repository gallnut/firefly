#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "firefly/core/logging.h"
#include "firefly/execution/engine.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/sampling/argmax.h"

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
#endif
}  // namespace

void Engine::process_prefill(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context)
{
    cudaStream_t                                                          decode_stream = context.stream();
    const device::Context&                                                decode_context = context;
    std::map<int, std::vector<scheduler::SequencePtr>, std::greater<int>> prefill_groups;
    for (auto req : requests)
    {
        int unmatched_tokens = req->prompt_tokens.size() - req->context_len;
        int seq_len = std::min(unmatched_tokens, options_.max_prefill_chunk_size);
        if (seq_len > 0)
        {
            prefill_groups[seq_len].push_back(req);
        }
    }

    for (auto& [seq_len, group] : prefill_groups)
    {
        int  group_size = group.size();
        int  max_blocks = 0;
        int  min_context_len = std::numeric_limits<int>::max();
        bool any_prompt_finished = false;

        std::vector<int> h_input_ids(group_size * seq_len);
        std::vector<int> h_context_lens(group_size);
        for (int i = 0; i < group_size; ++i)
        {
            auto req = group[i];
            std::copy_n(req->prompt_tokens.begin() + req->context_len, seq_len, h_input_ids.begin() + i * seq_len);
            h_context_lens[i] = req->context_len;
            min_context_len = std::min(min_context_len, req->context_len);
            any_prompt_finished = any_prompt_finished || (req->context_len + seq_len == (int)req->prompt_tokens.size());
            max_blocks = std::max<int>(max_blocks, req->block_table.size());
        }

        for (const auto& req : group)
        {
            FIREFLY_LOG_DEBUG("prefill",
                              "begin request_id={} batch={} seq_len={} context={} prompt={} state_slot={} blocks={}",
                              req->id, group_size, seq_len, req->context_len, req->prompt_tokens.size(),
                              req->state_slot, req->block_table.size());
        }

        Tensor d_input({(long)group_size, seq_len}, DType::I32, Device::CUDA, decode_context);
        cudaMemcpyAsync(d_input.data(), h_input_ids.data(), h_input_ids.size() * sizeof(int), cudaMemcpyHostToDevice,
                        decode_stream);

        Tensor d_context_lens({(long)group_size}, DType::I32, Device::CUDA, decode_context);
        cudaMemcpyAsync(d_context_lens.data(), h_context_lens.data(), h_context_lens.size() * sizeof(int),
                        cudaMemcpyHostToDevice, decode_stream);
        std::vector<int> h_state_slots(group_size);
        for (int i = 0; i < group_size; ++i) h_state_slots[i] = group[i]->state_slot;
        Tensor d_state_slots({(long)group_size}, DType::I32, Device::CUDA, decode_context);
        cudaMemcpyAsync(d_state_slots.data(), h_state_slots.data(), h_state_slots.size() * sizeof(int),
                        cudaMemcpyHostToDevice, decode_stream);

        std::vector<int> h_block_table(group_size * max_blocks, decode_scratch_first_block_);
        for (int i = 0; i < group_size; ++i)
        {
            auto& blocks = group[i]->block_table;
            std::copy(blocks.begin(), blocks.end(), h_block_table.begin() + i * max_blocks);
        }

        Tensor d_block_table_tensor({(long)(group_size * max_blocks)}, DType::I32, Device::CUDA, decode_context);
        int*   d_block_table = static_cast<int*>(d_block_table_tensor.data());
        cudaMemcpyAsync(d_block_table, h_block_table.data(), h_block_table.size() * sizeof(int), cudaMemcpyHostToDevice,
                        decode_stream);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        if (profile_prefill_enabled())
        {
            auto& events = prefill_profile_events();
            cudaEventRecord(events.start, decode_stream);
        }
#endif

        copy_prefix_cow_blocks(group, decode_context);
        kernels::prepare_attention_prefill(d_block_table, group_size, seq_len, min_context_len, max_blocks,
                                           config_.num_attention_heads, runtime_requirements_.kv_cache_head_count,
                                           runtime_requirements_.kv_cache_head_dim, decode_context);

        model::ModelInput model_input{d_input,
                                      d_context_lens,
                                      {k_caches_, v_caches_, kv_scale_caches_, d_block_table, max_blocks},
                                      static_cast<int*>(d_state_slots.data())};
        model::ForwardOptions forward_options{
            .context = decode_context, .compute_logits = any_prompt_finished, .min_context_len = min_context_len};
        Tensor logits = model_->forward(model_input, forward_options);

        FIREFLY_LOG_DEBUG("prefill", "forward launched batch={} seq_len={} min_context={} prompt_finished={}",
                          group_size, seq_len, min_context_len, any_prompt_finished);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        if (profile_prefill_enabled())
        {
            auto& events = prefill_profile_events();
            cudaEventRecord(events.stop, decode_stream);
            cudaEventSynchronize(events.stop);
            float elapsed_ms = 0.0f;
            cudaEventElapsedTime(&elapsed_ms, events.start, events.stop);
            FIREFLY_LOG_INFO("performance",
                             "prefill profile batch_size={} sequence_length={} prompt_finished={} time_ms={:.3f} "
                             "tokens_per_second={:.2f}",
                             group_size, seq_len, any_prompt_finished, elapsed_ms,
                             group_size * seq_len * 1000.0f / std::max(elapsed_ms, 1e-3f));
        }
#endif

        for (auto& req : group)
        {
            req->context_len += seq_len;
            FIREFLY_LOG_DEBUG("prefill", "complete request_id={} context={} remaining={} generated={}", req->id,
                              req->context_len, req->prompt_tokens.size() - req->context_len,
                              req->generated_tokens.size());
        }

        if (any_prompt_finished)
        {
            Tensor next_tokens({(long)group_size}, DType::I32, Device::CUDA, decode_context);
            kernels::argmax(logits, next_tokens, decode_context);
            std::vector<int> h_next_tokens(group_size);
            cudaMemcpyAsync(h_next_tokens.data(), next_tokens.data(), h_next_tokens.size() * sizeof(int),
                            cudaMemcpyDeviceToHost, decode_stream);
            FIREFLY_LOG_DEBUG("prefill", "synchronize begin batch={} seq_len={} context={}", group_size, seq_len,
                              min_context_len);
            const cudaError_t sync_error = cudaStreamSynchronize(decode_stream);
            if (sync_error != cudaSuccess)
            {
                throw std::runtime_error(std::string("prefill synchronization failed: ") +
                                         cudaGetErrorString(sync_error));
            }
            FIREFLY_LOG_DEBUG("prefill", "synchronize complete batch={} seq_len={} context={}", group_size, seq_len,
                              min_context_len);

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
}
}  // namespace firefly::execution
