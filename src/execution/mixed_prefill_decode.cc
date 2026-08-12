#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "firefly/execution/engine.h"
#include "firefly/device/error.h"
#include "firefly/kernels/sampling/argmax.h"

namespace firefly::execution
{

Status Engine::process_mixed_batch(const std::vector<scheduler::SequencePtr>& requests,
                                   const device::Context& context)
{
    if (requests.empty()) return {};
    cudaStream_t stream = context.stream();
    const int    batch_size = static_cast<int>(requests.size());
    int          max_blocks = 0;
    for (const auto& request : requests)
        max_blocks = std::max(max_blocks, static_cast<int>(request->block_table.size()));

    std::vector<int> flat_tokens;
    std::vector<int> seq_offsets(batch_size + 1);
    std::vector<int> seq_lengths(batch_size);
    std::vector<int> context_lens(batch_size);
    std::vector<int> state_slots(batch_size);
    std::vector<int> block_table(batch_size * max_blocks, decode_scratch_first_block_);
    seq_offsets[0] = 0;

    for (int i = 0; i < batch_size; ++i)
    {
        const auto& request = requests[i];
        int         chunk = 1;
        if (request->generated_tokens.empty())
        {
            const int unmatched = static_cast<int>(request->prompt_tokens.size()) - request->context_len;
            chunk = std::min(unmatched, options_.max_prefill_chunk_size);
            for (int token = 0; token < chunk; ++token)
                flat_tokens.push_back(request->prompt_tokens[request->context_len + token]);
        }
        else
        {
            flat_tokens.push_back(request->generated_tokens.back());
        }

        seq_offsets[i + 1] = seq_offsets[i] + chunk;
        seq_lengths[i] = chunk;
        context_lens[i] = request->context_len;
        state_slots[i] = request->state_slot;
        for (int block = 0; block < static_cast<int>(request->block_table.size()); ++block)
            block_table[i * max_blocks + block] = request->block_table[block];
    }

    const int total_tokens = static_cast<int>(flat_tokens.size());
    Tensor d_input = FIREFLY_TRY(Tensor::create({total_tokens, 1}, DType::I32, Device::CUDA, context));
    Tensor d_context_lens = FIREFLY_TRY(Tensor::create({batch_size}, DType::I32, Device::CUDA, context));
    Tensor d_seq_offsets = FIREFLY_TRY(Tensor::create({batch_size + 1}, DType::I32, Device::CUDA, context));
    Tensor d_seq_lengths = FIREFLY_TRY(Tensor::create({batch_size}, DType::I32, Device::CUDA, context));
    Tensor d_state_slots = FIREFLY_TRY(Tensor::create({batch_size}, DType::I32, Device::CUDA, context));
    Tensor d_block_table = FIREFLY_TRY(Tensor::create(
        {static_cast<int64_t>(batch_size) * max_blocks}, DType::I32, Device::CUDA, context));
    Tensor d_next_tokens = FIREFLY_TRY(Tensor::create({batch_size}, DType::I32, Device::CUDA, context));

    cudaMemcpyAsync(d_input.data(), flat_tokens.data(), flat_tokens.size() * sizeof(int), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_context_lens.data(), context_lens.data(), context_lens.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seq_offsets.data(), seq_offsets.data(), seq_offsets.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seq_lengths.data(), seq_lengths.data(), seq_lengths.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_state_slots.data(), state_slots.data(), state_slots.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_table.data(), block_table.data(), block_table.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    const cudaError_t copy_status = cudaStreamSynchronize(stream);
    FIREFLY_TRY(device::check_cuda(copy_status, "copy mixed batch inputs to device"));

    model::ModelInput model_input{
        d_input,
        d_context_lens,
        {k_caches_, v_caches_, kv_scale_caches_, static_cast<int*>(d_block_table.data()), max_blocks},
        static_cast<int*>(d_state_slots.data()),
        static_cast<const int*>(d_seq_offsets.data()),
        static_cast<const int*>(d_seq_lengths.data()),
    };
    model::ForwardOptions forward_options{.context = context};
    Tensor layered_hidden_states;
    if (speculative_decoder_ != nullptr)
        FIREFLY_TRY(speculative_decoder_->configure_target_ragged_forward(forward_options, layered_hidden_states,
                                                                          total_tokens));

    Tensor logits = FIREFLY_TRY_CONTEXT(model_->forward(model_input, forward_options), "execute mixed batch forward");
    if (speculative_decoder_ != nullptr)
        FIREFLY_TRY(speculative_decoder_->capture_target_context_ragged(
            layered_hidden_states, requests, std::span(seq_offsets).first(batch_size), seq_lengths, context));
    FIREFLY_TRY(kernels::argmax(logits, d_next_tokens, context));
    std::vector<int> host_next_tokens(batch_size);
    cudaMemcpyAsync(host_next_tokens.data(), d_next_tokens.data(), host_next_tokens.size() * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    const cudaError_t sync_error = cudaStreamSynchronize(stream);
    FIREFLY_TRY(device::check_cuda(sync_error, "copy mixed batch outputs to host"));

    for (int i = 0; i < batch_size; ++i)
    {
        auto request = requests[i];
        if (request->generated_tokens.empty())
        {
            request->context_len += seq_lengths[i];
            if (request->context_len == static_cast<int>(request->prompt_tokens.size()))
                request->generated_tokens.push_back(host_next_tokens[i]);
        }
        else
        {
            request->generated_tokens.push_back(host_next_tokens[i]);
            request->context_len += 1;
        }
    }
    return {};
}

}  // namespace firefly::execution
