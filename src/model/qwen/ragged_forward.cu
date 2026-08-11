#include "firefly/model/qwen/ragged_forward.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/attention/ragged_attention.h"
#include "firefly/kernels/transformer/linear.h"
#include "firefly/kernels/transformer/rms_norm.h"

namespace firefly::model::qwen
{
namespace
{
constexpr int page_size = 16;
}  // namespace

RaggedForwardState prepare_ragged_forward(const QwenModel& model, const ModelInput& input,
                                          const ForwardOptions& options)
{
    RaggedForwardState state;
    state.batch = static_cast<int>(input.context_lens.numel());
    state.total_tokens = static_cast<int>(input.input_ids.numel());
    state.max_blocks = input.kv_cache.max_blocks_per_sequence;
    state.q_elements_per_token = model.config.num_attention_heads * model.config.head_dim;
    state.out_elements_per_token = model.config.num_attention_heads * model.config.head_dim;

    const int* seq_offsets = input.seq_offsets;
    const int* seq_lengths = input.seq_lengths;
    const int* context_lens = static_cast<const int*>(input.context_lens.data());

    state.host_seq_offsets.resize(state.batch + 1);
    state.host_seq_lengths.resize(state.batch);
    cudaMemcpy(state.host_seq_offsets.data(), seq_offsets, (state.batch + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(state.host_seq_lengths.data(), seq_lengths, state.batch * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < state.batch; ++i)
    {
        state.max_seq_len = std::max(state.max_seq_len, state.host_seq_lengths[i]);
        if (state.host_seq_lengths[i] == 1) ++state.decode_count;
    }

    state.positions = Tensor({state.total_tokens}, DType::I32, Device::CUDA, options.context);
    kernels::fill_ragged_positions(state.positions, seq_offsets, seq_lengths, context_lens, state.batch,
                                   state.total_tokens, options.context);

    state.host_context_lens.resize(state.batch);
    std::vector<int> host_block_table(state.batch * state.max_blocks);
    cudaMemcpy(state.host_context_lens.data(), context_lens, state.batch * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_block_table.data(), input.kv_cache.block_table, host_block_table.size() * sizeof(int),
               cudaMemcpyDeviceToHost);

    state.decode_rows.reserve(state.decode_count);
    state.decode_context_lens.reserve(state.decode_count);
    state.decode_block_tables.reserve(state.decode_count * state.max_blocks);
    state.prefill_q_indptr.push_back(0);
    state.prefill_kv_indptr.push_back(0);
    for (int i = 0; i < state.batch; ++i)
    {
        if (state.host_seq_lengths[i] == 1)
        {
            state.decode_rows.push_back(i);
            state.decode_context_lens.push_back(state.host_context_lens[i]);
            for (int b = 0; b < state.max_blocks; ++b)
                state.decode_block_tables.push_back(host_block_table[i * state.max_blocks + b]);
            continue;
        }

        state.prefill_rows.push_back(i);
        const int row_len = state.host_seq_lengths[i];
        const int total_len = state.host_context_lens[i] + row_len;
        const int pages = (total_len + page_size - 1) / page_size;
        state.prefill_q_indptr.push_back(state.prefill_q_indptr.back() + row_len);
        state.prefill_kv_indptr.push_back(state.prefill_kv_indptr.back() + pages);
        state.prefill_last_page_len.push_back((total_len - 1) % page_size + 1);
        for (int t = 0; t < row_len; ++t)
            state.prefill_token_indices.push_back(state.host_seq_offsets[i] + t);
    }
    state.prefill_count = static_cast<int>(state.prefill_rows.size());
    state.prefill_q_rows = state.prefill_q_indptr.empty() ? 0 : state.prefill_q_indptr.back();
    state.prefill_contiguous = state.prefill_count == state.batch;

    if (state.prefill_count > 0)
    {
        state.prefill_planned = kernels::prepare_attention_prefill_ragged(
            state.prefill_q_indptr, state.prefill_kv_indptr, state.prefill_last_page_len, state.max_blocks,
            model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim,
            options.context);
    }

    state.d_decode_rows = Tensor({state.decode_count}, DType::I32, Device::CUDA, options.context);
    cudaMemcpyAsync(state.d_decode_rows.data(), state.decode_rows.data(), state.decode_rows.size() * sizeof(int),
                    cudaMemcpyHostToDevice, options.context.stream());
    state.d_decode_context_lens = Tensor({state.decode_count}, DType::I32, Device::CUDA, options.context);
    state.d_decode_block_table = Tensor({static_cast<int64_t>(state.decode_count) * state.max_blocks}, DType::I32,
                                        Device::CUDA, options.context);
    cudaMemcpyAsync(state.d_decode_context_lens.data(), state.decode_context_lens.data(),
                    state.decode_context_lens.size() * sizeof(int), cudaMemcpyHostToDevice, options.context.stream());
    cudaMemcpyAsync(state.d_decode_block_table.data(), state.decode_block_tables.data(),
                    state.decode_block_tables.size() * sizeof(int), cudaMemcpyHostToDevice,
                    options.context.stream());

    state.decode_q = Tensor({(long)state.decode_count, 1, (long)model.config.num_attention_heads,
                             (long)model.config.head_dim},
                            model.config.dtype, Device::CUDA, options.context);
    state.decode_out = Tensor({(long)state.decode_count, 1,
                               (long)(model.config.num_attention_heads * model.config.head_dim)},
                              model.config.dtype, Device::CUDA, options.context);
    if (state.prefill_count > 0 && !state.prefill_contiguous)
    {
        state.prefill_q = Tensor({(long)state.prefill_q_rows, 1, (long)model.config.num_attention_heads,
                                  (long)model.config.head_dim},
                                 model.config.dtype, Device::CUDA, options.context);
        state.prefill_out = Tensor(
            {(long)state.prefill_q_rows, 1, (long)(model.config.num_attention_heads * model.config.head_dim)},
            model.config.dtype, Device::CUDA, options.context);
        state.d_prefill_token_indices = Tensor({(long)state.prefill_q_rows}, DType::I32, Device::CUDA,
                                               options.context);
        cudaMemcpyAsync(state.d_prefill_token_indices.data(), state.prefill_token_indices.data(),
                        state.prefill_token_indices.size() * sizeof(int), cudaMemcpyHostToDevice,
                        options.context.stream());
    }
    state.d_last_tokens = Tensor({state.batch}, DType::I32, Device::CUDA, options.context);

    return state;
}

void run_ragged_attention(const QwenModel& model, const ModelInput& input, RaggedForwardState& state,
                          int layer_index, Tensor& q, Tensor& attn_out, float scale, const ForwardOptions& options)
{
    const int* seq_offsets = input.seq_offsets;

    if (state.decode_count > 0)
    {
        kernels::gather_decode_tokens(q, state.decode_q, seq_offsets,
                                      static_cast<const int*>(state.d_decode_rows.data()), state.decode_count,
                                      state.q_elements_per_token, options.context);
        kernels::prepare_attention_decode(state.decode_context_lens.data(), state.decode_block_tables.data(),
                                          state.decode_count, state.max_blocks, model.config.num_attention_heads,
                                          model.config.num_key_value_heads, model.config.head_dim, options.context);
        kernels::AttentionOptions decode_options{
            .backend = kernels::AttentionBackend::FlashInfer,
            .block_table = static_cast<const int*>(state.d_decode_block_table.data()),
            .context_lengths = static_cast<const int*>(state.d_decode_context_lens.data()),
            .kv_head_count = model.config.num_key_value_heads,
            .max_context_blocks = state.max_blocks,
        };
        kernels::attention(state.decode_q, input.kv_cache.key_layers[layer_index],
                           input.kv_cache.value_layers[layer_index], state.decode_out, decode_options,
                           options.context);
        kernels::scatter_decode_tokens(state.decode_out, attn_out, seq_offsets,
                                       static_cast<const int*>(state.d_decode_rows.data()), state.decode_count,
                                       state.out_elements_per_token, options.context);
    }

    bool prefill_launched = false;
    if (state.prefill_count > 0 && state.prefill_planned)
    {
        Tensor batch_q;
        Tensor batch_out;
        if (state.prefill_contiguous)
        {
            batch_q = Tensor::from_external(q.data(),
                                            {state.total_tokens, 1, model.config.num_attention_heads,
                                             model.config.head_dim},
                                            model.config.dtype, Device::CUDA);
            batch_out = Tensor::from_external(attn_out.data(),
                                              {state.total_tokens, 1,
                                               model.config.num_attention_heads * model.config.head_dim},
                                              model.config.dtype, Device::CUDA);
        }
        else
        {
            kernels::gather_prefill_tokens(q, state.prefill_q,
                                           static_cast<const int*>(state.d_prefill_token_indices.data()),
                                           state.prefill_q_rows, state.q_elements_per_token, options.context);
            batch_q = Tensor::from_external(state.prefill_q.data(),
                                            {state.prefill_q_rows, 1, model.config.num_attention_heads,
                                             model.config.head_dim},
                                            model.config.dtype, Device::CUDA);
            batch_out = Tensor::from_external(
                state.prefill_out.data(),
                {state.prefill_q_rows, 1, model.config.num_attention_heads * model.config.head_dim},
                model.config.dtype, Device::CUDA);
        }
        prefill_launched = kernels::launch_attention_prefill_ragged(
            batch_q, input.kv_cache.key_layers[layer_index], input.kv_cache.value_layers[layer_index], batch_out,
            input.kv_cache.block_table, model.config.num_key_value_heads, state.max_blocks, scale, options.context);
        if (prefill_launched && !state.prefill_contiguous)
        {
            kernels::scatter_prefill_tokens(state.prefill_out, attn_out,
                                            static_cast<const int*>(state.d_prefill_token_indices.data()),
                                            state.prefill_q_rows, state.out_elements_per_token, options.context);
        }
    }
    if (!prefill_launched)
    {
        for (int row : state.prefill_rows)
        {
            const int token_start = state.host_seq_offsets[row];
            const int row_tokens = state.host_seq_lengths[row];
            Tensor row_q = Tensor::from_external(
                static_cast<std::byte*>(q.data()) +
                    (int64_t)token_start * state.q_elements_per_token * dtype_size(model.config.dtype),
                {1, row_tokens, model.config.num_attention_heads, model.config.head_dim}, model.config.dtype,
                Device::CUDA);
            Tensor row_out = Tensor::from_external(
                static_cast<std::byte*>(attn_out.data()) +
                    (int64_t)token_start * state.out_elements_per_token * dtype_size(model.config.dtype),
                {1, row_tokens, model.config.num_attention_heads * model.config.head_dim}, model.config.dtype,
                Device::CUDA);
            kernels::AttentionOptions row_options{
                .backend = kernels::AttentionBackend::FlashInfer,
                .block_table = input.kv_cache.block_table + (int64_t)row * state.max_blocks,
                .context_lengths = static_cast<const int*>(input.context_lens.data()) + row,
                .kv_head_count = model.config.num_key_value_heads,
                .max_context_blocks = state.max_blocks,
                .prefill_context_length = state.host_context_lens[row],
            };
            kernels::attention(row_q, input.kv_cache.key_layers[layer_index],
                               input.kv_cache.value_layers[layer_index], row_out, row_options, options.context);
        }
    }
}

Tensor finish_ragged_forward(const QwenModel& model, RaggedForwardState& state, Tensor& hidden_states,
                             const ForwardOptions& options)
{
    Tensor last_hidden({(long)state.batch, 1, (long)model.config.hidden_size}, model.config.dtype, Device::CUDA,
                       options.context);
    Tensor final_norm_out({(long)state.batch, 1, (long)model.config.hidden_size}, model.config.dtype, Device::CUDA,
                          options.context);
    std::vector<int> last_tokens_host(state.batch);
    for (int row = 0; row < state.batch; ++row)
        last_tokens_host[row] = state.host_seq_offsets[row] + state.host_seq_lengths[row] - 1;
    cudaMemcpyAsync(state.d_last_tokens.data(), last_tokens_host.data(), last_tokens_host.size() * sizeof(int),
                    cudaMemcpyHostToDevice, options.context.stream());
    kernels::gather_last_hidden(hidden_states, last_hidden, model.config.hidden_size,
                                static_cast<const int*>(state.d_last_tokens.data()), state.batch, options.context);
    kernels::rms_norm(last_hidden, model.norm, final_norm_out, model.config.rms_norm_eps, options.context);
    Tensor logits({(long)state.batch, 1, (long)model.config.vocab_size}, model.config.dtype, Device::CUDA,
                  options.context);
    kernels::matmul(final_norm_out, model.lm_head, logits, options.context);
    return logits;
}

}  // namespace firefly::model::qwen
