#pragma once

#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/model/forward_context.h"
#include "firefly/model/qwen/model.h"

namespace firefly::model::qwen
{

// Per-forward state for the ragged (mixed prefill + decode) attention path.
// Metadata and device buffers are prepared once and reused across layers.
struct RaggedForwardState
{
    int  batch = 0;
    int  total_tokens = 0;
    int  max_blocks = 0;
    int  max_seq_len = 1;
    int  decode_count = 0;
    int  prefill_count = 0;
    int  prefill_q_rows = 0;
    bool prefill_contiguous = false;
    bool prefill_planned = false;
    int  q_elements_per_token = 0;
    int  out_elements_per_token = 0;

    Tensor positions;
    Tensor d_decode_rows;
    Tensor d_decode_context_lens;
    Tensor d_decode_block_table;
    Tensor decode_q;
    Tensor decode_out;
    Tensor prefill_q;
    Tensor prefill_out;
    Tensor d_prefill_token_indices;
    Tensor d_last_tokens;

    std::vector<int> host_seq_offsets;
    std::vector<int> host_seq_lengths;
    std::vector<int> host_context_lens;
    std::vector<int> decode_rows;
    std::vector<int> decode_context_lens;
    std::vector<int> decode_block_tables;
    std::vector<int> prefill_rows;
    std::vector<int> prefill_q_indptr;
    std::vector<int> prefill_kv_indptr;
    std::vector<int> prefill_last_page_len;
    std::vector<int> prefill_token_indices;
};

RaggedForwardState prepare_ragged_forward(const QwenModel& model, const ModelInput& input,
                                          const ForwardOptions& options);

void run_ragged_attention(const QwenModel& model, const ModelInput& input, RaggedForwardState& state,
                          int layer_index, Tensor& q, Tensor& attn_out, float scale, const ForwardOptions& options);

Tensor finish_ragged_forward(const QwenModel& model, RaggedForwardState& state, Tensor& hidden_states,
                             const ForwardOptions& options);

}  // namespace firefly::model::qwen
