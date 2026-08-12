#pragma once

#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/model/forward_context.h"
#include "firefly/model/qwen/model.h"

namespace firefly::model::qwen
{

/**
 * @brief Per-forward metadata and reusable buffers for Qwen ragged mixed batching.
 *
 * Host vectors classify rows into decode and prefill subsets. Device tensors hold
 * gather/scatter indices and subset buffers that are reused across transformer layers.
 */
struct RaggedForwardState
{
    int  batch = 0; ///< Number of logical sequences in the ragged batch.
    int  total_tokens = 0; ///< Sum of all row token counts.
    int  max_blocks = 0; ///< Block-table row stride.
    int  max_seq_len = 1; ///< Largest current row length.
    int  decode_count = 0; ///< Number of single-token decode rows.
    int  prefill_count = 0; ///< Number of multi-token prefill rows.
    int  prefill_q_rows = 0; ///< Total query tokens belonging to prefill rows.
    bool prefill_contiguous = false; ///< Whether every row is prefill and already contiguous.
    bool prefill_planned = false; ///< Whether FlashInfer accepted the ragged prefill plan.
    int  q_elements_per_token = 0; ///< Query tensor stride per flattened token.
    int  out_elements_per_token = 0; ///< Attention-output stride per flattened token.

    Tensor positions; ///< Flattened token positions consumed by rotary embedding.
    Tensor d_decode_rows; ///< Device indices of rows classified as decode.
    Tensor d_decode_context_lens; ///< Device context lengths for decode rows.
    Tensor d_decode_block_table; ///< Compact device block table for decode rows.
    Tensor decode_q; ///< Gathered query tensor for the decode subset.
    Tensor decode_out; ///< Attention output tensor for the decode subset.
    Tensor prefill_q; ///< Gathered query tensor for the prefill subset.
    Tensor prefill_out; ///< Attention output tensor for the prefill subset.
    Tensor d_prefill_token_indices; ///< Device indices mapping compact prefill tokens to ragged positions.
    Tensor d_last_tokens; ///< Device indices of the last token in every ragged row.

    std::vector<int> host_seq_offsets; ///< Host copy of flattened row starting offsets.
    std::vector<int> host_seq_lengths; ///< Host copy of current row token counts.
    std::vector<int> host_context_lens; ///< Host copy of total context lengths after this forward.
    std::vector<int> decode_rows; ///< Logical row indices classified as decode.
    std::vector<int> decode_context_lens; ///< Context length for every compact decode row.
    std::vector<int> decode_block_tables; ///< Flattened block table compacted to decode rows.
    std::vector<int> prefill_rows; ///< Logical row indices classified as prefill.
    std::vector<int> prefill_q_indptr; ///< CSR-style query offsets for compact prefill rows.
    std::vector<int> prefill_kv_indptr; ///< CSR-style KV-page offsets for compact prefill rows.
    std::vector<int> prefill_last_page_len; ///< Number of valid tokens in each row's final KV page.
    std::vector<int> prefill_token_indices; ///< Ragged token indices gathered into compact prefill storage.
};

/**
 * @brief Classifies ragged rows and allocates gather/scatter and attention-planning buffers.
 * @param model Qwen model whose dimensions and attention weights determine buffer shapes.
 * @param input Flattened token input, cache view, and device ragged-row metadata.
 * @param options Forward controls including CUDA context and attention backend hints.
 * @return Initialized state reused for every layer of the current forward pass.
 */
Result<RaggedForwardState> prepare_ragged_forward(const QwenModel& model, const ModelInput& input,
                                                  const ForwardOptions& options);

/**
 * @brief Executes decode and prefill attention subsets for one Qwen layer and scatters results.
 * @param model Qwen model providing head dimensions and layer cache layout.
 * @param input Forward input containing paged-cache and ragged-row metadata.
 * @param state Prepared reusable subset buffers and host planning metadata.
 * @param layer_index Zero-based transformer layer whose cache is updated and read.
 * @param q Flattened query tensor, used as source and temporary subset storage.
 * @param attn_out Flattened destination receiving attention output in original ragged order.
 * @param scale Multiplicative query-key score scale.
 * @param options CUDA context and execution controls for the current forward.
 */
Status run_ragged_attention(const QwenModel& model, const ModelInput& input, RaggedForwardState& state,
                            int layer_index, Tensor& q, Tensor& attn_out, float scale, const ForwardOptions& options);

/**
 * @brief Gathers each row's final hidden state, applies final normalization, and computes logits.
 * @param model Qwen model providing normalization and language-head weights.
 * @param state Prepared state containing device indices of final row tokens.
 * @param hidden_states Flattened final transformer hidden states.
 * @param options Controls whether logits and hidden-state side outputs are materialized.
 * @return Per-row logits, or an empty tensor when `compute_logits` is false.
 */
Result<Tensor> finish_ragged_forward(const QwenModel& model, RaggedForwardState& state, Tensor& hidden_states,
                                     const ForwardOptions& options);

}  // namespace firefly::model::qwen
