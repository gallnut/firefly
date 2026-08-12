#pragma once

#include "firefly/kernels/attention/attention.h"

namespace firefly::kernels::attention_detail
{
/** @brief Internal paged-decode launch policy derived from environment and context length. */
struct DecodeConfig
{
    bool force_single = false; ///< Force one-pass decode even for long contexts.
    bool force_split = false; ///< Force split-KV decode regardless of context length.
    int  split_size = 256; ///< Cache-token span processed by one split partial reduction.
};

/**
 * @brief Reserves internal custom-kernel decode scratch for a maximum shape.
 * @param batch_size Maximum decode rows.
 * @param num_heads Maximum query-head count.
 * @param max_context_blocks Maximum logical cache pages in one row.
 * @param head_dim Attention head width.
 * @param split_size Cache-token span processed by each split partial.
 */
Status reserve_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim, int split_size);
/**
 * @brief Launches custom attention over contiguous K/V tensors.
 * @param query Mutable query tensor shaped `[batch, query_tokens, query_heads, head_dim]`.
 * @param key Contiguous key tensor.
 * @param value Contiguous value tensor matching `key` token and head dimensions.
 * @param output Preallocated attention output matching query rows and heads.
 * @param options Layout and causal-mask policy.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 */
Status launch_contiguous(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                         float scale, const device::Context& context);
/**
 * @brief Launches custom paged multi-token prefill attention.
 * @param query Multi-token query tensor.
 * @param key Paged key-cache tensor.
 * @param value Paged value-cache tensor.
 * @param output Preallocated attention output.
 * @param options Device block table, context lengths, head count, and prefill metadata.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 */
Status launch_paged_prefill(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                            const AttentionOptions& options, float scale, const device::Context& context);
/**
 * @brief Launches custom single-token paged decode with one-pass or split reduction.
 * @param query Single-token query tensor.
 * @param key Paged key-cache tensor.
 * @param value Paged value-cache tensor.
 * @param output Preallocated single-token attention output.
 * @param options Device cache metadata and head count.
 * @param decode_config One-pass versus split-KV launch policy.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 */
Status launch_paged_decode(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                           const AttentionOptions& options, const DecodeConfig& decode_config, float scale,
                           const device::Context& context);
/**
 * @brief Selects paged prefill or decode from the query sequence length.
 * @param query Query tensor whose sequence extent selects the launch path.
 * @param key Paged key-cache tensor.
 * @param value Paged value-cache tensor.
 * @param output Preallocated attention output.
 * @param options Device cache metadata and backend hints.
 * @param decode_config Decode launch policy used for single-token queries.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 */
Status launch_paged(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                    const DecodeConfig& decode_config, float scale, const device::Context& context);
/**
 * @brief Launches quantized paged attention with optional pre-quantized query input.
 * @param query Floating-point query tensor used when no valid quantized query is supplied.
 * @param key Signed-int8 paged key-cache tensor.
 * @param value Signed-int8 paged value-cache tensor.
 * @param output Preallocated floating-point attention output.
 * @param options Cache scales, device metadata, and optional quantized query tensors.
 * @param decode_config Decode launch policy used for single-token queries.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 */
Status launch_quantized_paged(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                              const AttentionOptions& options, const DecodeConfig& decode_config, float scale,
                              const device::Context& context);
}  // namespace firefly::kernels::attention_detail
