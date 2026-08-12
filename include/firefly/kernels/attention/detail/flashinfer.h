#pragma once

#include "firefly/core/tensor.h"

#include <vector>

namespace firefly::kernels::attention_backend
{
/**
 * @brief Builds FlashInfer metadata for uniform paged prefill.
 * @param block_table Device logical-to-physical cache-page table.
 * @param batch_size Number of equal-length prompt rows.
 * @param sequence_length New query tokens per row.
 * @param context_length Total key/value context after the current chunk.
 * @param max_context_blocks Block-table row stride.
 * @param num_qo_heads Number of query/output heads.
 * @param num_kv_heads Number of cached key/value heads.
 * @param head_dim Scalar width of each head.
 * @param stream CUDA stream used for plan setup and metadata copies.
 */
Status prepare_flashinfer_prefill(const int* block_table, int batch_size, int sequence_length, int context_length,
                                  int max_context_blocks, int num_qo_heads, int num_kv_heads, int head_dim,
                                  cudaStream_t stream);
/**
 * @brief Builds FlashInfer metadata for variable-length ragged paged prefill.
 * @param q_indptr Host cumulative query-token offsets.
 * @param kv_indptr Host cumulative KV-page offsets.
 * @param last_page_len Host valid-token count in each row's final page.
 * @param max_context_blocks Block-table row stride.
 * @param num_qo_heads Number of query/output heads.
 * @param num_kv_heads Number of cached key/value heads.
 * @param head_dim Scalar width of each head.
 * @param stream CUDA stream used for plan setup and metadata copies.
 */
Status prepare_flashinfer_prefill_ragged(const std::vector<int>& q_indptr, const std::vector<int>& kv_indptr,
                                         const std::vector<int>& last_page_len, int max_context_blocks,
                                         int num_qo_heads, int num_kv_heads, int head_dim, cudaStream_t stream);
/**
 * @brief Launches a previously prepared FlashInfer paged decode.
 * @param query Single-token query tensor.
 * @param key_cache Model-dtype paged key cache.
 * @param value_cache Model-dtype paged value cache.
 * @param output Preallocated attention output.
 * @param kv_head_count Number of key/value heads per cached token.
 * @param head_dim Scalar width of each attention head.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 * @return `true` when a compatible plan and dtype were available and the kernel launched.
 */
Result<bool> launch_flashinfer_decode(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                      int kv_head_count, int head_dim, float scale, const device::Context& context);
/**
 * @brief Launches FlashInfer attention over contiguous prefill Q/K/V tensors.
 * @param query Dense query tensor.
 * @param key Dense key tensor.
 * @param value Dense value tensor matching `key`.
 * @param output Preallocated attention output.
 * @param kv_head_count Number of key/value heads.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 * @return `true` when FlashInfer supports the supplied shape and dtype and dispatched the kernel.
 */
Result<bool> launch_flashinfer_contiguous_prefill(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                                                  int kv_head_count, float scale, const device::Context& context);
/**
 * @brief Launches FlashInfer-compatible prefill against Firefly's signed-int8 paged cache.
 * @param query Floating-point multi-token query tensor.
 * @param key_cache Signed-int8 paged key cache.
 * @param value_cache Signed-int8 paged value cache.
 * @param scales Floating-point per-token key/value scale cache.
 * @param block_table Device logical-to-physical cache-page table.
 * @param output Preallocated floating-point attention output.
 * @param kv_head_count Number of key/value heads per cached token.
 * @param max_context_blocks Block-table row stride.
 * @param context_length Total key/value context after the current chunk.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 * @return `true` when the compatible quantized prefill implementation launched.
 */
Result<bool> launch_flashinfer_quantized_prefill(Tensor& query, Tensor& key_cache, Tensor& value_cache,
                                                 Tensor& scales, const int* block_table, Tensor& output,
                                                 int kv_head_count, int max_context_blocks, int context_length,
                                                 float scale, const device::Context& context);
/**
 * @brief Launches uniform-length paged FlashInfer prefill using the current plan.
 * @param query Dense multi-token query tensor.
 * @param key_cache Model-dtype paged key cache.
 * @param value_cache Model-dtype paged value cache.
 * @param output Preallocated attention output.
 * @param block_table Device cache page table used by the plan.
 * @param kv_head_count Number of key/value heads per cached token.
 * @param sequence_length New query tokens per row.
 * @param max_context_blocks Block-table row stride.
 * @param context_length Total key/value context after the current chunk.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 * @return `true` when the existing plan matches and the kernel launched.
 */
Result<bool> launch_flashinfer_prefill(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                       const int* block_table, int kv_head_count, int sequence_length,
                                       int max_context_blocks, int context_length, float scale,
                                       const device::Context& context);
/**
 * @brief Launches ragged paged FlashInfer prefill using the current plan.
 * @param query Flattened ragged query tensor.
 * @param key_cache Model-dtype paged key cache.
 * @param value_cache Model-dtype paged value cache.
 * @param output Preallocated flattened attention output.
 * @param block_table Device cache page table used by the plan.
 * @param kv_head_count Number of key/value heads per cached token.
 * @param max_context_blocks Block-table row stride.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 * @return `true` when the existing ragged plan matches and the kernel launched.
 */
Result<bool> launch_flashinfer_prefill_ragged(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                              const int* block_table, int kv_head_count, int max_context_blocks,
                                              float scale, const device::Context& context);
}  // namespace firefly::kernels::attention_backend
