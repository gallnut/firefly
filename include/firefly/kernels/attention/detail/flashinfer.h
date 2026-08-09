#pragma once

#include "firefly/core/tensor.h"

#include <vector>

namespace firefly::kernels::attention_backend
{
void prepare_flashinfer_prefill(const int* block_table, int batch_size, int sequence_length, int context_length,
                                int max_context_blocks, int num_qo_heads, int num_kv_heads, int head_dim,
                                cudaStream_t stream);
void prepare_flashinfer_prefill_ragged(const std::vector<int>& q_indptr, const std::vector<int>& kv_indptr,
                                       const std::vector<int>& last_page_len, int max_context_blocks,
                                       int num_qo_heads, int num_kv_heads, int head_dim, cudaStream_t stream);
bool launch_flashinfer_decode(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output, int kv_head_count,
                              int head_dim, float scale, const device::Context& context);
bool launch_flashinfer_contiguous_prefill(Tensor& query, Tensor& key, Tensor& value, Tensor& output, int kv_head_count,
                                          float scale, const device::Context& context);
bool launch_flashinfer_quantized_prefill(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& scales,
                                         const int* block_table, Tensor& output, int kv_head_count,
                                         int max_context_blocks, int context_length, float scale,
                                         const device::Context& context);
bool launch_flashinfer_prefill(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                               const int* block_table, int kv_head_count, int sequence_length, int max_context_blocks,
                               int context_length, float scale, const device::Context& context);
bool launch_flashinfer_prefill_ragged(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                      const int* block_table, int kv_head_count, int max_context_blocks, float scale,
                                      const device::Context& context);
}  // namespace firefly::kernels::attention_backend
