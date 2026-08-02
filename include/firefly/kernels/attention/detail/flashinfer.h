#pragma once

#include "firefly/core/tensor.h"

namespace firefly::kernels::attention_backend
{
bool launch_flashinfer_decode(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                              int kv_head_count, int head_dim, float scale, const device::Context& context);
bool launch_flashinfer_prefill(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                               const int* block_table, int kv_head_count, int sequence_length,
                               int max_context_blocks, int context_length, float scale,
                               const device::Context& context);
}  // namespace firefly::kernels::attention_backend
