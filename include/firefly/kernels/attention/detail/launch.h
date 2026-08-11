#pragma once

#include "firefly/kernels/attention/attention.h"

namespace firefly::kernels::attention_detail
{
struct DecodeConfig
{
    bool force_single = false;
    bool force_split = false;
    int  split_size = 256;
};

void reserve_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim, int split_size);
void launch_contiguous(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                       float scale, const device::Context& context);
void launch_paged_prefill(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                          float scale, const device::Context& context);
void launch_paged_decode(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                         const DecodeConfig& decode_config, float scale, const device::Context& context);
void launch_paged(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                  const DecodeConfig& decode_config, float scale, const device::Context& context);
void launch_quantized_paged(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                            const DecodeConfig& decode_config, float scale, const device::Context& context);
}  // namespace firefly::kernels::attention_detail
