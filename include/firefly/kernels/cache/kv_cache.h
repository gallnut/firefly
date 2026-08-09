#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

void append_paged_kv(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                     const int* block_table, const int* context_lens, int max_blocks_per_sequence,
                     Tensor* scale_cache = nullptr,
                     const device::Context& context = {});

// Ragged variant: key/value are flattened (total_tokens, kv_heads, head_dim);
// seq_offsets/lengths describe each sequence's token range, context_lens the
// position where its chunk starts in the cache.
void append_paged_kv_ragged(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                            const int* block_table, const int* seq_offsets, const int* seq_lengths,
                            const int* context_lens, int batch_size, int max_sequence_length,
                            int max_blocks_per_sequence,
                            const device::Context& context = {});

}  // namespace firefly::kernels
