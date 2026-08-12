#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

/**
 * @brief Appends a uniform dense token block to paged key/value cache tensors.
 * @param key Dense new keys shaped `[batch, sequence, kv_heads, head_dim]`.
 * @param value Dense new values matching `key`.
 * @param key_cache Mutable paged key storage in model dtype or signed int8.
 * @param value_cache Mutable paged value storage matching `key_cache`.
 * @param block_table Device logical-to-physical block mapping for each batch row.
 * @param context_lens Device insertion offset preceding the current token block.
 * @param max_blocks_per_sequence Block-table row stride.
 * @param scale_cache Optional quantization scale cache; when non-null, K/V are quantized to int8.
 * @param context CUDA stream used for asynchronous cache writes.
 * @note Block table and context lengths are device pointers. Execution is asynchronous.
 */
Status append_paged_kv(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                       const int* block_table, const int* context_lens, int max_blocks_per_sequence,
                       Tensor* scale_cache = nullptr, const device::Context& context = {});

/**
 * @brief Appends flattened ragged key/value rows to each sequence's paged cache positions.
 * @param key Flattened keys shaped `[total_tokens, kv_heads, head_dim]`.
 * @param value Flattened values matching `key`.
 * @param key_cache Mutable paged key storage.
 * @param value_cache Mutable paged value storage.
 * @param block_table Device logical-to-physical block mapping for each batch row.
 * @param seq_offsets Device start offset per sequence.
 * @param seq_lengths Device token count per sequence.
 * @param context_lens Device cache insertion position per sequence.
 * @param batch_size Number of ragged rows.
 * @param max_sequence_length Largest query-token count in any row.
 * @param max_blocks_per_sequence Block-table row stride.
 * @param context CUDA stream used for asynchronous cache writes.
 * @note This overload currently writes model-dtype cache tensors without quantization scales.
 */
Status append_paged_kv_ragged(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                              const int* block_table, const int* seq_offsets, const int* seq_lengths,
                              const int* context_lens, int batch_size, int max_sequence_length,
                              int max_blocks_per_sequence, const device::Context& context = {});

}  // namespace firefly::kernels
