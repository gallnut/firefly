#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

/**
 * @brief Fills one absolute position per flattened ragged token from device row metadata.
 * @param positions Preallocated integer output with `total_tokens` entries.
 * @param seq_offsets Device starting offset of each ragged row.
 * @param seq_lengths Device query-token count of each row.
 * @param context_lens Device total context length of each row after the current forward.
 * @param batch Number of ragged rows.
 * @param total_tokens Sum of row query-token counts.
 * @param context CUDA stream used for asynchronous execution.
 */
Status fill_ragged_positions(Tensor& positions, const int* seq_offsets, const int* seq_lengths,
                             const int* context_lens, int batch, int total_tokens,
                             const device::Context& context);

/**
 * @brief Gathers one flattened token row for each selected decode sequence.
 * @param source Flattened ragged source tensor.
 * @param destination Compact output with `decode_count` token rows.
 * @param seq_offsets Device starting offset of each logical row.
 * @param decode_rows Device logical row indices selected for decode.
 * @param decode_count Number of selected rows.
 * @param elements_per_token Flattened scalar width copied per token.
 * @param context CUDA stream used for asynchronous execution.
 */
Status gather_decode_tokens(const Tensor& source, Tensor& destination, const int* seq_offsets,
                            const int* decode_rows, int decode_count, int elements_per_token,
                            const device::Context& context);

/**
 * @brief Scatters compact decode rows back into a flattened ragged tensor.
 * @param source Compact tensor containing `decode_count` token rows.
 * @param destination Flattened ragged destination tensor updated in place.
 * @param seq_offsets Device starting offset of each logical row.
 * @param decode_rows Device logical row indices corresponding to compact source rows.
 * @param decode_count Number of rows to scatter.
 * @param elements_per_token Flattened scalar width copied per token.
 * @param context CUDA stream used for asynchronous execution.
 */
Status scatter_decode_tokens(const Tensor& source, Tensor& destination, const int* seq_offsets,
                             const int* decode_rows, int decode_count, int elements_per_token,
                             const device::Context& context);

/**
 * @brief Gathers arbitrary flattened token rows into a compact prefill tensor.
 * @param source Flattened ragged source tensor.
 * @param destination Compact output containing `total_rows` token rows.
 * @param token_indices Device source-token index for every output row.
 * @param total_rows Number of token rows to gather.
 * @param elements_per_token Flattened scalar width copied per row.
 * @param context CUDA stream used for asynchronous execution.
 */
Status gather_prefill_tokens(const Tensor& source, Tensor& destination, const int* token_indices, int total_rows,
                             int elements_per_token, const device::Context& context);

/**
 * @brief Scatters compact prefill rows into arbitrary flattened token positions.
 * @param source Compact source containing `total_rows` token rows.
 * @param destination Flattened ragged destination updated in place.
 * @param token_indices Device destination-token index for every source row.
 * @param total_rows Number of token rows to scatter.
 * @param elements_per_token Flattened scalar width copied per row.
 * @param context CUDA stream used for asynchronous execution.
 */
Status scatter_prefill_tokens(const Tensor& source, Tensor& destination, const int* token_indices, int total_rows,
                              int elements_per_token, const device::Context& context);

/**
 * @brief Gathers the final hidden-state row of each ragged sequence for language-head projection.
 * @param source Flattened hidden states shaped `[total_tokens, hidden]` logically.
 * @param destination Preallocated output shaped `[rows, hidden]`.
 * @param hidden Hidden-state width.
 * @param last_tokens Device flattened token index for every row's final token.
 * @param rows Number of ragged sequences.
 * @param context CUDA stream used for asynchronous execution.
 */
Status gather_last_hidden(const Tensor& source, Tensor& destination, int hidden, const int* last_tokens, int rows,
                          const device::Context& context);

}  // namespace firefly::kernels
