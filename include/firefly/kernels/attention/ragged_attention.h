#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

// Ragged batch support operators shared by model families that process
// prefill and decode sequences in one flattened forward.
void fill_ragged_positions(Tensor& positions, const int* seq_offsets, const int* seq_lengths,
                           const int* context_lens, int batch, int total_tokens,
                           const device::Context& context);

void gather_decode_tokens(const Tensor& source, Tensor& destination, const int* seq_offsets,
                          const int* decode_rows, int decode_count, int elements_per_token,
                          const device::Context& context);

void scatter_decode_tokens(const Tensor& source, Tensor& destination, const int* seq_offsets,
                           const int* decode_rows, int decode_count, int elements_per_token,
                           const device::Context& context);

void gather_prefill_tokens(const Tensor& source, Tensor& destination, const int* token_indices, int total_rows,
                           int elements_per_token, const device::Context& context);

void scatter_prefill_tokens(const Tensor& source, Tensor& destination, const int* token_indices, int total_rows,
                            int elements_per_token, const device::Context& context);

void gather_last_hidden(const Tensor& source, Tensor& destination, int hidden, const int* last_tokens, int rows,
                        const device::Context& context);

}  // namespace firefly::kernels
