#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens,
                const device::Context& context = {});

// Ragged variant: positions is a (total_tokens,) tensor giving the absolute
// position of every token, so sequences with different lengths can share a call.
void apply_rope_positions(Tensor& q, Tensor& k, const Tensor& positions, float theta,
                          const device::Context& context = {});
}
