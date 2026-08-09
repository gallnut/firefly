#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

// Hybrid (linear + full) attention support operators.
void prepare_full_attention(const Tensor& projected_query, Tensor& query, Tensor& gate, Tensor& key,
                            const Tensor& query_norm, const Tensor& key_norm, const int* context_lengths,
                            int sequence_length, int rotary_dimension, float rope_theta, double epsilon,
                            const device::Context& context);

void apply_attention_gate(Tensor& attention_output, const Tensor& gate, const device::Context& context);

}  // namespace firefly::kernels
