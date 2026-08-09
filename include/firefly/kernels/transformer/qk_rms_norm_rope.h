#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{
void prepare_rope_factors(Tensor& factors, int seq_len, int head_dim, float theta, const int* context_lens,
                          const device::Context& context = {});

void qk_rms_norm_rope(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                      const Tensor& rope_factors, double epsilon, const device::Context& context = {});

void qk_rms_norm_rope_quantized(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                                const Tensor& rope_factors, Tensor& quantized_q, Tensor& quantized_q_scales,
                                double epsilon, const device::Context& context = {});
}
