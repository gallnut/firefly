#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens,
                const device::Context& context = {});
}
