#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void swiglu(const Tensor& gate, const Tensor& up, Tensor& output, const device::Context& context = {});
void swiglu_fused(const Tensor& projected, Tensor& output, const device::Context& context = {});
}
