#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void matmul(const Tensor& input, const Tensor& weight, Tensor& output, const device::Context& context = {});
}
