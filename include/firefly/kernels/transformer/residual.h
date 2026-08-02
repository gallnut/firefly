#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void add_inplace(Tensor& x, const Tensor& y, const device::Context& context = {});
}
