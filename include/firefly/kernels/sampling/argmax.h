#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void argmax(const Tensor& logits, Tensor& output_token, const device::Context& context = {});
}
