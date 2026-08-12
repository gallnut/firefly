#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Adds `y` elementwise into `x` in place.
 * @param x Mutable destination and left operand.
 * @param y Immutable right operand with the same shape and dtype as `x`.
 * @param context CUDA stream used for asynchronous execution.
 */
Status add_inplace(Tensor& x, const Tensor& y, const device::Context& context = {});
}
