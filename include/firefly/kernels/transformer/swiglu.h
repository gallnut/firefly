#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Computes `SiLU(gate) * up` elementwise into `output`.
 * @param gate Gate activation tensor.
 * @param up Value activation tensor matching `gate`.
 * @param output Preallocated result tensor matching the input shape and dtype.
 * @param context CUDA stream used for asynchronous execution.
 */
Status swiglu(const Tensor& gate, const Tensor& up, Tensor& output, const device::Context& context = {});
/**
 * @brief Computes SwiGLU from a final dimension containing concatenated gate and up activations.
 * @param projected Tensor whose final dimension is `[gate, up]` concatenation.
 * @param output Preallocated tensor with half the projected final dimension.
 * @param context CUDA stream used for asynchronous execution.
 */
Status swiglu_fused(const Tensor& projected, Tensor& output, const device::Context& context = {});
}
