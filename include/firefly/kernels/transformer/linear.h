#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Multiplies the flattened leading dimensions of `input` by a row-major weight matrix.
 * @param input Activation tensor whose final dimension is the reduction width.
 * @param weight Matrix shaped `[output_features, input_features]`.
 * @param output Preallocated activation tensor with final dimension `output_features`.
 * @param context CUDA stream used for asynchronous execution.
 * @note Output storage must be preallocated with compatible shape and dtype.
 */
Status matmul(const Tensor& input, const Tensor& weight, Tensor& output, const device::Context& context = {});
}
