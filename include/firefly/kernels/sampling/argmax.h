#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Selects the maximum-logit token for every leading row.
 * @param logits Tensor whose final dimension is vocabulary.
 * @param output_token Preallocated `DType::I32` tensor containing one ID per row.
 * @param context CUDA execution context; the operation is asynchronous.
 */
Status argmax(const Tensor& logits, Tensor& output_token, const device::Context& context = {});
}
