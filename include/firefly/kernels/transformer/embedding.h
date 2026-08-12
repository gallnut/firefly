#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Gathers embedding-table rows for integer token IDs into a preallocated output tensor.
 * @param input_ids Integer tensor containing vocabulary IDs.
 * @param embedding_table Weight tensor shaped `[vocabulary, hidden]`.
 * @param output Preallocated tensor shaped as `input_ids` followed by `hidden`.
 * @param context CUDA stream used for asynchronous execution.
 */
Status embedding_lookup(const Tensor& input_ids, const Tensor& embedding_table, Tensor& output,
                        const device::Context& context = {});
}
