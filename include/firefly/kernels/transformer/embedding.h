#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void embedding_lookup(const Tensor& input_ids, const Tensor& embedding_table, Tensor& output,
                      const device::Context& context = {});
}
