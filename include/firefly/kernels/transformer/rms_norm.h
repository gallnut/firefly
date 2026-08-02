#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
void rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
              const device::Context& context = {});
void add_rms_norm(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                  const device::Context& context = {});
}
