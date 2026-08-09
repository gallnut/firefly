#pragma once

#include <memory>

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels::linear_attention
{

class GatedDeltaNetWorkspace
{
public:
    struct Impl;

    GatedDeltaNetWorkspace();
    ~GatedDeltaNetWorkspace();

    GatedDeltaNetWorkspace(const GatedDeltaNetWorkspace&) = delete;
    GatedDeltaNetWorkspace& operator=(const GatedDeltaNetWorkspace&) = delete;
    GatedDeltaNetWorkspace(GatedDeltaNetWorkspace&&) noexcept;
    GatedDeltaNetWorkspace& operator=(GatedDeltaNetWorkspace&&) noexcept;

private:
    std::unique_ptr<Impl> impl_;

    friend void gated_delta_net(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, Tensor&, const int*, const int*, Tensor&,
                                GatedDeltaNetWorkspace*, double, const device::Context&);
};

void causal_convolution(const Tensor& projected_qkv, const Tensor& weight, Tensor& convolution_state,
                        const int* state_slots, const int* context_lengths, Tensor& output,
                        const device::Context& context);

void gated_delta_net(const Tensor& mixed_qkv, const Tensor& gate, const Tensor& decay, const Tensor& beta,
                     const Tensor& decay_log, const Tensor& decay_bias, const Tensor& norm_weight,
                     Tensor& recurrent_state, const int* state_slots, const int* context_lengths, Tensor& output,
                     GatedDeltaNetWorkspace* workspace, double epsilon, const device::Context& context);

}  // namespace firefly::kernels::linear_attention
