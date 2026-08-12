#pragma once

#include <memory>

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels::linear_attention
{

/**
 * @brief Move-only owner of reusable temporary storage for Gated Delta Net kernels.
 *
 * The implementation is hidden to keep CUDA template details out of public headers.
 * A workspace may be reused across sequential calls but is not thread-safe.
 */
class GatedDeltaNetWorkspace
{
public:
    /** @brief Opaque implementation containing reusable device buffers. */
    struct Impl;

    /** @brief Constructs an empty workspace that grows on demand. */
    GatedDeltaNetWorkspace();
    /** @brief Releases all workspace buffers. */
    ~GatedDeltaNetWorkspace();

    /** @brief Workspace ownership cannot be copied. */
    GatedDeltaNetWorkspace(const GatedDeltaNetWorkspace&) = delete;
    /** @brief Workspace ownership cannot be copy-assigned. */
    GatedDeltaNetWorkspace& operator=(const GatedDeltaNetWorkspace&) = delete;
    /** @brief Transfers workspace ownership. */
    GatedDeltaNetWorkspace(GatedDeltaNetWorkspace&&) noexcept;
    /** @brief Releases current buffers and transfers workspace ownership. */
    GatedDeltaNetWorkspace& operator=(GatedDeltaNetWorkspace&&) noexcept;

private:
    std::unique_ptr<Impl> impl_; ///< Owned hidden buffers and capacity metadata.

    /** @brief Allows the kernel launcher to resize and access opaque workspace storage. */
    friend Status gated_delta_net(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&, Tensor&, const int*, const int*, Tensor&,
                                  GatedDeltaNetWorkspace*, double, const device::Context&);
};

/**
 * @brief Applies stateful depthwise causal convolution to projected Q/K/V channels.
 * @param projected_qkv Input shaped `[batch, sequence, channels]`.
 * @param weight Per-channel causal convolution taps.
 * @param convolution_state Mutable per-slot rolling state updated by the call.
 * @param state_slots Device sequence-slot index per batch row.
 * @param context_lengths Device prior context length per batch row.
 * @param output Preallocated convolved activation tensor matching `projected_qkv`.
 * @param context CUDA stream used for asynchronous execution.
 * @note Rows with zero prior context reset their addressed slot before processing.
 */
Status causal_convolution(const Tensor& projected_qkv, const Tensor& weight, Tensor& convolution_state,
                          const int* state_slots, const int* context_lengths, Tensor& output,
                          const device::Context& context);

/**
 * @brief Executes the recurrent Gated Delta Net update and normalized output projection input.
 * @param mixed_qkv Convolution-mixed query, key, and value activation tensor.
 * @param gate Learned output-gate activation.
 * @param decay Per-token decay activation.
 * @param beta Per-token delta-update coefficient.
 * @param decay_log Learned per-head log-decay weight.
 * @param decay_bias Learned decay bias.
 * @param norm_weight Recurrent-output normalization weight.
 * @param recurrent_state Mutable per-slot matrix state updated for every processed token.
 * @param state_slots Device sequence-slot index per batch row.
 * @param context_lengths Device prior context length per batch row.
 * @param output Preallocated normalized recurrent output.
 * @param workspace Optional reusable scratch owner; null selects temporary internal storage.
 * @param epsilon Numerical stabilizer for output normalization.
 * @param context CUDA stream used for asynchronous execution.
 */
Status gated_delta_net(const Tensor& mixed_qkv, const Tensor& gate, const Tensor& decay, const Tensor& beta,
                       const Tensor& decay_log, const Tensor& decay_bias, const Tensor& norm_weight,
                       Tensor& recurrent_state, const int* state_slots, const int* context_lengths, Tensor& output,
                       GatedDeltaNetWorkspace* workspace, double epsilon, const device::Context& context);

}  // namespace firefly::kernels::linear_attention
