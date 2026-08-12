#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Applies RMS normalization across the final tensor dimension.
 * @param input Source activation tensor.
 * @param weight Per-channel normalization weight.
 * @param output Preallocated normalized activation tensor.
 * @param epsilon Numerical stabilizer added to the mean square.
 * @param context CUDA stream used for asynchronous execution.
 */
Status rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                const device::Context& context = {});
/**
 * @brief Adds `input` into `residual` in place and normalizes the updated residual.
 * @param residual Mutable residual stream and normalization source.
 * @param input Activation added elementwise into `residual`.
 * @param weight Per-channel normalization weight.
 * @param output Preallocated normalized output.
 * @param epsilon Numerical stabilizer added to the mean square.
 * @param context CUDA stream used for asynchronous execution.
 */
Status add_rms_norm(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                    const device::Context& context = {});

/**
 * @brief Applies zero-centered RMSNorm: `output = input * rsqrt(mean(input^2)+eps) * (1+weight)`.
 * @param input Source activation tensor.
 * @param weight Per-channel zero-centered normalization weight.
 * @param output Preallocated normalized activation tensor.
 * @param epsilon Numerical stabilizer added to the mean square.
 * @param context CUDA stream used for asynchronous execution.
 */
Status rms_norm_zero_centered(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                              const device::Context& context = {});
/**
 * @brief Adds into the residual and applies zero-centered RMSNorm to the updated value.
 * @param residual Mutable residual stream and normalization source.
 * @param input Activation added elementwise into `residual`.
 * @param weight Per-channel zero-centered normalization weight.
 * @param output Preallocated normalized output.
 * @param epsilon Numerical stabilizer added to the mean square.
 * @param context CUDA stream used for asynchronous execution.
 */
Status add_rms_norm_zero_centered(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output,
                                  double epsilon, const device::Context& context = {});
}
