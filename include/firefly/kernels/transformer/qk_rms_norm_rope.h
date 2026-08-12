#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{
/**
 * @brief Precomputes cosine/sine rotary factors for dense query/key preprocessing.
 * @param factors Preallocated factor tensor consumed by fused Q/K preprocessing.
 * @param seq_len Number of new token positions per dense row.
 * @param head_dim Full attention head width.
 * @param theta Rotary embedding base period.
 * @param context_lens Device prior context length per batch row.
 * @param context CUDA stream used for asynchronous execution.
 */
Status prepare_rope_factors(Tensor& factors, int seq_len, int head_dim, float theta, const int* context_lens,
                            const device::Context& context = {});

/**
 * @brief Applies per-head RMS normalization and rotary embedding to query and key tensors in place.
 * @param q Mutable query tensor.
 * @param k Mutable key tensor.
 * @param q_weight Per-head query normalization weights.
 * @param k_weight Per-head key normalization weights.
 * @param rope_factors Precomputed cosine/sine factors for all processed positions.
 * @param epsilon RMS normalization stabilizer.
 * @param context CUDA stream used for asynchronous execution.
 */
Status qk_rms_norm_rope(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                        const Tensor& rope_factors, double epsilon, const device::Context& context = {});

/**
 * @brief Applies fused Q/K normalization and RoPE while emitting int8 query values and scales.
 * @param q Mutable floating-point query tensor.
 * @param k Mutable floating-point key tensor.
 * @param q_weight Per-head query normalization weights.
 * @param k_weight Per-head key normalization weights.
 * @param rope_factors Precomputed cosine/sine factors.
 * @param quantized_q Preallocated signed-int8 query output.
 * @param quantized_q_scales Preallocated per-query-row floating-point scales.
 * @param epsilon RMS normalization stabilizer.
 * @param context CUDA stream used for asynchronous execution.
 */
Status qk_rms_norm_rope_quantized(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                                  const Tensor& rope_factors, Tensor& quantized_q, Tensor& quantized_q_scales,
                                  double epsilon, const device::Context& context = {});
}
