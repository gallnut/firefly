#pragma once
#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
namespace firefly::kernels
{
/**
 * @brief Applies rotary position embedding to dense query and key tensors in place.
 * @param q Mutable dense query tensor.
 * @param k Mutable dense key tensor.
 * @param head_dim Number of leading head channels receiving rotary embedding.
 * @param seq_len Number of new token positions per dense row.
 * @param theta Rotary embedding base period.
 * @param context_lens Device prior context length per batch row.
 * @param context CUDA stream used for asynchronous execution.
 */
Status apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens,
                  const device::Context& context = {});

/**
 * @brief Applies rotary embedding using one explicit absolute position per flattened ragged token.
 * @param q Mutable flattened ragged query tensor.
 * @param k Mutable flattened ragged key tensor.
 * @param positions Integer tensor containing one absolute position per flattened token.
 * @param theta Rotary embedding base period.
 * @param context CUDA stream used for asynchronous execution.
 */
Status apply_rope_positions(Tensor& q, Tensor& k, const Tensor& positions, float theta,
                            const device::Context& context = {});
}
