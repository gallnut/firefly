#pragma once

#include <cstdint>

#include "firefly/core/tensor.h"

namespace firefly::kernels::attention_detail
{
/** @brief Thread-local reusable storage for split decode and query quantization. */
struct DecodeWorkspace
{
    Tensor  partial_m; ///< Per-split maximum logits used by stable softmax reduction.
    Tensor  partial_l; ///< Per-split exponential sums used by stable softmax reduction.
    Tensor  partial_acc; ///< Per-split weighted value accumulators.
    Tensor  quantized_query; ///< Reusable signed-int8 query buffer.
    Tensor  quantized_query_scales; ///< Reusable floating-point query scales.
    int64_t partial_m_capacity = 0; ///< Allocated scalar capacity of `partial_m`.
    int64_t partial_l_capacity = 0; ///< Allocated scalar capacity of `partial_l`.
    int64_t partial_acc_capacity = 0; ///< Allocated scalar capacity of `partial_acc`.
    int64_t quantized_query_capacity = 0; ///< Allocated scalar capacity of `quantized_query`.
    int64_t quantized_query_scales_capacity = 0; ///< Allocated scalar capacity of query scales.
};

/**
 * @brief Returns the current thread's persistent decode workspace.
 * @return Mutable workspace unique to the calling host thread.
 * @warning References remain valid until thread exit but concurrent use on one thread is unsupported.
 */
DecodeWorkspace& decode_workspace();
/**
 * @brief Grows the current thread's workspace to satisfy a requested decode shape.
 * @param batch_size Maximum decode rows required by the next launch.
 * @param num_heads Maximum query-head count.
 * @param num_splits Maximum split-KV partials per query head.
 * @param head_dim Scalar width of each attention head.
 * @note Existing allocations are retained when their capacities are already sufficient.
 */
Status ensure_decode_workspace(int batch_size, int num_heads, int num_splits, int head_dim);
}  // namespace firefly::kernels::attention_detail
