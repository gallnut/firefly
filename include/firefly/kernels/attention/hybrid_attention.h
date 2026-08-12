#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{

/**
 * @brief Splits a fused Qwen3.5 projection, normalizes Q/K, applies partial RoPE, and extracts the output gate.
 * @param projected_query Fused projection containing query, gate, and key channels.
 * @param query Preallocated normalized and rotated query output.
 * @param gate Preallocated learned attention output-gate tensor.
 * @param key Preallocated normalized and rotated key output.
 * @param query_norm Per-head query normalization weights.
 * @param key_norm Per-head key normalization weights.
 * @param context_lengths Device prior context length per batch row.
 * @param sequence_length Number of query positions in each dense row.
 * @param rotary_dimension Leading head channels receiving rotary embedding.
 * @param rope_theta Rotary embedding base period.
 * @param epsilon RMS normalization stabilizer.
 * @param context CUDA stream used for asynchronous execution.
 * @note All CUDA work is asynchronous on `context.stream()`.
 */
Status prepare_full_attention(const Tensor& projected_query, Tensor& query, Tensor& gate, Tensor& key,
                              const Tensor& query_norm, const Tensor& key_norm, const int* context_lengths,
                              int sequence_length, int rotary_dimension, float rope_theta, double epsilon,
                              const device::Context& context);

/**
 * @brief Applies the learned sigmoid gate to a full-attention output tensor in place.
 * @param attention_output Mutable attention activation tensor.
 * @param gate Gate logits matching attention output rows and head channels.
 * @param context CUDA stream used for asynchronous execution.
 */
Status apply_attention_gate(Tensor& attention_output, const Tensor& gate, const device::Context& context);

}  // namespace firefly::kernels
