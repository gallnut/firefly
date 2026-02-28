#pragma once

#include "firefly/tensor.h"

namespace firefly::kernels
{

void         set_default_stream(cudaStream_t stream);
cudaStream_t get_default_stream();

/**
 * @brief Performs embedding lookup.
 *
 * @param input_ids Input token indices.
 * @param embedding_table The embedding weight matrix.
 * @param output Output tensor.
 */
void embedding_lookup(const Tensor& input_ids, const Tensor& embedding_table, Tensor& output);

/**
 * @brief Applies Root Mean Square (RMS) normalization.
 *
 * @param input Input tensor.
 * @param weight Weight tensor for scaling.
 * @param output Output tensor.
 * @param epsilon Small constant for numerical stability.
 */
void rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon);

/**
 * @brief Applies the SwiGLU activation function.
 *
 * Output = Linear( F.silu(Gate) * Up )
 *
 * @param gate The gate tensor.
 * @param up The up tensor.
 * @param output Output tensor.
 */
void swiglu(const Tensor& gate, const Tensor& up, Tensor& output);

/**
 * @brief Performs in-place matrix addition (x += y).
 *
 * @param x The tensor to be added to and modified.
 * @param y The tensor to add.
 */
void add_inplace(Tensor& x, const Tensor& y);

/**
 * @brief Performs matrix multiplication (C = A * B).
 *
 * @param input Input tensor A.
 * @param weight Weight tensor B.
 * @param output Output tensor C.
 */
void matmul(const Tensor& input, const Tensor& weight, Tensor& output);

/**
 * @brief Applies Rotary Positional Embeddings (RoPE).
 *
 * @param q Query tensor.
 * @param k Key tensor.
 * @param head_dim Dimension of each attention head.
 * @param seq_len Sequence length.
 * @param theta Base for the exponential frequency calculation.
 * @param context_lens Array of context lengths per batch element.
 */
void apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens);

/**
 * @brief Computes Scaled Dot-Product Attention (FlashAttention / PagedAttention).
 *
 * @param q Query tensor.
 * @param k Key tensor or Key cache.
 * @param v Value tensor or Value cache.
 * @param output Output tensor.
 * @param kv_cache_block_table Block table for PagedAttention (nullptr for FlashAttention).
 * @param kv_head_num Number of KV heads.
 * @param seq_len Sequence length.
 * @param max_context_blocks Maximum number of context blocks per sequence.
 * @param context_lens Array of context lengths per batch element.
 */
void attention(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* kv_cache_block_table, int kv_head_num,
               int seq_len, int max_context_blocks, const int* context_lens);

/**
 * @brief Performs argmax for greedy sampling.
 *
 * @param logits Input logits tensor.
 * @param output_token Output tensor containing the sampled token index.
 */
void argmax(const Tensor& logits, Tensor& output_token);

}  // namespace firefly::kernels
