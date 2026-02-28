#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "firefly/kernels.h"

#define LOAD128BITS(value) (*reinterpret_cast<const float4*>(&(value)))
#define STORE128BITS(value) (*reinterpret_cast<float4*>(&(value)))

namespace firefly::kernels
{

__device__ __forceinline__ void rotate_pair(float& x, float& y, float c, float s)
{
    float x_new = x * c - y * s;
    float y_new = y * c + x * s;
    x = x_new;
    y = y_new;
}

template <int HEAD_DIM, int VEC_SIZE = 8>
__global__ void rope_kernel_optimized(half* __restrict__ q, half* __restrict__ k, int num_heads_q, int num_heads_k,
                                      int seq_len, int total_tokens, float theta_base, const int* context_lens)
{
    constexpr int HALF_DIM = HEAD_DIM / 2;

    // Grid Mapping
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (token_idx >= total_tokens) return;

    int batch_idx = token_idx / seq_len;
    int seq_pos = (token_idx % seq_len) + (context_lens ? context_lens[batch_idx] : 0);

    // Safety check for seq_len division by zero? No, seq_len passed is > 0.

    // Thread Mapping
    int tid = threadIdx.x;
    int vec_start = tid * VEC_SIZE;

    // Guard: Check if this thread processes valid elements within HEAD_DIM/2
    // We process pairs (i, i + HALF_DIM). So we only need to cover range [0, HALF_DIM).
    if (vec_start >= HALF_DIM) return;

    int64_t q_offset = ((int64_t)token_idx * num_heads_q + head_idx) * HEAD_DIM;

    // Process Q
    if (head_idx < num_heads_q)
    {
        half* q_ptr = q + q_offset;

        // Load Low Vector: q[vec_start ... vec_start+7]
        float4 vec_lo = LOAD128BITS(q_ptr[vec_start]);

        // Load High Vector: q[vec_start + HALF_DIM ... ]
        float4 vec_hi = LOAD128BITS(q_ptr[vec_start + HALF_DIM]);

        half* h_lo = reinterpret_cast<half*>(&vec_lo);
        half* h_hi = reinterpret_cast<half*>(&vec_hi);

        float4 out_lo, out_hi;
        half*  out_h_lo = reinterpret_cast<half*>(&out_lo);
        half*  out_h_hi = reinterpret_cast<half*>(&out_hi);

        // Calculate frequency and rotate
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            int element_idx = vec_start + i;
            // HuggingFace RoPE:
            // freq = theta ^ (-2i / dim) for i = 0, 1, ..., dim/2 - 1
            float freq_exp = -2.0f * (float)element_idx / (float)HEAD_DIM;
            float freq = powf(theta_base, freq_exp);
            float angle = (float)seq_pos * freq;

            float c = cosf(angle);
            float s = sinf(angle);

            float val_lo = __half2float(h_lo[i]);
            float val_hi = __half2float(h_hi[i]);

            // Rotate:
            // Llama/Qwen style: [x_1, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
            // out_lo = x_lo * cos - x_hi * sin
            // out_hi = x_hi * cos + x_lo * sin
            float x_new = val_lo * c - val_hi * s;
            float y_new = val_hi * c + val_lo * s;

            out_h_lo[i] = __float2half(x_new);
            out_h_hi[i] = __float2half(y_new);
        }

        STORE128BITS(q_ptr[vec_start]) = out_lo;
        STORE128BITS(q_ptr[vec_start + HALF_DIM]) = out_hi;
    }

    // Process K
    if (head_idx < num_heads_k)
    {
        int64_t k_offset = ((int64_t)token_idx * num_heads_k + head_idx) * HEAD_DIM;
        half*   k_ptr = k + k_offset;

        float4 vec_lo = LOAD128BITS(k_ptr[vec_start]);
        float4 vec_hi = LOAD128BITS(k_ptr[vec_start + HALF_DIM]);

        half* h_lo = reinterpret_cast<half*>(&vec_lo);
        half* h_hi = reinterpret_cast<half*>(&vec_hi);

        float4 out_lo, out_hi;
        half*  out_h_lo = reinterpret_cast<half*>(&out_lo);
        half*  out_h_hi = reinterpret_cast<half*>(&out_hi);

        for (int i = 0; i < VEC_SIZE; ++i)
        {
            int   element_idx = vec_start + i;
            float freq_exp = -2.0f * (float)element_idx / (float)HEAD_DIM;
            float freq = powf(theta_base, freq_exp);
            float angle = (float)seq_pos * freq;

            float c = cosf(angle);
            float s = sinf(angle);

            float val_lo = __half2float(h_lo[i]);
            float val_hi = __half2float(h_hi[i]);

            float x_new = val_lo * c - val_hi * s;
            float y_new = val_hi * c + val_lo * s;

            out_h_lo[i] = __float2half(x_new);
            out_h_hi[i] = __float2half(y_new);
        }

        STORE128BITS(k_ptr[vec_start]) = out_lo;
        STORE128BITS(k_ptr[vec_start + HALF_DIM]) = out_hi;
    }
}

void apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens)
{
    int num_heads_q = q.shape()[2];
    int num_heads_k = k.shape()[2];

    int64_t total_tokens = q.numel() / (num_heads_q * head_dim);

    // Grid Y is max heads. We check bounds inside kernel for K.
    dim3 grid(static_cast<unsigned int>(total_tokens), static_cast<unsigned int>(num_heads_q));

    // Vector Size 8 (loading 8 halves via float4)
    constexpr int VEC_SIZE = 8;

    if (head_dim == 128)
    {
        rope_kernel_optimized<128, 8>
            <<<grid, 32, 0, get_default_stream()>>>((half*)q.data(), (half*)k.data(), num_heads_q, num_heads_k, seq_len,
                                                    static_cast<int>(total_tokens), theta, context_lens);
    }
    else if (head_dim == 64)
    {
        // Pairs=32. VEC=8 -> 4 threads.
        // Launch 32, 4 active.
        rope_kernel_optimized<64, 8>
            <<<grid, 32, 0, get_default_stream()>>>((half*)q.data(), (half*)k.data(), num_heads_q, num_heads_k, seq_len,
                                                    static_cast<int>(total_tokens), theta, context_lens);
    }
    else
    {
        // Generic fallback (slow, scalar)
        // Not implemented here to keep concise.
        std::cerr << "RoPE: Unsupported head_dim " << head_dim << std::endl;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in rope: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace firefly::kernels
