#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "firefly/cuda_dtype.cuh"
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

template <typename scalar_t, int HEAD_DIM, int VEC_SIZE = 8>
__global__ void rope_kernel_optimized(scalar_t* __restrict__ q, scalar_t* __restrict__ k, int num_heads_q,
                                      int num_heads_k, int seq_len, int total_tokens, float theta_base,
                                      const int* context_lens)
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
        scalar_t* q_ptr = q + q_offset;

        // Load Low Vector: q[vec_start ... vec_start+7]
        float4 vec_lo = LOAD128BITS(q_ptr[vec_start]);

        // Load High Vector: q[vec_start + HALF_DIM ... ]
        float4 vec_hi = LOAD128BITS(q_ptr[vec_start + HALF_DIM]);

        scalar_t* h_lo = reinterpret_cast<scalar_t*>(&vec_lo);
        scalar_t* h_hi = reinterpret_cast<scalar_t*>(&vec_hi);

        float4    out_lo, out_hi;
        scalar_t* out_h_lo = reinterpret_cast<scalar_t*>(&out_lo);
        scalar_t* out_h_hi = reinterpret_cast<scalar_t*>(&out_hi);

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

            float val_lo = CudaScalar<scalar_t>::to_float(h_lo[i]);
            float val_hi = CudaScalar<scalar_t>::to_float(h_hi[i]);

            // Rotate:
            // Llama/Qwen style: [x_1, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
            // out_lo = x_lo * cos - x_hi * sin
            // out_hi = x_hi * cos + x_lo * sin
            float x_new = val_lo * c - val_hi * s;
            float y_new = val_hi * c + val_lo * s;

            out_h_lo[i] = CudaScalar<scalar_t>::from_float(x_new);
            out_h_hi[i] = CudaScalar<scalar_t>::from_float(y_new);
        }

        STORE128BITS(q_ptr[vec_start]) = out_lo;
        STORE128BITS(q_ptr[vec_start + HALF_DIM]) = out_hi;
    }

    // Process K
    if (head_idx < num_heads_k)
    {
        int64_t   k_offset = ((int64_t)token_idx * num_heads_k + head_idx) * HEAD_DIM;
        scalar_t* k_ptr = k + k_offset;

        float4 vec_lo = LOAD128BITS(k_ptr[vec_start]);
        float4 vec_hi = LOAD128BITS(k_ptr[vec_start + HALF_DIM]);

        scalar_t* h_lo = reinterpret_cast<scalar_t*>(&vec_lo);
        scalar_t* h_hi = reinterpret_cast<scalar_t*>(&vec_hi);

        float4    out_lo, out_hi;
        scalar_t* out_h_lo = reinterpret_cast<scalar_t*>(&out_lo);
        scalar_t* out_h_hi = reinterpret_cast<scalar_t*>(&out_hi);

        for (int i = 0; i < VEC_SIZE; ++i)
        {
            int   element_idx = vec_start + i;
            float freq_exp = -2.0f * (float)element_idx / (float)HEAD_DIM;
            float freq = powf(theta_base, freq_exp);
            float angle = (float)seq_pos * freq;

            float c = cosf(angle);
            float s = sinf(angle);

            float val_lo = CudaScalar<scalar_t>::to_float(h_lo[i]);
            float val_hi = CudaScalar<scalar_t>::to_float(h_hi[i]);

            float x_new = val_lo * c - val_hi * s;
            float y_new = val_hi * c + val_lo * s;

            out_h_lo[i] = CudaScalar<scalar_t>::from_float(x_new);
            out_h_hi[i] = CudaScalar<scalar_t>::from_float(y_new);
        }

        STORE128BITS(k_ptr[vec_start]) = out_lo;
        STORE128BITS(k_ptr[vec_start + HALF_DIM]) = out_hi;
    }
}

template <typename scalar_t>
__global__ void rope_kernel_scalar(scalar_t* __restrict__ q, scalar_t* __restrict__ k, int num_heads_q, int num_heads_k,
                                   int seq_len, int total_tokens, int head_dim, float theta_base,
                                   const int* context_lens)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int half_dim = head_dim / 2;

    if (token_idx >= total_tokens || pair_idx >= half_dim) return;

    int batch_idx = token_idx / seq_len;
    int seq_pos = (token_idx % seq_len) + (context_lens ? context_lens[batch_idx] : 0);

    float freq_exp = -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim);
    float angle = static_cast<float>(seq_pos) * powf(theta_base, freq_exp);
    float c = cosf(angle);
    float s = sinf(angle);

    if (head_idx < num_heads_q)
    {
        scalar_t* q_ptr = q + ((int64_t)token_idx * num_heads_q + head_idx) * head_dim;
        float     lo = CudaScalar<scalar_t>::to_float(q_ptr[pair_idx]);
        float     hi = CudaScalar<scalar_t>::to_float(q_ptr[pair_idx + half_dim]);
        q_ptr[pair_idx] = CudaScalar<scalar_t>::from_float(lo * c - hi * s);
        q_ptr[pair_idx + half_dim] = CudaScalar<scalar_t>::from_float(hi * c + lo * s);
    }

    if (head_idx < num_heads_k)
    {
        scalar_t* k_ptr = k + ((int64_t)token_idx * num_heads_k + head_idx) * head_dim;
        float     lo = CudaScalar<scalar_t>::to_float(k_ptr[pair_idx]);
        float     hi = CudaScalar<scalar_t>::to_float(k_ptr[pair_idx + half_dim]);
        k_ptr[pair_idx] = CudaScalar<scalar_t>::from_float(lo * c - hi * s);
        k_ptr[pair_idx + half_dim] = CudaScalar<scalar_t>::from_float(hi * c + lo * s);
    }
}

template <typename scalar_t>
void dispatch_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens,
                   int num_heads_q, int num_heads_k, int64_t total_tokens)
{
    int  max_heads = num_heads_q > num_heads_k ? num_heads_q : num_heads_k;
    dim3 grid(static_cast<unsigned int>(total_tokens), static_cast<unsigned int>(max_heads));

    if (head_dim == 128)
    {
        rope_kernel_optimized<scalar_t, 128, 8><<<grid, 32, 0, get_default_stream()>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()), num_heads_q, num_heads_k, seq_len,
            static_cast<int>(total_tokens), theta, context_lens);
    }
    else if (head_dim == 64)
    {
        rope_kernel_optimized<scalar_t, 64, 8><<<grid, 32, 0, get_default_stream()>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()), num_heads_q, num_heads_k, seq_len,
            static_cast<int>(total_tokens), theta, context_lens);
    }
    else
    {
        int  threads = 256;
        dim3 scalar_grid((head_dim / 2 + threads - 1) / threads, static_cast<unsigned int>(total_tokens),
                         static_cast<unsigned int>(max_heads));
        rope_kernel_scalar<scalar_t><<<scalar_grid, threads, 0, get_default_stream()>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()), num_heads_q, num_heads_k, seq_len,
            static_cast<int>(total_tokens), head_dim, theta, context_lens);
    }
}

void apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens)
{
    require_float16_or_bfloat16(q.dtype(), "apply_rope");
    require_same_dtype(q.dtype(), k.dtype(), "apply_rope");

    if (q.shape().size() != 4 || k.shape().size() != 4)
    {
        throw std::runtime_error("RoPE expects rank-4 q/k tensors");
    }
    if (seq_len <= 0)
    {
        throw std::runtime_error("RoPE requires seq_len > 0");
    }
    if (head_dim != q.shape()[3] || head_dim != k.shape()[3])
    {
        throw std::runtime_error("RoPE head_dim does not match q/k tensor shapes");
    }
    if (q.shape()[0] != k.shape()[0] || q.shape()[1] != k.shape()[1])
    {
        throw std::runtime_error("RoPE requires q/k to have the same batch and sequence dimensions");
    }
    if (head_dim % 2 != 0)
    {
        throw std::runtime_error("RoPE requires an even head_dim");
    }

    int num_heads_q = q.shape()[2];
    int num_heads_k = k.shape()[2];

    int64_t total_tokens = q.numel() / (num_heads_q * head_dim);

    if (q.dtype() == DType::BF16)
    {
        dispatch_rope<__nv_bfloat16>(q, k, head_dim, seq_len, theta, context_lens, num_heads_q, num_heads_k,
                                     total_tokens);
    }
    else
    {
        dispatch_rope<half>(q, k, head_dim, seq_len, theta, context_lens, num_heads_q, num_heads_k, total_tokens);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA Error in rope: ") + cudaGetErrorString(err));
    }
}

}  // namespace firefly::kernels
