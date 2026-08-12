#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "firefly/device/error.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"
#include "firefly/kernels/transformer/rope.h"

#define LOAD128BITS(value) (*reinterpret_cast<const float4*>(&(value)))
#define STORE128BITS(value) (*reinterpret_cast<float4*>(&(value)))

namespace firefly::kernels
{
using detail::CudaScalar;

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
                                      const int* context_lens, const int* positions)
{
    constexpr int half_dim = HEAD_DIM / 2;
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int vec_start = threadIdx.x * VEC_SIZE;
    if (token_idx >= total_tokens || vec_start >= half_dim) return;

    int batch_idx = token_idx / seq_len;
    int seq_pos = positions ? positions[token_idx] : token_idx % seq_len + (context_lens ? context_lens[batch_idx] : 0);

    auto rotate_head = [&](scalar_t* head)
    {
        float4 low = LOAD128BITS(head[vec_start]);
        float4 high = LOAD128BITS(head[vec_start + half_dim]);
        scalar_t* low_values = reinterpret_cast<scalar_t*>(&low);
        scalar_t* high_values = reinterpret_cast<scalar_t*>(&high);
#pragma unroll
        for (int element = 0; element < VEC_SIZE; ++element)
        {
            int dimension = vec_start + element;
            float exponent = -2.0f * static_cast<float>(dimension) / static_cast<float>(HEAD_DIM);
            float angle = static_cast<float>(seq_pos) * powf(theta_base, exponent);
            float c = cosf(angle);
            float s = sinf(angle);
            float lo = CudaScalar<scalar_t>::to_float(low_values[element]);
            float hi = CudaScalar<scalar_t>::to_float(high_values[element]);
            low_values[element] = CudaScalar<scalar_t>::from_float(lo * c - hi * s);
            high_values[element] = CudaScalar<scalar_t>::from_float(hi * c + lo * s);
        }
        STORE128BITS(head[vec_start]) = low;
        STORE128BITS(head[vec_start + half_dim]) = high;
    };

    if (head_idx < num_heads_q)
    {
        rotate_head(q + (static_cast<int64_t>(token_idx) * num_heads_q + head_idx) * HEAD_DIM);
    }
    if (head_idx < num_heads_k)
    {
        rotate_head(k + (static_cast<int64_t>(token_idx) * num_heads_k + head_idx) * HEAD_DIM);
    }
}

template <typename scalar_t>
__global__ void rope_kernel_scalar(scalar_t* __restrict__ q, scalar_t* __restrict__ k, int num_heads_q, int num_heads_k,
                                   int seq_len, int total_tokens, int head_dim, float theta_base,
                                   const int* context_lens, const int* positions)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int half_dim = head_dim / 2;

    if (token_idx >= total_tokens || pair_idx >= half_dim) return;

    int batch_idx = token_idx / seq_len;
    int seq_pos = positions ? positions[token_idx] : (token_idx % seq_len) + (context_lens ? context_lens[batch_idx] : 0);

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
                   const int* positions, int num_heads_q, int num_heads_k, int64_t total_tokens,
                   cudaStream_t stream)
{
    int  max_heads = num_heads_q > num_heads_k ? num_heads_q : num_heads_k;

    if (head_dim == 128)
    {
        dim3 grid(static_cast<unsigned int>(total_tokens), static_cast<unsigned int>(max_heads));
        rope_kernel_optimized<scalar_t, 128><<<grid, 32, 0, stream>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()), num_heads_q, num_heads_k, seq_len,
            static_cast<int>(total_tokens), theta, context_lens, positions);
    }
    else if (head_dim == 64)
    {
        dim3 grid(static_cast<unsigned int>(total_tokens), static_cast<unsigned int>(max_heads));
        rope_kernel_optimized<scalar_t, 64><<<grid, 32, 0, stream>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()), num_heads_q, num_heads_k, seq_len,
            static_cast<int>(total_tokens), theta, context_lens, positions);
    }
    else
    {
        int  threads = 256;
        dim3 scalar_grid((head_dim / 2 + threads - 1) / threads, static_cast<unsigned int>(total_tokens),
                         static_cast<unsigned int>(max_heads));
        rope_kernel_scalar<scalar_t><<<scalar_grid, threads, 0, stream>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()), num_heads_q, num_heads_k, seq_len,
            static_cast<int>(total_tokens), head_dim, theta, context_lens, positions);
    }
}

Status apply_rope(Tensor& q, Tensor& k, int head_dim, int seq_len, float theta, const int* context_lens,
                  const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(q.dtype(), "apply_rope"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), k.dtype(), "apply_rope"));

    if (q.shape().size() != 4 || k.shape().size() != 4)
    {
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE expects rank-4 q/k tensors"});
    }
    if (seq_len <= 0)
    {
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE requires seq_len > 0"});
    }
    if (head_dim != q.shape()[3] || head_dim != k.shape()[3])
    {
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE head_dim does not match q/k tensor shapes"});
    }
    if (q.shape()[0] != k.shape()[0] || q.shape()[1] != k.shape()[1])
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "RoPE requires q/k to have the same batch and sequence dimensions"});
    }
    if (head_dim % 2 != 0)
    {
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE requires an even head_dim"});
    }

    int num_heads_q = q.shape()[2];
    int num_heads_k = k.shape()[2];

    int64_t total_tokens = q.numel() / (num_heads_q * head_dim);

    if (q.dtype() == DType::BF16)
    {
        dispatch_rope<__nv_bfloat16>(q, k, head_dim, seq_len, theta, context_lens, nullptr, num_heads_q,
                                     num_heads_k, total_tokens, context.stream());
    }
    else
    {
        dispatch_rope<half>(q, k, head_dim, seq_len, theta, context_lens, nullptr, num_heads_q, num_heads_k,
                            total_tokens, context.stream());
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return unexpected(device::cuda_error(err, "launch RoPE kernel"));
    return {};
}

Status apply_rope_positions(Tensor& q, Tensor& k, const Tensor& positions, float theta,
                            const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(q.dtype(), "apply_rope_positions"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), k.dtype(), "apply_rope_positions"));
    if (q.shape().size() != 4 || k.shape().size() != 4)
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE positions expects rank-4 q/k tensors"});
    if (positions.dtype() != DType::I32)
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE positions expects I32 positions"});
    if (q.shape()[0] != k.shape()[0] || q.shape()[1] != k.shape()[1])
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "RoPE positions requires q/k to have the same batch and sequence dimensions"});
    if (q.shape()[3] != k.shape()[3] || q.shape()[3] % 2 != 0)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "RoPE positions requires matching even head_dim"});

    const int head_dim = q.shape()[3];
    const int num_heads_q = q.shape()[2];
    const int num_heads_k = k.shape()[2];
    const int total_tokens = static_cast<int>(q.numel() / (num_heads_q * head_dim));
    if (total_tokens != positions.numel())
        return unexpected(Error{ErrorCode::InvalidArgument, "RoPE positions token count mismatch"});

    if (q.dtype() == DType::BF16)
    {
        dispatch_rope<__nv_bfloat16>(q, k, head_dim, total_tokens, theta, nullptr,
                                     static_cast<const int*>(positions.data()), num_heads_q, num_heads_k,
                                     total_tokens, context.stream());
    }
    else
    {
        dispatch_rope<half>(q, k, head_dim, total_tokens, theta, nullptr,
                            static_cast<const int*>(positions.data()), num_heads_q, num_heads_k, total_tokens,
                            context.stream());
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return unexpected(device::cuda_error(err, "launch positioned RoPE kernel"));
    return {};
}

}  // namespace firefly::kernels
