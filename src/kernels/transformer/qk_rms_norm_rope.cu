#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>

#include "firefly/device/error.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"
#include "firefly/kernels/transformer/qk_rms_norm_rope.h"

namespace firefly::kernels
{
namespace
{
using detail::CudaScalar;

template <typename value_t>
__device__ __forceinline__ value_t warp_reduce_sum(value_t value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void prepare_rope_factors_kernel(float2* __restrict__ factors, int seq_len, int total_tokens,
                                             int head_dim, float theta, const int* __restrict__ context_lens)
{
    int pair = threadIdx.x;
    int token = blockIdx.x;
    int half_dim = head_dim / 2;
    if (token >= total_tokens || pair >= half_dim) return;

    int batch = token / seq_len;
    int position = token % seq_len + (context_lens ? context_lens[batch] : 0);
    float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
    float angle = static_cast<float>(position) * powf(theta, exponent);
    factors[static_cast<int64_t>(token) * half_dim + pair] = make_float2(cosf(angle), sinf(angle));
}

template <typename scalar_t, int head_dim>
__device__ __forceinline__ void normalize_head(scalar_t* __restrict__ values,
                                               const scalar_t* __restrict__ weight, float epsilon)
{
    constexpr int vector_size = 8;
    int lane = threadIdx.x & 31;
    int start = lane * vector_size;
    float sum_sq = 0.0f;
    if (start < head_dim)
    {
        float4 input_vector = *reinterpret_cast<const float4*>(values + start);
        const scalar_t* input_values = reinterpret_cast<const scalar_t*>(&input_vector);
#pragma unroll
        for (int element = 0; element < vector_size; ++element)
        {
            float value = CudaScalar<scalar_t>::to_float(input_values[element]);
            sum_sq += value * value;
        }
    }

    float inv_rms = rsqrtf(warp_reduce_sum(sum_sq) / static_cast<float>(head_dim) + epsilon);
    if (start < head_dim)
    {
        float4 input_vector = *reinterpret_cast<const float4*>(values + start);
        float4 weight_vector = *reinterpret_cast<const float4*>(weight + start);
        scalar_t* input_values = reinterpret_cast<scalar_t*>(&input_vector);
        const scalar_t* weight_values = reinterpret_cast<const scalar_t*>(&weight_vector);
#pragma unroll
        for (int element = 0; element < vector_size; ++element)
        {
            float value = CudaScalar<scalar_t>::to_float(input_values[element]);
            float scale = CudaScalar<scalar_t>::to_float(weight_values[element]);
            input_values[element] = CudaScalar<scalar_t>::from_float(value * inv_rms * scale);
        }
        *reinterpret_cast<float4*>(values + start) = input_vector;
    }
    __syncwarp();
}

template <typename scalar_t, int head_dim>
__device__ __forceinline__ void rotate_head(scalar_t* __restrict__ values,
                                            const float2* __restrict__ factors)
{
    constexpr int half_dim = head_dim / 2;
    constexpr int vector_size = 8;
    int start = (threadIdx.x & 31) * vector_size;
    if (start >= half_dim) return;

    float4 low_vector = *reinterpret_cast<const float4*>(values + start);
    float4 high_vector = *reinterpret_cast<const float4*>(values + half_dim + start);
    scalar_t* low = reinterpret_cast<scalar_t*>(&low_vector);
    scalar_t* high = reinterpret_cast<scalar_t*>(&high_vector);
#pragma unroll
    for (int element = 0; element < vector_size; ++element)
    {
        float2 factor = factors[start + element];
        float low_value = CudaScalar<scalar_t>::to_float(low[element]);
        float high_value = CudaScalar<scalar_t>::to_float(high[element]);
        low[element] = CudaScalar<scalar_t>::from_float(low_value * factor.x - high_value * factor.y);
        high[element] = CudaScalar<scalar_t>::from_float(high_value * factor.x + low_value * factor.y);
    }
    *reinterpret_cast<float4*>(values + start) = low_vector;
    *reinterpret_cast<float4*>(values + half_dim + start) = high_vector;
}

template <typename scalar_t, int head_dim, bool quantize_query>
__global__ void qk_rms_norm_rope_kernel(scalar_t* __restrict__ q, scalar_t* __restrict__ k,
                                        const scalar_t* __restrict__ q_weight,
                                        const scalar_t* __restrict__ k_weight,
                                        const float2* __restrict__ factors, int num_q_heads, int num_k_heads,
                                        int8_t* __restrict__ quantized_q,
                                        float* __restrict__ quantized_q_scales, float epsilon)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    int warp = threadIdx.x >> 5;
    const float2* token_factors = factors + static_cast<int64_t>(token) * (head_dim / 2);

    if (warp == 0)
    {
        scalar_t* q_head = q + (static_cast<int64_t>(token) * num_q_heads + head) * head_dim;
        normalize_head<scalar_t, head_dim>(q_head, q_weight, epsilon);
        rotate_head<scalar_t, head_dim>(q_head, token_factors);
        if constexpr (quantize_query)
        {
            __syncwarp();
            constexpr int values_per_lane = head_dim / 32;
            int start = (threadIdx.x & 31) * values_per_lane;
            float maximum = 0.0f;
#pragma unroll
            for (int index = 0; index < values_per_lane; ++index)
            {
                maximum = fmaxf(maximum, fabsf(CudaScalar<scalar_t>::to_float(q_head[start + index])));
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
            }
            float query_scale = fmaxf(maximum / 127.0f, 1.0e-8f);
            if ((threadIdx.x & 31) == 0)
            {
                quantized_q_scales[static_cast<int64_t>(token) * num_q_heads + head] = query_scale;
            }
            char4 packed;
            packed.x = static_cast<int8_t>(rintf(CudaScalar<scalar_t>::to_float(q_head[start]) / query_scale));
            packed.y = static_cast<int8_t>(rintf(CudaScalar<scalar_t>::to_float(q_head[start + 1]) / query_scale));
            packed.z = static_cast<int8_t>(rintf(CudaScalar<scalar_t>::to_float(q_head[start + 2]) / query_scale));
            packed.w = static_cast<int8_t>(rintf(CudaScalar<scalar_t>::to_float(q_head[start + 3]) / query_scale));
            reinterpret_cast<char4*>(quantized_q +
                                     (static_cast<int64_t>(token) * num_q_heads + head) * head_dim)[threadIdx.x & 31] =
                packed;
        }
    }
    else if (head < num_k_heads)
    {
        scalar_t* k_head = k + (static_cast<int64_t>(token) * num_k_heads + head) * head_dim;
        normalize_head<scalar_t, head_dim>(k_head, k_weight, epsilon);
        rotate_head<scalar_t, head_dim>(k_head, token_factors);
    }
}

template <typename scalar_t, bool quantize_query>
Status launch_qk_rms_norm_rope(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                               const Tensor& factors, int head_dim, int total_tokens, double epsilon,
                               Tensor* quantized_q, Tensor* quantized_q_scales, cudaStream_t stream)
{
    dim3 grid(total_tokens, q.shape()[2]);
    if (head_dim == 128)
    {
        qk_rms_norm_rope_kernel<scalar_t, 128, quantize_query><<<grid, 64, 0, stream>>>(
            static_cast<scalar_t*>(q.data()), static_cast<scalar_t*>(k.data()),
            static_cast<const scalar_t*>(q_weight.data()), static_cast<const scalar_t*>(k_weight.data()),
            static_cast<const float2*>(factors.data()), q.shape()[2], k.shape()[2],
            quantized_q ? static_cast<int8_t*>(quantized_q->data()) : nullptr,
            quantized_q_scales ? static_cast<float*>(quantized_q_scales->data()) : nullptr,
            static_cast<float>(epsilon));
    }
    else
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "qk_rms_norm_rope currently supports head_dim 128"});
    }
    return {};
}
}  // namespace

Status prepare_rope_factors(Tensor& factors, int seq_len, int head_dim, float theta, const int* context_lens,
                            const device::Context& context)
{
    if (factors.dtype() != DType::F32 || factors.shape().size() != 3 || factors.shape()[1] != head_dim / 2 ||
        factors.shape()[2] != 2)
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "RoPE factors must have shape [tokens, head_dim / 2, 2] and dtype F32"});
    }
    if (seq_len <= 0 || head_dim <= 0 || head_dim % 2 != 0)
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "prepare_rope_factors requires positive seq_len and even head_dim"});
    }

    int total_tokens = factors.shape()[0];
    prepare_rope_factors_kernel<<<total_tokens, head_dim / 2, 0, context.stream()>>>(
        static_cast<float2*>(factors.data()), seq_len, total_tokens, head_dim, theta, context_lens);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "prepare RoPE factors"));
    return {};
}

Status qk_rms_norm_rope(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                        const Tensor& rope_factors, double epsilon, const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(q.dtype(), "qk_rms_norm_rope"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), k.dtype(), "qk_rms_norm_rope"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), q_weight.dtype(), "qk_rms_norm_rope"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), k_weight.dtype(), "qk_rms_norm_rope"));
    if (q.shape().size() != 4 || k.shape().size() != 4 || q.shape()[0] != k.shape()[0] ||
        q.shape()[1] != k.shape()[1] || q.shape()[3] != k.shape()[3])
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "qk_rms_norm_rope requires compatible rank-4 q/k tensors"});
    }
    int head_dim = q.shape()[3];
    int total_tokens = q.shape()[0] * q.shape()[1];
    if (q_weight.numel() != head_dim || k_weight.numel() != head_dim ||
        rope_factors.shape().size() != 3 || rope_factors.shape()[0] != total_tokens ||
        rope_factors.shape()[1] != head_dim / 2 || rope_factors.shape()[2] != 2)
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "qk_rms_norm_rope tensor dimensions do not match"});
    }

    if (q.dtype() == DType::BF16)
    {
        FIREFLY_TRY((launch_qk_rms_norm_rope<__nv_bfloat16, false>(q, k, q_weight, k_weight, rope_factors,
                                                                   head_dim, total_tokens, epsilon, nullptr,
                                                                   nullptr, context.stream())));
    }
    else
    {
        FIREFLY_TRY((launch_qk_rms_norm_rope<half, false>(q, k, q_weight, k_weight, rope_factors, head_dim,
                                                          total_tokens, epsilon, nullptr, nullptr,
                                                          context.stream())));
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "launch fused QK RMSNorm RoPE kernel"));
    return {};
}

Status qk_rms_norm_rope_quantized(Tensor& q, Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                                  const Tensor& rope_factors, Tensor& quantized_q, Tensor& quantized_q_scales,
                                  double epsilon, const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(q.dtype(), "qk_rms_norm_rope_quantized"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), k.dtype(), "qk_rms_norm_rope_quantized"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), q_weight.dtype(), "qk_rms_norm_rope_quantized"));
    FIREFLY_TRY(require_same_dtype(q.dtype(), k_weight.dtype(), "qk_rms_norm_rope_quantized"));
    if (q.shape().size() != 4 || k.shape().size() != 4 || q.shape()[0] != k.shape()[0] ||
        q.shape()[1] != k.shape()[1] || q.shape()[3] != 128 || k.shape()[3] != 128 ||
        quantized_q.dtype() != DType::I8 || quantized_q.numel() != q.numel() ||
        quantized_q_scales.dtype() != DType::F32 ||
        quantized_q_scales.numel() != q.shape()[0] * q.shape()[1] * q.shape()[2])
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "qk_rms_norm_rope_quantized tensor dimensions do not match"});
    }
    int total_tokens = q.shape()[0] * q.shape()[1];
    if (q_weight.numel() != 128 || k_weight.numel() != 128 || rope_factors.shape().size() != 3 ||
        rope_factors.shape()[0] != total_tokens || rope_factors.shape()[1] != 64 || rope_factors.shape()[2] != 2)
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "qk_rms_norm_rope_quantized tensor dimensions do not match"});
    }

    if (q.dtype() == DType::BF16)
    {
        FIREFLY_TRY((launch_qk_rms_norm_rope<__nv_bfloat16, true>(q, k, q_weight, k_weight, rope_factors, 128,
                                                                  total_tokens, epsilon, &quantized_q,
                                                                  &quantized_q_scales, context.stream())));
    }
    else
    {
        FIREFLY_TRY((launch_qk_rms_norm_rope<half, true>(q, k, q_weight, k_weight, rope_factors, 128,
                                                         total_tokens, epsilon, &quantized_q,
                                                         &quantized_q_scales, context.stream())));
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return unexpected(device::cuda_error(error, "launch quantized fused QK RMSNorm RoPE kernel"));
    return {};
}
}  // namespace firefly::kernels
