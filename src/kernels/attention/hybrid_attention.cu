#include "firefly/kernels/attention/hybrid_attention.h"
#include "firefly/device/error.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "firefly/core/types.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"

namespace firefly::kernels
{
namespace
{
using detail::CudaScalar;

constexpr int warp_size = 32;

__device__ __forceinline__ float warp_sum(float value)
{
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

template <int Threads>
__device__ __forceinline__ float block_sum(float value)
{
    constexpr int warp_count = (Threads + warp_size - 1) / warp_size;
    __shared__ float partials[warp_count];
    const int lane = threadIdx.x & (warp_size - 1);
    const int warp = threadIdx.x / warp_size;
    value = warp_sum(value);
    if (lane == 0) partials[warp] = value;
    __syncthreads();
    value = threadIdx.x < warp_count ? partials[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) partials[0] = value;
    __syncthreads();
    return partials[0];
}

template <typename scalar_t>
__global__ void prepare_full_attention_kernel(const scalar_t* projected_query, scalar_t* query, scalar_t* gate,
                                              scalar_t* key, const scalar_t* query_norm,
                                              const scalar_t* key_norm, const int* context_lengths,
                                              int sequence_length, int query_heads, int key_heads, int head_dim,
                                              int rotary_dimension, float rope_theta, float epsilon)
{
    constexpr int threads = 256;
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = token / sequence_length;
    const int position = token % sequence_length + (context_lengths ? context_lengths[batch] : 0);
    __shared__ float normalized_key[threads];

    if (head < query_heads)
    {
        const scalar_t* projected_head =
            projected_query + (static_cast<int64_t>(token) * query_heads + head) * head_dim * 2;
        scalar_t* query_head = query + (static_cast<int64_t>(token) * query_heads + head) * head_dim;
        scalar_t* gate_head = gate + (static_cast<int64_t>(token) * query_heads + head) * head_dim;
        float square_sum = 0.0f;
        if (threadIdx.x < head_dim)
        {
            const float value = CudaScalar<scalar_t>::to_float(projected_head[threadIdx.x]);
            square_sum = value * value;
            gate_head[threadIdx.x] = projected_head[head_dim + threadIdx.x];
        }
        const float inverse_rms = rsqrtf(block_sum<threads>(square_sum) / static_cast<float>(head_dim) + epsilon);
        if (threadIdx.x < head_dim)
        {
            float value = CudaScalar<scalar_t>::to_float(projected_head[threadIdx.x]);
            value *= inverse_rms * (1.0f + CudaScalar<scalar_t>::to_float(query_norm[threadIdx.x]));
            if (threadIdx.x < rotary_dimension)
            {
                const int half = rotary_dimension / 2;
                const int pair = threadIdx.x % half;
                const int partner = threadIdx.x < half ? threadIdx.x + half : threadIdx.x - half;
                const float other = CudaScalar<scalar_t>::to_float(projected_head[partner]) * inverse_rms *
                                    (1.0f + CudaScalar<scalar_t>::to_float(query_norm[partner]));
                const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dimension);
                const float angle = static_cast<float>(position) * powf(rope_theta, exponent);
                const float cosine = cosf(angle);
                const float sine = sinf(angle);
                value = threadIdx.x < half ? value * cosine - other * sine : value * cosine + other * sine;
            }
            query_head[threadIdx.x] = CudaScalar<scalar_t>::from_float(value);
        }
    }

    if (head < key_heads)
    {
        scalar_t* key_head = key + (static_cast<int64_t>(token) * key_heads + head) * head_dim;
        float square_sum = 0.0f;
        if (threadIdx.x < head_dim)
        {
            const float value = CudaScalar<scalar_t>::to_float(key_head[threadIdx.x]);
            square_sum = value * value;
        }
        const float inverse_rms = rsqrtf(block_sum<threads>(square_sum) / static_cast<float>(head_dim) + epsilon);
        if (threadIdx.x < head_dim)
        {
            normalized_key[threadIdx.x] = CudaScalar<scalar_t>::to_float(key_head[threadIdx.x]) * inverse_rms *
                                          (1.0f + CudaScalar<scalar_t>::to_float(key_norm[threadIdx.x]));
        }
        __syncthreads();
        if (threadIdx.x < head_dim)
        {
            float value = normalized_key[threadIdx.x];
            if (threadIdx.x < rotary_dimension)
            {
                const int half = rotary_dimension / 2;
                const int pair = threadIdx.x % half;
                const int partner = threadIdx.x < half ? threadIdx.x + half : threadIdx.x - half;
                const float other = normalized_key[partner];
                const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dimension);
                const float angle = static_cast<float>(position) * powf(rope_theta, exponent);
                const float cosine = cosf(angle);
                const float sine = sinf(angle);
                value = threadIdx.x < half ? value * cosine - other * sine : value * cosine + other * sine;
            }
            key_head[threadIdx.x] = CudaScalar<scalar_t>::from_float(value);
        }
    }
}

template <typename scalar_t>
__global__ void attention_gate_kernel(scalar_t* output, const scalar_t* gate, int64_t element_count)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < element_count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const float gate_value = CudaScalar<scalar_t>::to_float(gate[index]);
        const float value = CudaScalar<scalar_t>::to_float(output[index]);
        output[index] = CudaScalar<scalar_t>::from_float(value / (1.0f + expf(-gate_value)));
    }
}

template <typename Function>
Status dispatch_half_type(DType dtype, Function&& function)
{
    FIREFLY_TRY(require_float16_or_bfloat16(dtype, "hybrid attention kernel"));
    if (dtype == DType::BF16) function.template operator()<__nv_bfloat16>();
    else function.template operator()<half>();
    return {};
}

Status check_launch(const char* operation)
{
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, operation));
    return {};
}
}  // namespace

Status prepare_full_attention(const Tensor& projected_query, Tensor& query, Tensor& gate, Tensor& key,
                              const Tensor& query_norm, const Tensor& key_norm, const int* context_lengths,
                              int sequence_length, int rotary_dimension, float rope_theta, double epsilon,
                              const device::Context& context)
{
    const int tokens = query.shape()[0] * query.shape()[1];
    const int query_heads = query.shape()[2];
    const int key_heads = key.shape()[2];
    const int head_dim = query.shape()[3];
    if (key.shape()[3] != head_dim || projected_query.numel() != query.numel() * 2 || gate.numel() != query.numel())
        return unexpected(Error{ErrorCode::InvalidArgument, "invalid hybrid full-attention tensor shape"});
    FIREFLY_TRY(dispatch_half_type(query.dtype(), [&]<typename scalar_t>()
    {
        dim3 grid(tokens, std::max(query_heads, key_heads));
        prepare_full_attention_kernel<scalar_t><<<grid, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(projected_query.data()), static_cast<scalar_t*>(query.data()),
            static_cast<scalar_t*>(gate.data()), static_cast<scalar_t*>(key.data()),
            static_cast<const scalar_t*>(query_norm.data()), static_cast<const scalar_t*>(key_norm.data()),
            context_lengths, sequence_length, query_heads, key_heads, head_dim, rotary_dimension, rope_theta,
            static_cast<float>(epsilon));
    }));
    return check_launch("hybrid full-attention preparation");
}

Status apply_attention_gate(Tensor& attention_output, const Tensor& gate, const device::Context& context)
{
    if (attention_output.numel() != gate.numel())
        return unexpected(Error{ErrorCode::InvalidArgument, "attention gate tensor size mismatch"});
    FIREFLY_TRY(dispatch_half_type(attention_output.dtype(), [&]<typename scalar_t>()
    {
        const int blocks = std::min<int64_t>((attention_output.numel() + 255) / 256, 4096);
        attention_gate_kernel<scalar_t><<<blocks, 256, 0, context.stream()>>>(
            static_cast<scalar_t*>(attention_output.data()), static_cast<const scalar_t*>(gate.data()),
            attention_output.numel());
    }));
    return check_launch("hybrid attention gate");
}

}  // namespace firefly::kernels
