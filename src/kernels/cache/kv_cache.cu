#include "firefly/kernels/cache/kv_cache.h"
#include "firefly/device/error.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

#include "firefly/kernels/detail/cuda_scalar.cuh"

namespace firefly::kernels
{
using detail::CudaScalar;
namespace
{
constexpr int page_size = 16;

template <typename scalar_t>
__global__ void append_paged_kv_kernel(const scalar_t* key, const scalar_t* value, scalar_t* key_cache,
                                       scalar_t* value_cache, const int* block_table, const int* context_lens,
                                       int max_blocks_per_sequence, int num_kv_heads, int head_dim)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;

    int position = (context_lens ? context_lens[batch] : 0) + token;
    int logical_block = position / page_size;
    int block_offset = position % page_size;
    int physical_block = block_table[batch * max_blocks_per_sequence + logical_block];

    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x)
    {
        int64_t source = ((int64_t)batch * gridDim.x + token) * num_kv_heads * head_dim + head * head_dim +
                         dimension;
        int64_t destination = (int64_t)physical_block * page_size * num_kv_heads * head_dim +
                              (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)head * head_dim + dimension;
        key_cache[destination] = key[source];
        value_cache[destination] = value[source];
    }
}

template <typename scalar_t>
void launch_append(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                   const int* block_table, const int* context_lens, int max_blocks_per_sequence, cudaStream_t stream)
{
    int batch_size = key.shape()[0];
    int sequence_length = key.shape()[1];
    int num_kv_heads = key.shape()[2];
    int head_dim = key.shape()[3];
    dim3 grid(sequence_length, num_kv_heads, batch_size);
    append_paged_kv_kernel<scalar_t><<<grid, 256, 0, stream>>>(
        static_cast<const scalar_t*>(key.data()), static_cast<const scalar_t*>(value.data()),
        static_cast<scalar_t*>(key_cache.data()), static_cast<scalar_t*>(value_cache.data()), block_table,
        context_lens, max_blocks_per_sequence, num_kv_heads, head_dim);
}

template <typename scalar_t>
__global__ void append_paged_kv_ragged_kernel(const scalar_t* key, const scalar_t* value, scalar_t* key_cache,
                                              scalar_t* value_cache, const int* block_table,
                                              const int* seq_offsets, const int* seq_lengths,
                                              const int* context_lens, int max_blocks_per_sequence,
                                              int num_kv_heads, int head_dim)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    if (token >= seq_lengths[batch]) return;

    int flat = seq_offsets[batch] + token;
    int position = context_lens[batch] + token;
    int logical_block = position / page_size;
    int block_offset = position % page_size;
    int physical_block = block_table[batch * max_blocks_per_sequence + logical_block];

    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x)
    {
        int64_t source = ((int64_t)flat * num_kv_heads + head) * head_dim + dimension;
        int64_t destination = ((int64_t)physical_block * page_size + block_offset) * num_kv_heads * head_dim +
                              (int64_t)head * head_dim + dimension;
        key_cache[destination] = key[source];
        value_cache[destination] = value[source];
    }
}

template <typename scalar_t>
void launch_append_ragged(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                          const int* block_table, const int* seq_offsets, const int* seq_lengths,
                          const int* context_lens, int batch_size, int max_sequence_length,
                          int max_blocks_per_sequence, cudaStream_t stream)
{
    int num_kv_heads = key.shape()[1];
    int head_dim = key.shape()[2];
    dim3 grid(max_sequence_length, num_kv_heads, batch_size);
    append_paged_kv_ragged_kernel<scalar_t><<<grid, 256, 0, stream>>>(
        static_cast<const scalar_t*>(key.data()), static_cast<const scalar_t*>(value.data()),
        static_cast<scalar_t*>(key_cache.data()), static_cast<scalar_t*>(value_cache.data()), block_table,
        seq_offsets, seq_lengths, context_lens, max_blocks_per_sequence, num_kv_heads, head_dim);
}

__device__ float cache_scale(float value)
{
    return fmaxf(fabsf(value) / 127.0f, 1.0e-8f);
}

__device__ __forceinline__ float warp_reduce_max(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename scalar_t>
__global__ void append_quantized_paged_kv_kernel(const scalar_t* key, const scalar_t* value, int8_t* key_cache,
                                                  int8_t* value_cache, float* scales, const int* block_table,
                                                  const int* context_lens, int max_blocks_per_sequence,
                                                  int num_kv_heads, int head_dim)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int position = (context_lens ? context_lens[batch] : 0) + token;
    int logical_block = position / page_size;
    int block_offset = position % page_size;
    int physical_block = block_table[batch * max_blocks_per_sequence + logical_block];
    int64_t source = ((int64_t)batch * gridDim.x + token) * num_kv_heads * head_dim + (int64_t)head * head_dim;
    int64_t destination = (int64_t)physical_block * page_size * num_kv_heads * head_dim +
                          (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)head * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x)
    {
        key_max = fmaxf(key_max, fabsf(CudaScalar<scalar_t>::to_float(key[source + dimension])));
        value_max = fmaxf(value_max, fabsf(CudaScalar<scalar_t>::to_float(value[source + dimension])));
    }
    key_max = warp_reduce_max(key_max);
    value_max = warp_reduce_max(value_max);

    __shared__ float shared_key_max[32];
    __shared__ float shared_value_max[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
    {
        shared_key_max[warp] = key_max;
        shared_value_max[warp] = value_max;
    }
    __syncthreads();
    if (warp == 0)
    {
        int warp_count = (blockDim.x + 31) >> 5;
        key_max = lane < warp_count ? shared_key_max[lane] : 0.0f;
        value_max = lane < warp_count ? shared_value_max[lane] : 0.0f;
        key_max = warp_reduce_max(key_max);
        value_max = warp_reduce_max(value_max);
        if (lane == 0)
        {
            shared_key_max[0] = cache_scale(key_max);
            shared_value_max[0] = cache_scale(value_max);
        }
    }
    __syncthreads();
    float key_scale = shared_key_max[0];
    float value_scale = shared_value_max[0];
    if (threadIdx.x == 0)
    {
        int64_t scale_offset =
            ((int64_t)physical_block * num_kv_heads + head) * page_size * 2 + block_offset * 2;
        scales[scale_offset] = key_scale;
        scales[scale_offset + 1] = value_scale;
    }
    float key_inverse_scale = 1.0f / key_scale;
    float value_inverse_scale = 1.0f / value_scale;
    for (int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x)
    {
        key_cache[destination + dimension] = static_cast<int8_t>(
            rintf(CudaScalar<scalar_t>::to_float(key[source + dimension]) * key_inverse_scale));
        value_cache[destination + dimension] = static_cast<int8_t>(
            rintf(CudaScalar<scalar_t>::to_float(value[source + dimension]) * value_inverse_scale));
    }
}

template <typename scalar_t>
void launch_quantized_append(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                             Tensor& scale_cache, const int* block_table, const int* context_lens,
                             int max_blocks_per_sequence, cudaStream_t stream)
{
    int batch_size = key.shape()[0];
    int sequence_length = key.shape()[1];
    int num_kv_heads = key.shape()[2];
    int head_dim = key.shape()[3];
    dim3 grid(sequence_length, num_kv_heads, batch_size);
    int threads = head_dim <= 128 ? 128 : 256;
    append_quantized_paged_kv_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        static_cast<const scalar_t*>(key.data()), static_cast<const scalar_t*>(value.data()),
        static_cast<int8_t*>(key_cache.data()), static_cast<int8_t*>(value_cache.data()),
        static_cast<float*>(scale_cache.data()), block_table, context_lens, max_blocks_per_sequence,
        num_kv_heads, head_dim);
}
}  // namespace

Status append_paged_kv(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                       const int* block_table, const int* context_lens, int max_blocks_per_sequence,
                       Tensor* scale_cache, const device::Context& context)
{
    if (block_table == nullptr)
        return unexpected(Error{ErrorCode::InvalidArgument, "append_paged_kv requires a block table"});
    if (key.shape().size() != 4 || value.shape() != key.shape())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "append_paged_kv expects matching rank-4 key/value tensors"});
    if (key.dtype() != value.dtype())
        return unexpected(Error{ErrorCode::InvalidArgument, "append_paged_kv requires matching dtypes"});

    if (scale_cache != nullptr)
    {
        if (key_cache.dtype() != DType::I8 || value_cache.dtype() != DType::I8 || scale_cache->dtype() != DType::F32)
            return unexpected(Error{ErrorCode::InvalidArgument,
                                    "quantized append requires I8 caches and F32 scales"});
        if (key.dtype() == DType::BF16)
            launch_quantized_append<__nv_bfloat16>(key, value, key_cache, value_cache, *scale_cache, block_table,
                                                   context_lens, max_blocks_per_sequence, context.stream());
        else if (key.dtype() == DType::F16)
            launch_quantized_append<half>(key, value, key_cache, value_cache, *scale_cache, block_table,
                                          context_lens, max_blocks_per_sequence, context.stream());
        else
            return unexpected(Error{ErrorCode::InvalidArgument,
                                    "quantized append only supports float16/bfloat16 inputs"});
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return unexpected(device::cuda_error(error, "append quantized paged KV"));
        return {};
    }

    if (key.dtype() != key_cache.dtype() || key.dtype() != value_cache.dtype())
        return unexpected(Error{ErrorCode::InvalidArgument, "append_paged_kv requires matching dtypes"});

    if (key.dtype() == DType::BF16)
        launch_append<__nv_bfloat16>(key, value, key_cache, value_cache, block_table, context_lens,
                                     max_blocks_per_sequence, context.stream());
    else if (key.dtype() == DType::F16)
        launch_append<half>(key, value, key_cache, value_cache, block_table, context_lens,
                            max_blocks_per_sequence, context.stream());
    else
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "append_paged_kv only supports float16/bfloat16"});
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "append paged KV"));
    return {};
}

Status append_paged_kv_ragged(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                              const int* block_table, const int* seq_offsets, const int* seq_lengths,
                              const int* context_lens, int batch_size, int max_sequence_length,
                              int max_blocks_per_sequence, const device::Context& context)
{
    if (block_table == nullptr || seq_offsets == nullptr || seq_lengths == nullptr || context_lens == nullptr)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "append_paged_kv_ragged requires block table and ragged metadata"});
    if (key.shape().size() != 3 || value.shape() != key.shape())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "append_paged_kv_ragged expects rank-3 flattened key/value tensors"});
    if (key.dtype() != value.dtype() || key.dtype() != key_cache.dtype() || key.dtype() != value_cache.dtype())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "append_paged_kv_ragged requires matching dtypes"});

    if (key.dtype() == DType::BF16)
        launch_append_ragged<__nv_bfloat16>(key, value, key_cache, value_cache, block_table, seq_offsets,
                                            seq_lengths, context_lens, batch_size, max_sequence_length,
                                            max_blocks_per_sequence, context.stream());
    else if (key.dtype() == DType::F16)
        launch_append_ragged<half>(key, value, key_cache, value_cache, block_table, seq_offsets, seq_lengths,
                                   context_lens, batch_size, max_sequence_length, max_blocks_per_sequence,
                                   context.stream());
    else
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "append_paged_kv_ragged only supports float16/bfloat16"});
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "append ragged paged KV"));
    return {};
}

}  // namespace firefly::kernels
