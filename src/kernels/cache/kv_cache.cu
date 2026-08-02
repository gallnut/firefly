#include "firefly/kernels/cache/kv_cache.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>


namespace firefly::kernels
{
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
}  // namespace

void append_paged_kv(const Tensor& key, const Tensor& value, Tensor& key_cache, Tensor& value_cache,
                     const int* block_table, const int* context_lens, int max_blocks_per_sequence,
                     const device::Context& context)
{
    if (block_table == nullptr) throw std::invalid_argument("append_paged_kv requires a block table");
    if (key.shape().size() != 4 || value.shape() != key.shape())
        throw std::invalid_argument("append_paged_kv expects matching rank-4 key/value tensors");
    if (key.dtype() != value.dtype() || key.dtype() != key_cache.dtype() || key.dtype() != value_cache.dtype())
        throw std::invalid_argument("append_paged_kv requires matching dtypes");

    if (key.dtype() == DType::BF16)
        launch_append<__nv_bfloat16>(key, value, key_cache, value_cache, block_table, context_lens,
                                     max_blocks_per_sequence, context.stream());
    else if (key.dtype() == DType::F16)
        launch_append<half>(key, value, key_cache, value_cache, block_table, context_lens,
                            max_blocks_per_sequence, context.stream());
    else
        throw std::invalid_argument("append_paged_kv only supports float16/bfloat16");
}

}  // namespace firefly::kernels
