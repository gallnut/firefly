#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <iostream>

#include "firefly/kernels.h"

namespace firefly::kernels
{

template <int BLOCK_THREADS>
__global__ void argmax_kernel(const half* __restrict__ logits, int* __restrict__ output_tokens, int vocab_size)
{
    typedef cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                           temp_storage;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const half* row_logits = logits + batch_idx * vocab_size;

    cub::KeyValuePair<int, float> thread_kv;
    thread_kv.key = -1;
    thread_kv.value = -1e20f;

    for (int i = tid; i < vocab_size; i += BLOCK_THREADS)
    {
        float val = __half2float(row_logits[i]);
        if (val > thread_kv.value)
        {
            thread_kv.key = i;
            thread_kv.value = val;
        }
    }

    cub::KeyValuePair<int, float> block_max = BlockReduce(temp_storage).Reduce(thread_kv, cub::ArgMax());

    // Write result
    if (tid == 0)
    {
        output_tokens[batch_idx] = block_max.key;
    }
}

void argmax(const Tensor& logits, Tensor& output_token)
{
    // logits: [batch_size, vocab_size]
    int batch_size = logits.shape()[0];
    int vocab_size = logits.shape().back();

    dim3 grid(batch_size);
    dim3 block(256);  // 256 threads is usually enough to saturate memory bandwidth for reduction

    argmax_kernel<256>
        <<<grid, block, 0, get_default_stream()>>>((const half*)logits.data(), (int*)output_token.data(), vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in argmax: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace firefly::kernels
