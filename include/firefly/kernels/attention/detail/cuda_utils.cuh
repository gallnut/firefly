#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

namespace firefly::kernels::attention_detail
{
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

inline int thread_count(int head_dim)
{
    if (head_dim <= 0 || head_dim > 1024)
    {
        throw std::runtime_error("attention currently supports 1 <= head_dim <= 1024");
    }

    int threads = 32;
    while (threads < head_dim) threads <<= 1;
    return threads;
}
}  // namespace firefly::kernels::attention_detail
