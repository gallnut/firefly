#pragma once

#include <cuda_runtime.h>

#include "firefly/core/error.h"

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

inline Result<int> thread_count(int head_dim)
{
    if (head_dim <= 0 || head_dim > 1024)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "attention currently supports 1 <= head_dim <= 1024"});

    int threads = 32;
    while (threads < head_dim) threads <<= 1;
    return threads;
}
}  // namespace firefly::kernels::attention_detail
