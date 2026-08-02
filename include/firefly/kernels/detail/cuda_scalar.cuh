#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace firefly::kernels::detail
{
template <typename T>
struct CudaScalar;

template <>
struct CudaScalar<half>
{
    __device__ static float to_float(half value) { return __half2float(value); }
    __device__ static half  from_float(float value) { return __float2half(value); }
};

template <>
struct CudaScalar<__nv_bfloat16>
{
    __device__ static float         to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
    __device__ static __nv_bfloat16 from_float(float value) { return __float2bfloat16(value); }
};
}  // namespace firefly::kernels::detail
