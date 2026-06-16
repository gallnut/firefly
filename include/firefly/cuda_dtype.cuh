#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

#include "firefly/types.h"

namespace firefly::kernels
{

template <typename T>
struct CudaScalar;

template <>
struct CudaScalar<half>
{
    static constexpr DType dtype = DType::F16;

    __device__ static float to_float(half value) { return __half2float(value); }
    __device__ static half  from_float(float value) { return __float2half(value); }
};

template <>
struct CudaScalar<__nv_bfloat16>
{
    static constexpr DType dtype = DType::BF16;

    __device__ static float         to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
    __device__ static __nv_bfloat16 from_float(float value) { return __float2bfloat16(value); }
};

inline void require_float16_or_bfloat16(DType dtype, const char* op)
{
    if (dtype != DType::F16 && dtype != DType::BF16)
    {
        throw std::runtime_error(std::string(op) + " only supports F16/BF16 tensors");
    }
}

inline void require_same_dtype(DType a, DType b, const char* op)
{
    if (a != b)
    {
        throw std::runtime_error(std::string(op) + " requires tensors with the same dtype");
    }
}

}  // namespace firefly::kernels
