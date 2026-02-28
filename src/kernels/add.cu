#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "firefly/kernels.h"

// Vectorized load/store types for half
#define LOAD128BITS(value) (*reinterpret_cast<const float4*>(&(value)))
#define STORE128BITS(value) (*reinterpret_cast<float4*>(&(value)))

namespace firefly::kernels
{

template <int VEC_SIZE = 8>
__global__ void add_inplace_kernel_optimized(half* __restrict__ x, const half* __restrict__ y, int size)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (idx + VEC_SIZE > size)
    {
        return;
    }

    // Vectorized loads
    float4 x_vec = LOAD128BITS(x[idx]);
    float4 y_vec = LOAD128BITS(y[idx]);

    half* x_h = reinterpret_cast<half*>(&x_vec);
    half* y_h = reinterpret_cast<half*>(&y_vec);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
    {
        // use float addition
        float a = __half2float(x_h[i]);
        float b = __half2float(y_h[i]);
        x_h[i] = __float2half(a + b);
    }

    // Vectorized store back to x
    STORE128BITS(x[idx]) = x_vec;
}

__global__ void add_inplace_kernel(half* x, const half* y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float a = __half2float(x[idx]);
        float b = __half2float(y[idx]);
        x[idx] = __float2half(a + b);
    }
}

void add_inplace(Tensor& x, const Tensor& y)
{
    int64_t numel = x.numel();

    // Check alignment for vectorized kernel
    bool is_aligned = (reinterpret_cast<uintptr_t>(x.data()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(y.data()) % 16 == 0) && (numel % 8 == 0);

    if (is_aligned)
    {
        constexpr int vec_size = 8;
        int           threads = 256;
        int           num_blocks = (numel + (threads * vec_size) - 1) / (threads * vec_size);
        add_inplace_kernel_optimized<vec_size><<<num_blocks, threads, 0, get_default_stream()>>>(
            (half*)x.data(), (const half*)y.data(), static_cast<int>(numel));
    }
    else
    {
        int threads = 256;
        int num_blocks = (numel + threads - 1) / threads;
        add_inplace_kernel<<<num_blocks, threads, 0, get_default_stream()>>>((half*)x.data(), (const half*)y.data(),
                                                                             static_cast<int>(numel));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in add_inplace: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace firefly::kernels
