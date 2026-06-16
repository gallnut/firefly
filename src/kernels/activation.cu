#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "firefly/cuda_dtype.cuh"
#include "firefly/kernels.h"

#define LOAD128BITS(value) (*reinterpret_cast<const float4*>(&(value)))
#define STORE128BITS(value) (*reinterpret_cast<float4*>(&(value)))

namespace firefly::kernels
{

// Helper: Swish Activation
__device__ __forceinline__ float swish_func(float x)
{
    // swish(x) = x * sigmoid(x) = x / (1.0f + expf(-x))
    return x / (1.0f + expf(-x));
}

// SwiGLU Kernel (Optimized with Vectorized Load/Store)
// Out = Swish(Gate) * Up
template <typename scalar_t, int VEC_SIZE = 8>
__global__ void swiglu_kernel_optimized(const scalar_t* __restrict__ gate, const scalar_t* __restrict__ up,
                                        scalar_t* __restrict__ output, int size)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (idx + VEC_SIZE > size)
    {
        return;
    }

    // Vectorized loads
    // Each float4 load brings in 8 halfs (128 bits)
    float4 gate_vec = LOAD128BITS(gate[idx]);
    float4 up_vec = LOAD128BITS(up[idx]);

    // Reinterpret as half arrays for access
    scalar_t* gate_h = reinterpret_cast<scalar_t*>(&gate_vec);
    scalar_t* up_h = reinterpret_cast<scalar_t*>(&up_vec);

    float4    out_vec;
    scalar_t* out_h = reinterpret_cast<scalar_t*>(&out_vec);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
    {
        float g_val = CudaScalar<scalar_t>::to_float(gate_h[i]);
        float u_val = CudaScalar<scalar_t>::to_float(up_h[i]);

        // SwiGLU logic
        float res = swish_func(g_val) * u_val;

        out_h[i] = CudaScalar<scalar_t>::from_float(res);
    }

    // Vectorized store
    STORE128BITS(output[idx]) = out_vec;
}

// Scalar Fallback (for unaligned tails)
// out = (swish(gate) * up) * down is usually handled in MLP block logic
// Here: out = swish(gate) * up
template <typename scalar_t>
__global__ void swiglu_kernel(const scalar_t* gate, const scalar_t* up, scalar_t* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Simple scalar access
        // gate[idx] is half
        // up[idx] is half
        // output[idx] is half
        // Pointers are valid device pointers?

        float g = CudaScalar<scalar_t>::to_float(gate[idx]);
        float u = CudaScalar<scalar_t>::to_float(up[idx]);
        // swish(x) = x * sigmoid(x)
        float swish = swish_func(g);
        output[idx] = CudaScalar<scalar_t>::from_float(swish * u);
    }
}

template <typename scalar_t>
void dispatch_swiglu(const Tensor& gate, const Tensor& up, Tensor& output, int64_t numel)
{
    // Check alignment for vectorized kernel
    // Pointers must be 16-byte aligned and size divisible by 8
    bool is_aligned = (reinterpret_cast<uintptr_t>(gate.data()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(up.data()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(output.data()) % 16 == 0) && (numel % 8 == 0);

    // Force fallback to scalar kernel for debugging
    if (false)  // is_aligned)
    {
        constexpr int vec_size = 8;
        int           threads = 256;
        int           elements_per_thread = vec_size;
        int           num_blocks = (numel + (threads * elements_per_thread) - 1) / (threads * elements_per_thread);

        swiglu_kernel_optimized<scalar_t, vec_size><<<num_blocks, threads, 0, get_default_stream()>>>(
            static_cast<const scalar_t*>(gate.data()), static_cast<const scalar_t*>(up.data()),
            static_cast<scalar_t*>(output.data()), static_cast<int>(numel));
    }
    else
    {
        // Fallback
        int block_size = 256;
        int grid_size = (numel + block_size - 1) / block_size;
        swiglu_kernel<scalar_t><<<grid_size, block_size, 0, get_default_stream()>>>(
            static_cast<const scalar_t*>(gate.data()), static_cast<const scalar_t*>(up.data()),
            static_cast<scalar_t*>(output.data()), static_cast<int>(numel));
    }
}

void swiglu(const Tensor& gate, const Tensor& up, Tensor& output)
{
    require_float16_or_bfloat16(gate.dtype(), "swiglu");
    require_same_dtype(gate.dtype(), up.dtype(), "swiglu");
    require_same_dtype(gate.dtype(), output.dtype(), "swiglu");

    int64_t numel = output.numel();

    if (gate.dtype() == DType::BF16)
    {
        dispatch_swiglu<__nv_bfloat16>(gate, up, output, numel);
    }
    else
    {
        dispatch_swiglu<half>(gate, up, output, numel);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in swiglu: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace firefly::kernels
