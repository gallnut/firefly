#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

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
template <int VEC_SIZE = 8>
__global__ void swiglu_kernel_optimized(const half* __restrict__ gate, const half* __restrict__ up,
                                        half* __restrict__ output, int size)
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
    half* gate_h = reinterpret_cast<half*>(&gate_vec);
    half* up_h = reinterpret_cast<half*>(&up_vec);

    float4 out_vec;
    half*  out_h = reinterpret_cast<half*>(&out_vec);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
    {
        float g_val = __half2float(gate_h[i]);
        float u_val = __half2float(up_h[i]);

        // SwiGLU logic
        float res = swish_func(g_val) * u_val;

        out_h[i] = __float2half(res);
    }

    // Vectorized store
    STORE128BITS(output[idx]) = out_vec;
}

// Scalar Fallback (for unaligned tails)
// out = (swish(gate) * up) * down is usually handled in MLP block logic
// Here: out = swish(gate) * up
__global__ void swiglu_kernel(const half* gate, const half* up, half* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Simple scalar access
        // gate[idx] is half
        // up[idx] is half
        // output[idx] is half
        // Pointers are valid device pointers?

        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        // swish(x) = x * sigmoid(x)
        float swish = swish_func(g);
        output[idx] = __float2half(swish * u);
    }
}

void swiglu(const Tensor& gate, const Tensor& up, Tensor& output)
{
    int64_t numel = output.numel();

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

        swiglu_kernel_optimized<vec_size><<<num_blocks, threads, 0, get_default_stream()>>>(
            (const half*)gate.data(), (const half*)up.data(), (half*)output.data(), static_cast<int>(numel));
    }
    else
    {
        // Fallback
        int block_size = 256;
        int grid_size = (numel + block_size - 1) / block_size;
        swiglu_kernel<<<grid_size, block_size, 0, get_default_stream()>>>(
            (const half*)gate.data(), (const half*)up.data(), (half*)output.data(), static_cast<int>(numel));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in swiglu: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace firefly::kernels
