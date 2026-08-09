#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include "firefly/core/logging.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"
#include "firefly/kernels/transformer/swiglu.h"

#define LOAD128BITS(value) (*reinterpret_cast<const float4*>(&(value)))
#define STORE128BITS(value) (*reinterpret_cast<float4*>(&(value)))

namespace firefly::kernels
{
using detail::CudaScalar;

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
void dispatch_swiglu(const Tensor& gate, const Tensor& up, Tensor& output, int64_t numel, cudaStream_t stream)
{
    // Check alignment for vectorized kernel
    // Pointers must be 16-byte aligned and size divisible by 8
    bool is_aligned = (reinterpret_cast<uintptr_t>(gate.data()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(up.data()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(output.data()) % 16 == 0) && (numel % 8 == 0);

    if (is_aligned)
    {
        constexpr int vec_size = 8;
        int           threads = 256;
        int           elements_per_thread = vec_size;
        int           num_blocks = (numel + (threads * elements_per_thread) - 1) / (threads * elements_per_thread);

        swiglu_kernel_optimized<scalar_t, vec_size><<<num_blocks, threads, 0, stream>>>(
            static_cast<const scalar_t*>(gate.data()), static_cast<const scalar_t*>(up.data()),
            static_cast<scalar_t*>(output.data()), static_cast<int>(numel));
    }
    else
    {
        // Fallback
        int block_size = 256;
        int grid_size = (numel + block_size - 1) / block_size;
        swiglu_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
            static_cast<const scalar_t*>(gate.data()), static_cast<const scalar_t*>(up.data()),
            static_cast<scalar_t*>(output.data()), static_cast<int>(numel));
    }
}

template <typename scalar_t, int VEC_SIZE = 8>
__global__ void swiglu_fused_kernel(const scalar_t* __restrict__ projected, scalar_t* __restrict__ output,
                                    int row_width, int64_t element_count)
{
    const int64_t output_index =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC_SIZE;
    if (output_index + VEC_SIZE > element_count) return;

    const int64_t row = output_index / row_width;
    const int column = output_index - row * row_width;
    const scalar_t* gate = projected + row * row_width * 2 + column;
    const scalar_t* up = gate + row_width;

    float4 gate_vector = LOAD128BITS(gate[0]);
    float4 up_vector = LOAD128BITS(up[0]);
    float4 output_vector;
    scalar_t* gate_values = reinterpret_cast<scalar_t*>(&gate_vector);
    scalar_t* up_values = reinterpret_cast<scalar_t*>(&up_vector);
    scalar_t* output_values = reinterpret_cast<scalar_t*>(&output_vector);
#pragma unroll
    for (int element = 0; element < VEC_SIZE; ++element)
    {
        const float gate_value = CudaScalar<scalar_t>::to_float(gate_values[element]);
        const float up_value = CudaScalar<scalar_t>::to_float(up_values[element]);
        output_values[element] = CudaScalar<scalar_t>::from_float(swish_func(gate_value) * up_value);
    }
    STORE128BITS(output[output_index]) = output_vector;
}

template <typename scalar_t>
void dispatch_swiglu_fused(const Tensor& projected, Tensor& output, int row_width, cudaStream_t stream)
{
    constexpr int vector_size = 8;
    constexpr int threads = 256;
    const int64_t element_count = output.numel();
    const int blocks = static_cast<int>((element_count + threads * vector_size - 1) /
                                        (threads * vector_size));
    swiglu_fused_kernel<scalar_t, vector_size><<<blocks, threads, 0, stream>>>(
        static_cast<const scalar_t*>(projected.data()), static_cast<scalar_t*>(output.data()), row_width,
        element_count);
}

void swiglu(const Tensor& gate, const Tensor& up, Tensor& output, const device::Context& context)
{
    require_float16_or_bfloat16(gate.dtype(), "swiglu");
    require_same_dtype(gate.dtype(), up.dtype(), "swiglu");
    require_same_dtype(gate.dtype(), output.dtype(), "swiglu");

    int64_t numel = output.numel();

    if (gate.dtype() == DType::BF16)
    {
        dispatch_swiglu<__nv_bfloat16>(gate, up, output, numel, context.stream());
    }
    else
    {
        dispatch_swiglu<half>(gate, up, output, numel, context.stream());
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        FIREFLY_LOG_ERROR("cuda", "kernel launch failed operation=swiglu error={} code={}",
                          cudaGetErrorString(err), static_cast<int>(err));
    }
}

void swiglu_fused(const Tensor& projected, Tensor& output, const device::Context& context)
{
    require_float16_or_bfloat16(projected.dtype(), "swiglu_fused");
    require_same_dtype(projected.dtype(), output.dtype(), "swiglu_fused");
    if (projected.shape().empty() || output.shape().empty() || projected.numel() != output.numel() * 2 ||
        projected.shape().back() != output.shape().back() * 2 || output.shape().back() % 8 != 0)
    {
        throw std::runtime_error("swiglu_fused requires projected [..., 2 * width] and output [..., width]");
    }

    const int row_width = output.shape().back();
    if (projected.dtype() == DType::BF16)
        dispatch_swiglu_fused<__nv_bfloat16>(projected, output, row_width, context.stream());
    else
        dispatch_swiglu_fused<half>(projected, output, row_width, context.stream());

    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        FIREFLY_LOG_ERROR("cuda", "kernel launch failed operation=swiglu_fused error={} code={}",
                          cudaGetErrorString(error), static_cast<int>(error));
}

}  // namespace firefly::kernels
