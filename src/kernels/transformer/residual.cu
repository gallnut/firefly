#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include "firefly/core/logging.h"
#include "firefly/device/error.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"
#include "firefly/kernels/transformer/residual.h"

// Vectorized load/store types for half
#define LOAD128BITS(value) (*reinterpret_cast<const float4*>(&(value)))
#define STORE128BITS(value) (*reinterpret_cast<float4*>(&(value)))

namespace firefly::kernels
{
using detail::CudaScalar;

template <typename scalar_t, int VEC_SIZE = 8>
__global__ void add_inplace_kernel_optimized(scalar_t* __restrict__ x, const scalar_t* __restrict__ y, int size)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (idx + VEC_SIZE > size)
    {
        return;
    }

    // Vectorized loads
    float4 x_vec = LOAD128BITS(x[idx]);
    float4 y_vec = LOAD128BITS(y[idx]);

    scalar_t* x_h = reinterpret_cast<scalar_t*>(&x_vec);
    scalar_t* y_h = reinterpret_cast<scalar_t*>(&y_vec);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
    {
        // use float addition
        float a = CudaScalar<scalar_t>::to_float(x_h[i]);
        float b = CudaScalar<scalar_t>::to_float(y_h[i]);
        x_h[i] = CudaScalar<scalar_t>::from_float(a + b);
    }

    // Vectorized store back to x
    STORE128BITS(x[idx]) = x_vec;
}

template <typename scalar_t>
__global__ void add_inplace_kernel(scalar_t* x, const scalar_t* y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float a = CudaScalar<scalar_t>::to_float(x[idx]);
        float b = CudaScalar<scalar_t>::to_float(y[idx]);
        x[idx] = CudaScalar<scalar_t>::from_float(a + b);
    }
}

template <typename scalar_t>
void dispatch_add_inplace(Tensor& x, const Tensor& y, int64_t numel, cudaStream_t stream)
{
    // Check alignment for vectorized kernel
    bool is_aligned = (reinterpret_cast<uintptr_t>(x.data()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(y.data()) % 16 == 0) && (numel % 8 == 0);

    if (is_aligned)
    {
        constexpr int vec_size = 8;
        int           threads = 256;
        int           num_blocks = (numel + (threads * vec_size) - 1) / (threads * vec_size);
        add_inplace_kernel_optimized<scalar_t, vec_size><<<num_blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(x.data()), static_cast<const scalar_t*>(y.data()), static_cast<int>(numel));
    }
    else
    {
        int threads = 256;
        int num_blocks = (numel + threads - 1) / threads;
        add_inplace_kernel<scalar_t><<<num_blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(x.data()), static_cast<const scalar_t*>(y.data()), static_cast<int>(numel));
    }
}

Status add_inplace(Tensor& x, const Tensor& y, const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(x.dtype(), "add_inplace"));
    FIREFLY_TRY(require_same_dtype(x.dtype(), y.dtype(), "add_inplace"));
    if (x.numel() != y.numel())
        return unexpected(Error{ErrorCode::InvalidArgument, "add_inplace tensor sizes differ"});

    int64_t numel = x.numel();

    if (x.dtype() == DType::BF16)
    {
        dispatch_add_inplace<__nv_bfloat16>(x, y, numel, context.stream());
    }
    else
    {
        dispatch_add_inplace<half>(x, y, numel, context.stream());
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return unexpected(device::cuda_error(err, "launch residual add kernel"));
    return {};
}

}  // namespace firefly::kernels
