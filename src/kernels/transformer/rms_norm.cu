#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "firefly/core/logging.h"
#include "firefly/device/error.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"
#include "firefly/kernels/transformer/rms_norm.h"

#define WARP_SIZE 32
#define LOAD128BITS(value) (reinterpret_cast<const float4*>(&(value))[0])
#define STORE128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

namespace firefly::kernels
{
using detail::CudaScalar;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, int NUM_THREADS>
__device__ __forceinline__ T block_reduce_sum(T val)
{
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ T  shared_warps[NUM_WARPS];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // 1. Warp Reduce
    val = warp_reduce_sum(val);

    // 2. First thread of each warp writes to shared mem
    if (lane == 0)
    {
        shared_warps[wid] = val;
    }
    __syncthreads();

    // 3. First warp reduces the partial sums
    val = (threadIdx.x < NUM_WARPS) ? shared_warps[threadIdx.x] : T(0);

    if (wid == 0)
    {
        val = warp_reduce_sum(val);
    }

    // Broadcast result to all threads (in shared memory)
    if (threadIdx.x == 0) shared_warps[0] = val;
    __syncthreads();

    return shared_warps[0];
}

template <typename scalar_t, int NUM_THREADS>
__global__ void rms_norm_kernel_optimized(const scalar_t* __restrict__ input, const scalar_t* __restrict__ weight,
                                          scalar_t* __restrict__ output, int hidden_size, float epsilon)
{
    // Rows: each block handles one token
    // GridDim.x = Batch * Seq
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int       offset = bid * hidden_size;
    const scalar_t* row_input = input + offset;
    scalar_t*       row_output = output + offset;

    float sum_sq = 0.0f;

    // Load 8 halfs (128-bit) per iteration
    using vec_t = float4;
    constexpr int VEC_SIZE = 8;

    // Stride loop
    for (int idx = tid * VEC_SIZE; idx < hidden_size; idx += NUM_THREADS * VEC_SIZE)
    {
        if (idx + VEC_SIZE <= hidden_size)
        {
            vec_t     in_vec = LOAD128BITS(row_input[idx]);
            scalar_t* in_h = reinterpret_cast<scalar_t*>(&in_vec);

#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
            {
                float val = CudaScalar<scalar_t>::to_float(in_h[i]);
                sum_sq += val * val;
            }
        }
        else
        {
            for (int i = idx; i < hidden_size; ++i)
            {
                float val = CudaScalar<scalar_t>::to_float(row_input[i]);
                sum_sq += val * val;
            }
        }
    }

    // Shared memory for broadcasting sum_sq
    __shared__ float s_sum_sq;

    // Block Reduction
    float reduced_sum_sq = block_reduce_sum<float, NUM_THREADS>(sum_sq);

    if (tid == 0)
    {
        s_sum_sq = reduced_sum_sq;
    }
    __syncthreads();

    // Compute Inverse RMS safely using float
    float mean_sq = s_sum_sq / (float)hidden_size;
    float inv_rms = rsqrtf(mean_sq + epsilon);

    for (int idx = tid * VEC_SIZE; idx < hidden_size; idx += NUM_THREADS * VEC_SIZE)
    {
        if (idx + VEC_SIZE <= hidden_size)
        {
            vec_t     in_vec = LOAD128BITS(row_input[idx]);
            scalar_t* in_h = reinterpret_cast<scalar_t*>(&in_vec);

            vec_t     w_vec = LOAD128BITS(weight[idx]);
            scalar_t* w_h = reinterpret_cast<scalar_t*>(&w_vec);

            vec_t     out_vec;
            scalar_t* out_h = reinterpret_cast<scalar_t*>(&out_vec);

#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
            {
                float val = CudaScalar<scalar_t>::to_float(in_h[i]);
                float w = CudaScalar<scalar_t>::to_float(w_h[i]);

                out_h[i] = CudaScalar<scalar_t>::from_float(val * inv_rms * w);
            }

            STORE128BITS(row_output[idx]) = out_vec;
        }
        else
        {
            for (int i = idx; i < hidden_size; ++i)
            {
                float val = CudaScalar<scalar_t>::to_float(row_input[i]);
                float w = CudaScalar<scalar_t>::to_float(weight[i]);
                row_output[i] = CudaScalar<scalar_t>::from_float(val * inv_rms * w);
            }
        }
    }
}

template <typename scalar_t, int NUM_THREADS, int FIXED_HIDDEN_SIZE = 0>
__global__ void add_rms_norm_kernel(scalar_t* __restrict__ residual, const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight, scalar_t* __restrict__ output,
                                    int hidden_size, float epsilon)
{
    const int tid = threadIdx.x;
    const int row_size = FIXED_HIDDEN_SIZE == 0 ? hidden_size : FIXED_HIDDEN_SIZE;
    const int offset = blockIdx.x * row_size;
    scalar_t* residual_row = residual + offset;
    const scalar_t* input_row = input + offset;
    scalar_t* output_row = output + offset;

    using vec_t = float4;
    constexpr int VEC_SIZE = sizeof(vec_t) / sizeof(scalar_t);

    float sum_sq = 0.0f;
    for (int index = tid * VEC_SIZE; index < row_size; index += NUM_THREADS * VEC_SIZE)
    {
        if (index + VEC_SIZE <= row_size)
        {
            vec_t residual_vec = LOAD128BITS(residual_row[index]);
            vec_t input_vec = LOAD128BITS(input_row[index]);
            scalar_t* residual_values = reinterpret_cast<scalar_t*>(&residual_vec);
            const scalar_t* input_values = reinterpret_cast<const scalar_t*>(&input_vec);

#pragma unroll
            for (int element = 0; element < VEC_SIZE; ++element)
            {
                const float sum = CudaScalar<scalar_t>::to_float(residual_values[element]) +
                                  CudaScalar<scalar_t>::to_float(input_values[element]);
                residual_values[element] = CudaScalar<scalar_t>::from_float(sum);
                const float rounded = CudaScalar<scalar_t>::to_float(residual_values[element]);
                sum_sq += rounded * rounded;
            }
            STORE128BITS(residual_row[index]) = residual_vec;
        }
        else
        {
            for (int element = index; element < row_size; ++element)
            {
                const float sum = CudaScalar<scalar_t>::to_float(residual_row[element]) +
                                  CudaScalar<scalar_t>::to_float(input_row[element]);
                const scalar_t rounded = CudaScalar<scalar_t>::from_float(sum);
                residual_row[element] = rounded;
                const float value = CudaScalar<scalar_t>::to_float(rounded);
                sum_sq += value * value;
            }
        }
    }

    const float reduced_sum_sq = block_reduce_sum<float, NUM_THREADS>(sum_sq);
    const float inv_rms = rsqrtf(reduced_sum_sq / static_cast<float>(row_size) + epsilon);

    for (int index = tid * VEC_SIZE; index < row_size; index += NUM_THREADS * VEC_SIZE)
    {
        if (index + VEC_SIZE <= row_size)
        {
            vec_t residual_vec = LOAD128BITS(residual_row[index]);
            vec_t weight_vec = LOAD128BITS(weight[index]);
            vec_t output_vec;
            const scalar_t* residual_values = reinterpret_cast<const scalar_t*>(&residual_vec);
            const scalar_t* weight_values = reinterpret_cast<const scalar_t*>(&weight_vec);
            scalar_t* output_values = reinterpret_cast<scalar_t*>(&output_vec);

#pragma unroll
            for (int element = 0; element < VEC_SIZE; ++element)
            {
                const float value = CudaScalar<scalar_t>::to_float(residual_values[element]);
                const float scale = CudaScalar<scalar_t>::to_float(weight_values[element]);
                output_values[element] = CudaScalar<scalar_t>::from_float(value * inv_rms * scale);
            }
            STORE128BITS(output_row[index]) = output_vec;
        }
        else
        {
            for (int element = index; element < row_size; ++element)
            {
                const float value = CudaScalar<scalar_t>::to_float(residual_row[element]);
                const float scale = CudaScalar<scalar_t>::to_float(weight[element]);
                output_row[element] = CudaScalar<scalar_t>::from_float(value * inv_rms * scale);
            }
        }
    }
}

template <typename scalar_t>
void dispatch_rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, int hidden_size, int num_tokens,
                       double epsilon, cudaStream_t stream)
{
    dim3 grid(num_tokens);

#define LAUNCH_RMS_NORM_OPTIMIZED(THREADS)                                                       \
    rms_norm_kernel_optimized<scalar_t, THREADS><<<grid, THREADS, 0, stream>>>(                  \
        static_cast<const scalar_t*>(input.data()), static_cast<const scalar_t*>(weight.data()), \
        static_cast<scalar_t*>(output.data()), hidden_size, static_cast<float>(epsilon));

    if (hidden_size <= 512)
    {
        // e.g. 512 / 8 = 64 threads
        LAUNCH_RMS_NORM_OPTIMIZED(64);
    }
    else if (hidden_size <= 1024)
    {
        // e.g. 1024 / 8 = 128 threads
        LAUNCH_RMS_NORM_OPTIMIZED(128);
    }
    else if (hidden_size <= 2048)
    {
        // e.g. 2048 / 8 = 256 threads
        LAUNCH_RMS_NORM_OPTIMIZED(256);
    }
    else if (hidden_size <= 4096)
    {
        // e.g. 4096 / 8 = 512 threads
        LAUNCH_RMS_NORM_OPTIMIZED(512);
    }
    else
    {
        // For very large sizes, cap at 1024 threads (handling 8192 elements per iteration)
        LAUNCH_RMS_NORM_OPTIMIZED(1024);
    }

#undef LAUNCH_RMS_NORM_OPTIMIZED
}

Status rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(input.dtype(), "rms_norm"));
    FIREFLY_TRY(require_same_dtype(input.dtype(), weight.dtype(), "rms_norm"));
    FIREFLY_TRY(require_same_dtype(input.dtype(), output.dtype(), "rms_norm"));
    if (input.shape().empty() || input.shape().back() <= 0 || input.numel() != output.numel() ||
        weight.numel() != input.shape().back())
        return unexpected(Error{ErrorCode::InvalidArgument, "rms_norm tensor shapes are incompatible"});

    const int hidden_size = input.shape().back();
    const int num_tokens = input.numel() / hidden_size;

    cudaStream_t stream = context.stream();

    if (input.dtype() == DType::BF16)
    {
        dispatch_rms_norm<__nv_bfloat16>(input, weight, output, hidden_size, num_tokens, epsilon, stream);
    }
    else
    {
        dispatch_rms_norm<half>(input, weight, output, hidden_size, num_tokens, epsilon, stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return unexpected(device::cuda_error(err, "launch rms_norm kernel"));
    return {};
}

template <typename scalar_t>
void dispatch_add_rms_norm(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output,
                           int hidden_size, int num_tokens, double epsilon, cudaStream_t stream)
{
    dim3 grid(num_tokens);
    if (hidden_size == 1024)
        add_rms_norm_kernel<scalar_t, 128, 1024><<<grid, 128, 0, stream>>>(
            static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
            static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), hidden_size,
            static_cast<float>(epsilon));
    else if (hidden_size <= 512)
        add_rms_norm_kernel<scalar_t, 64><<<grid, 64, 0, stream>>>(
            static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
            static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), hidden_size,
            static_cast<float>(epsilon));
    else if (hidden_size <= 1024)
        add_rms_norm_kernel<scalar_t, 128><<<grid, 128, 0, stream>>>(
            static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
            static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), hidden_size,
            static_cast<float>(epsilon));
    else if (hidden_size <= 2048)
        add_rms_norm_kernel<scalar_t, 256><<<grid, 256, 0, stream>>>(
            static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
            static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), hidden_size,
            static_cast<float>(epsilon));
    else if (hidden_size <= 4096)
        add_rms_norm_kernel<scalar_t, 512><<<grid, 512, 0, stream>>>(
            static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
            static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), hidden_size,
            static_cast<float>(epsilon));
    else
        add_rms_norm_kernel<scalar_t, 1024><<<grid, 1024, 0, stream>>>(
            static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
            static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), hidden_size,
            static_cast<float>(epsilon));
}

Status add_rms_norm(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                    const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(residual.dtype(), "add_rms_norm"));
    FIREFLY_TRY(require_same_dtype(residual.dtype(), input.dtype(), "add_rms_norm"));
    FIREFLY_TRY(require_same_dtype(residual.dtype(), weight.dtype(), "add_rms_norm"));
    FIREFLY_TRY(require_same_dtype(residual.dtype(), output.dtype(), "add_rms_norm"));
    if (residual.shape().empty() || residual.shape().back() <= 0 || residual.numel() != input.numel() ||
        residual.numel() != output.numel() || weight.numel() != residual.shape().back())
        return unexpected(Error{ErrorCode::InvalidArgument, "add_rms_norm tensor shapes are incompatible"});

    const int hidden_size = residual.shape().back();
    const int num_tokens = residual.numel() / hidden_size;
    if (residual.dtype() == DType::BF16)
        dispatch_add_rms_norm<__nv_bfloat16>(residual, input, weight, output, hidden_size, num_tokens, epsilon,
                                             context.stream());
    else
        dispatch_add_rms_norm<half>(residual, input, weight, output, hidden_size, num_tokens, epsilon,
                                    context.stream());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "launch add_rms_norm kernel"));
    return {};
}

template <typename scalar_t, bool AddResidual>
__global__ void zero_centered_rms_norm_kernel(scalar_t* residual, const scalar_t* input, const scalar_t* weight,
                                              scalar_t* output, int row_size, float epsilon)
{
    constexpr int threads = 128;
    const int row = blockIdx.x;
    scalar_t* residual_row = residual + static_cast<int64_t>(row) * row_size;
    const scalar_t* input_row = input + static_cast<int64_t>(row) * row_size;
    scalar_t* output_row = output + static_cast<int64_t>(row) * row_size;

    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < row_size; index += blockDim.x)
    {
        float value = CudaScalar<scalar_t>::to_float(input_row[index]);
        if constexpr (AddResidual)
        {
            value += CudaScalar<scalar_t>::to_float(residual_row[index]);
            scalar_t rounded = CudaScalar<scalar_t>::from_float(value);
            residual_row[index] = rounded;
            value = CudaScalar<scalar_t>::to_float(rounded);
        }
        square_sum += value * value;
    }
    const float inverse_rms =
        rsqrtf(block_reduce_sum<float, threads>(square_sum) / static_cast<float>(row_size) + epsilon);
    for (int index = threadIdx.x; index < row_size; index += blockDim.x)
    {
        const float value =
            CudaScalar<scalar_t>::to_float(AddResidual ? residual_row[index] : input_row[index]);
        const float scale = 1.0f + CudaScalar<scalar_t>::to_float(weight[index]);
        output_row[index] = CudaScalar<scalar_t>::from_float(value * inverse_rms * scale);
    }
}

template <typename scalar_t, bool AddResidual>
void dispatch_zero_centered_rms_norm(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output,
                                     int row_size, int rows, double epsilon, cudaStream_t stream)
{
    zero_centered_rms_norm_kernel<scalar_t, AddResidual><<<rows, 128, 0, stream>>>(
        static_cast<scalar_t*>(residual.data()), static_cast<const scalar_t*>(input.data()),
        static_cast<const scalar_t*>(weight.data()), static_cast<scalar_t*>(output.data()), row_size,
        static_cast<float>(epsilon));
}

Status rms_norm_zero_centered(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon,
                              const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(input.dtype(), "rms_norm_zero_centered"));
    FIREFLY_TRY(require_same_dtype(input.dtype(), weight.dtype(), "rms_norm_zero_centered"));
    FIREFLY_TRY(require_same_dtype(input.dtype(), output.dtype(), "rms_norm_zero_centered"));
    if (input.shape().empty() || input.shape().back() <= 0 || input.numel() != output.numel() ||
        weight.numel() != input.shape().back())
        return unexpected(Error{ErrorCode::InvalidArgument, "zero-centered rms_norm tensor shapes are incompatible"});
    const int row_size = input.shape().back();
    const int rows = input.numel() / row_size;
    Tensor empty;
    if (input.dtype() == DType::BF16)
        dispatch_zero_centered_rms_norm<__nv_bfloat16, false>(empty, input, weight, output, row_size, rows, epsilon,
                                                              context.stream());
    else
        dispatch_zero_centered_rms_norm<half, false>(empty, input, weight, output, row_size, rows, epsilon,
                                                     context.stream());
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return unexpected(device::cuda_error(error, "launch zero-centered rms_norm kernel"));
    return {};
}

Status add_rms_norm_zero_centered(Tensor& residual, const Tensor& input, const Tensor& weight, Tensor& output,
                                  double epsilon, const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(residual.dtype(), "add_rms_norm_zero_centered"));
    FIREFLY_TRY(require_same_dtype(residual.dtype(), input.dtype(), "add_rms_norm_zero_centered"));
    FIREFLY_TRY(require_same_dtype(residual.dtype(), weight.dtype(), "add_rms_norm_zero_centered"));
    FIREFLY_TRY(require_same_dtype(residual.dtype(), output.dtype(), "add_rms_norm_zero_centered"));
    if (residual.shape().empty() || residual.shape().back() <= 0 || residual.numel() != input.numel() ||
        residual.numel() != output.numel() || weight.numel() != residual.shape().back())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "zero-centered add_rms_norm tensor shapes are incompatible"});
    const int row_size = residual.shape().back();
    const int rows = residual.numel() / row_size;
    if (residual.dtype() == DType::BF16)
        dispatch_zero_centered_rms_norm<__nv_bfloat16, true>(residual, input, weight, output, row_size, rows,
                                                             epsilon, context.stream());
    else
        dispatch_zero_centered_rms_norm<half, true>(residual, input, weight, output, row_size, rows, epsilon,
                                                    context.stream());
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return unexpected(device::cuda_error(error, "launch zero-centered add_rms_norm kernel"));
    return {};
}

}  // namespace firefly::kernels
