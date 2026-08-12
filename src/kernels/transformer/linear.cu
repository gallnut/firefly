#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <sstream>

#include "firefly/device/error.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"
#include "firefly/kernels/transformer/linear.h"

namespace firefly::kernels
{
using detail::CudaScalar;

namespace
{
template <typename scalar_t>
__global__ void gemv_warp_row_vector_kernel(const scalar_t* __restrict__ weight,
                                            const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                                            int rows, int columns)
{
    extern __shared__ unsigned char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    constexpr int vector_elements = sizeof(float4) / sizeof(scalar_t);
    for (int index = threadIdx.x * vector_elements; index < columns; index += blockDim.x * vector_elements)
    {
        *reinterpret_cast<float4*>(shared_input + index) =
            *reinterpret_cast<const float4*>(input + index);
    }
    __syncthreads();

    constexpr int warp_size = 32;
    const int     row = blockIdx.x * (blockDim.x / warp_size) + (threadIdx.x / warp_size);
    if (row >= rows) return;
    const int lane = threadIdx.x & (warp_size - 1);
    const scalar_t* row_weight = weight + static_cast<int64_t>(row) * columns;

    float sum = 0.0f;
    for (int offset = lane * vector_elements; offset < columns; offset += warp_size * vector_elements)
    {
        const float4 weight_vector = *reinterpret_cast<const float4*>(row_weight + offset);
        const float4 input_vector = *reinterpret_cast<const float4*>(shared_input + offset);
        const scalar_t* weight_values = reinterpret_cast<const scalar_t*>(&weight_vector);
        const scalar_t* input_values = reinterpret_cast<const scalar_t*>(&input_vector);
#pragma unroll
        for (int element = 0; element < vector_elements; ++element)
            sum += CudaScalar<scalar_t>::to_float(weight_values[element]) *
                   CudaScalar<scalar_t>::to_float(input_values[element]);
    }
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) output[row] = CudaScalar<scalar_t>::from_float(sum);
}

const char* cublas_status_string(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "not supported";
        default:
            return "unknown";
    }
}

struct CublasState
{
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;

    ~CublasState()
    {
        if (handle) cublasDestroy(handle);
    }
};


Result<CublasState*> get_cublas_state()
{
    static thread_local CublasState state;
    if (state.handle) return &state;
    const cublasStatus_t create_status = cublasCreate(&state.handle);
    if (create_status != CUBLAS_STATUS_SUCCESS)
        return unexpected(Error{ErrorCode::Cublas,
                                "cublasCreate failed: " + std::string(cublas_status_string(create_status)),
                                static_cast<int>(create_status)});
    const cublasStatus_t math_status = cublasSetMathMode(state.handle, CUBLAS_TENSOR_OP_MATH);
    if (math_status != CUBLAS_STATUS_SUCCESS)
    {
        cublasDestroy(state.handle);
        state.handle = nullptr;
        return unexpected(Error{ErrorCode::Cublas,
                                "cublasSetMathMode failed: " + std::string(cublas_status_string(math_status)),
                                static_cast<int>(math_status)});
    }
    return &state;
}

cudaDataType_t cuda_data_type(DType dtype)
{
    return dtype == DType::BF16 ? CUDA_R_16BF : CUDA_R_16F;
}

std::string shape_string(const Tensor& tensor)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < tensor.shape().size(); ++i)
    {
        if (i > 0)
        {
            os << ", ";
        }
        os << tensor.shape()[i];
    }
    os << "]";
    return os.str();
}

}  // namespace

Status matmul(const Tensor& input, const Tensor& weight, Tensor& output, const device::Context& context)
{
    FIREFLY_TRY(require_float16_or_bfloat16(input.dtype(), "matmul"));
    FIREFLY_TRY(require_same_dtype(input.dtype(), weight.dtype(), "matmul"));
    FIREFLY_TRY(require_same_dtype(input.dtype(), output.dtype(), "matmul"));

    if (input.shape().empty() || weight.shape().size() != 2)
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "matmul expects non-empty input and rank-2 weight, got input " +
                                    shape_string(input) + " weight " + shape_string(weight)});
    }

    // A: input [M, K]
    // B: weight [N, K]
    // C: output [M, N]

    int M = input.numel() / input.shape().back();
    int K = input.shape().back();

    int  N_out;
    bool b_is_transposed;

    if (weight.shape()[1] == K)
    {
        // B is [N_out, K], compute A * B^T
        N_out = weight.shape()[0];
        b_is_transposed = true;
    }
    else if (weight.shape()[0] == K)
    {
        // B is [K, N_out], compute A * B
        N_out = weight.shape()[1];
        b_is_transposed = false;
    }
    else
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "matmul dimension mismatch: input " + shape_string(input) + " weight " +
                                    shape_string(weight)});
    }

    if (output.numel() != static_cast<int64_t>(M) * N_out)
    {
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "matmul output shape mismatch: input " + shape_string(input) + " weight " +
                                    shape_string(weight) + " output " + shape_string(output)});
    }

    if (M == 1 && K % 256 == 0 && K <= 32768 && weight.strides().size() == 2 && weight.strides()[1] == 1 &&
        input.strides().back() == 1 && output.strides().back() == 1)
    {
        constexpr int threads = 256;
        constexpr int rows_per_block = threads / 32;
        const int     blocks = (N_out + rows_per_block - 1) / rows_per_block;
        const size_t  shared_bytes = static_cast<size_t>(K) * dtype_size(input.dtype());
        if (input.dtype() == DType::BF16)
        {
            gemv_warp_row_vector_kernel<__nv_bfloat16><<<blocks, threads, shared_bytes, context.stream()>>>(
                static_cast<const __nv_bfloat16*>(weight.data()),
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(output.data()), N_out, K);
        }
        else
        {
            gemv_warp_row_vector_kernel<half><<<blocks, threads, shared_bytes, context.stream()>>>(
                static_cast<const half*>(weight.data()), static_cast<const half*>(input.data()),
                static_cast<half*>(output.data()), N_out, K);
        }
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return unexpected(device::cuda_error(error, "launch GEMV kernel"));
        return {};
    }

    CublasState* state = FIREFLY_TRY(get_cublas_state());
    if (state->stream != context.stream())
    {
        const cublasStatus_t stream_status = cublasSetStream(state->handle, context.stream());
        if (stream_status != CUBLAS_STATUS_SUCCESS)
            return unexpected(Error{ErrorCode::Cublas,
                                    "cublasSetStream failed: " +
                                        std::string(cublas_status_string(stream_status)),
                                    static_cast<int>(stream_status)});
        state->stream = context.stream();
    }

    float          alpha = 1.0f;
    float          beta = 0.0f;
    cudaDataType_t type = cuda_data_type(input.dtype());

    // Tensors are stored row-major. cuBLAS is column-major, so compute:
    // C_rm[M, N] = A_rm[M, K] * B_rm[K, N]
    // as C_cm[N, M] = B_cm[N, K] * A_cm[K, M].
    cublasOperation_t weight_op = b_is_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    int               weight_ld = b_is_transposed ? K : N_out;

    cublasStatus_t status =
        cublasGemmEx(state->handle, weight_op, CUBLAS_OP_N, N_out, M, K, &alpha, weight.data(), type, weight_ld,
                     input.data(), type, K, &beta, output.data(), type, N_out, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return unexpected(Error{ErrorCode::Cublas,
                                "cuBLAS Gemm failed: " + std::string(cublas_status_string(status)) +
                                    " input " + shape_string(input) + " weight " + shape_string(weight) +
                                    " output " + shape_string(output) + " dtype " +
                                    std::string(dtype_to_string(input.dtype())),
                                static_cast<int>(status)});
    }
    return {};
}

}  // namespace firefly::kernels
