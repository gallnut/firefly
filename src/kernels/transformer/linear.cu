#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <sstream>

#include "firefly/kernels/transformer/linear.h"

namespace firefly::kernels
{

namespace
{
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

cublasHandle_t get_cublas_handle()
{
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle)
    {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string("cublasCreate failed: ") + cublas_status_string(status));
        }
    }
    return handle;
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

void matmul(const Tensor& input, const Tensor& weight, Tensor& output, const device::Context& context)
{
    require_float16_or_bfloat16(input.dtype(), "matmul");
    require_same_dtype(input.dtype(), weight.dtype(), "matmul");
    require_same_dtype(input.dtype(), output.dtype(), "matmul");

    if (input.shape().empty() || weight.shape().size() != 2)
    {
        throw std::runtime_error("matmul expects non-empty input and rank-2 weight, got input " + shape_string(input) +
                                 " weight " + shape_string(weight));
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
        throw std::runtime_error("matmul dimension mismatch: input " + shape_string(input) + " weight " +
                                 shape_string(weight));
    }

    if (output.numel() != static_cast<int64_t>(M) * N_out)
    {
        throw std::runtime_error("matmul output shape mismatch: input " + shape_string(input) + " weight " +
                                 shape_string(weight) + " output " + shape_string(output));
    }

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, context.stream());
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float          alpha = 1.0f;
    float          beta = 0.0f;
    cudaDataType_t type = cuda_data_type(input.dtype());

    // Tensors are stored row-major. cuBLAS is column-major, so compute:
    // C_rm[M, N] = A_rm[M, K] * B_rm[K, N]
    // as C_cm[N, M] = B_cm[N, K] * A_cm[K, M].
    cublasOperation_t weight_op = b_is_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    int               weight_ld = b_is_transposed ? K : N_out;

    cublasStatus_t status =
        cublasGemmEx(handle, weight_op, CUBLAS_OP_N, N_out, M, K, &alpha, weight.data(), type, weight_ld, input.data(),
                     type, K, &beta, output.data(), type, N_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("cuBLAS Gemm failed: " + std::string(cublas_status_string(status)) + " input " +
                                 shape_string(input) + " weight " + shape_string(weight) + " output " +
                                 shape_string(output) + " dtype " + std::string(dtype_to_string(input.dtype())));
    }
}

}  // namespace firefly::kernels
