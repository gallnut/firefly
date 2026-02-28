#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "firefly/kernels.h"

namespace firefly::kernels
{

// Default GemmShape for large K, small M
using ShapeThreadBlock = cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeWarp = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeInst = cutlass::gemm::GemmShape<16, 8, 16>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>;

using GemmTN = cutlass::gemm::device::Gemm<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
                                           cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::RowMajor,
                                           cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                           ShapeThreadBlock, ShapeWarp, ShapeInst, EpilogueOp>;

using GemmNN = cutlass::gemm::device::Gemm<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
                                           cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::RowMajor,
                                           cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                           ShapeThreadBlock, ShapeWarp, ShapeInst, EpilogueOp>;

void matmul(const Tensor& input, const Tensor& weight, Tensor& output)
{
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
        std::cerr << "GEMM Error: Dimension mismatch A[" << M << "," << K << "] B[" << weight.shape()[0] << ","
                  << weight.shape()[1] << "]\n";
        return;
    }

    if (M != 1)
    {  // Only log interesting M sizes
       // std::cout << "GEMM Shape Log: M=" << M << " N=" << N_out << " K=" << K << std::endl;
    }

    cutlass::half_t alpha = cutlass::half_t(1.0f);
    cutlass::half_t beta = cutlass::half_t(0.0f);

    if (b_is_transposed)
    {
        typename GemmTN::Arguments args({M, N_out, K},                             // problem_size
                                        {(cutlass::half_t*)input.data(), K},       // tensor_A
                                        {(cutlass::half_t*)weight.data(), K},      // tensor_B (ColMajor ldb = K)
                                        {(cutlass::half_t*)output.data(), N_out},  // tensor_C
                                        {(cutlass::half_t*)output.data(), N_out},  // tensor_D
                                        {alpha, beta}                              // epilogue
        );

        GemmTN          gemm_op;
        cutlass::Status status = gemm_op(args, nullptr, get_default_stream());
        if (status != cutlass::Status::kSuccess)
        {
            std::cerr << "Cutlass GemmTN Failed: " << cutlassGetStatusString(status) << std::endl;
        }
    }
    else
    {
        typename GemmNN::Arguments args({M, N_out, K},                             // problem_size
                                        {(cutlass::half_t*)input.data(), K},       // tensor_A
                                        {(cutlass::half_t*)weight.data(), N_out},  // tensor_B
                                        {(cutlass::half_t*)output.data(), N_out},  // tensor_C
                                        {(cutlass::half_t*)output.data(), N_out},  // tensor_D
                                        {alpha, beta}                              // epilogue
        );

        GemmNN          gemm_op;
        cutlass::Status status = gemm_op(args, nullptr, get_default_stream());
        if (status != cutlass::Status::kSuccess)
        {
            std::cerr << "Cutlass GemmNN Failed: " << cutlassGetStatusString(status) << std::endl;
        }
    }
}

}  // namespace firefly::kernels
