#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "firefly/kernels.h"

#define WARP_SIZE 32
#define LOAD128BITS(value) (reinterpret_cast<const float4*>(&(value))[0])
#define STORE128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

namespace firefly::kernels
{

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

template <int NUM_THREADS>
__global__ void rms_norm_kernel_optimized(const half* __restrict__ input, const half* __restrict__ weight,
                                          half* __restrict__ output, int hidden_size, float epsilon)
{
    // Rows: each block handles one token
    // GridDim.x = Batch * Seq
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int   offset = bid * hidden_size;
    const half* row_input = input + offset;
    half*       row_output = output + offset;

    float sum_sq = 0.0f;

    // Load 8 halfs (128-bit) per iteration
    using vec_t = float4;
    constexpr int VEC_SIZE = 8;

    // Stride loop
    for (int idx = tid * VEC_SIZE; idx < hidden_size; idx += NUM_THREADS * VEC_SIZE)
    {
        vec_t in_vec = LOAD128BITS(row_input[idx]);
        half* in_h = reinterpret_cast<half*>(&in_vec);

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            float val = __half2float(in_h[i]);
            sum_sq += val * val;
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
        vec_t in_vec = LOAD128BITS(row_input[idx]);
        half* in_h = reinterpret_cast<half*>(&in_vec);

        vec_t w_vec = LOAD128BITS(weight[idx]);
        half* w_h = reinterpret_cast<half*>(&w_vec);

        vec_t out_vec;
        half* out_h = reinterpret_cast<half*>(&out_vec);

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            float val = __half2float(in_h[i]);
            float w = __half2float(w_h[i]);

            // Formula: x * inv_rms * weight
            out_h[i] = __float2half(val * inv_rms * w);
        }

        STORE128BITS(row_output[idx]) = out_vec;
    }
}

#define LAUNCH_RMS_NORM_OPTIMIZED(THREADS)                                                                             \
    rms_norm_kernel_optimized<THREADS><<<grid, THREADS, 0, stream>>>((const half*)input.data(),                        \
                                                                     (const half*)weight.data(), (half*)output.data(), \
                                                                     hidden_size, static_cast<float>(epsilon));

void rms_norm(const Tensor& input, const Tensor& weight, Tensor& output, double epsilon)
{
    const int hidden_size = input.shape().back();
    const int num_tokens = input.numel() / hidden_size;

    dim3         grid(num_tokens);
    cudaStream_t stream = get_default_stream();  // TODO: get stream from context if available

    // Heuristic:
    // We process 8 elements per thread (float4 load of halfs).
    // Ideally we want one thread block to cover the entire hidden_size without looping too much,
    // but bounded by Max Threads (1024).
    //
    // Dispatch based on hidden_size prevents register pressure changes or occupancy issues
    // from affecting all sizes, and allows compiler to optimize the block reduction for constant block size.

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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in rms_norm: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace firefly::kernels
