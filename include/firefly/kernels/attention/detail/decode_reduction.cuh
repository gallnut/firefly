#pragma once

#include "firefly/kernels/detail/cuda_scalar.cuh"

namespace firefly::kernels::attention_detail
{
using firefly::kernels::detail::CudaScalar;

template <typename scalar_t>
__global__ void paged_decode_split_reduce_kernel(const float* __restrict__ partial_m,
                                                 const float* __restrict__ partial_l,
                                                 const float* __restrict__ partial_acc, scalar_t* __restrict__ O,
                                                 int num_heads, int head_dim, int num_splits)
{
    __shared__ float final_m;
    __shared__ float final_l;

    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    int base = (batch_idx * num_heads + head_idx) * num_splits;

    if (tid == 0)
    {
        float m = -1.0e30f;
        for (int s = 0; s < num_splits; ++s)
        {
            m = fmaxf(m, partial_m[base + s]);
        }

        float l = 0.0f;
        if (m > -5.0e29f)
        {
            for (int s = 0; s < num_splits; ++s)
            {
                float ms = partial_m[base + s];
                if (ms > -5.0e29f)
                {
                    l += partial_l[base + s] * __expf(ms - m);
                }
            }
        }
        final_m = m;
        final_l = l;
    }
    __syncthreads();

    if (tid < head_dim)
    {
        float out = 0.0f;
        if (final_l > 0.0f)
        {
            for (int s = 0; s < num_splits; ++s)
            {
                float ms = partial_m[base + s];
                if (ms > -5.0e29f)
                {
                    out += partial_acc[(int64_t)(base + s) * head_dim + tid] * __expf(ms - final_m);
                }
            }
            out /= final_l;
        }

        int64_t out_offset = (int64_t)batch_idx * num_heads * head_dim + (int64_t)head_idx * head_dim + tid;
        O[out_offset] = CudaScalar<scalar_t>::from_float(out);
    }
}

template <typename scalar_t>
__global__ void paged_decode_split_dual_reduce_kernel(const float* __restrict__ partial_m,
                                                      const float* __restrict__ partial_l,
                                                      const float* __restrict__ partial_acc, scalar_t* __restrict__ O,
                                                      int num_heads, int head_dim, int num_splits)
{
    __shared__ float final_m0;
    __shared__ float final_l0;
    __shared__ float final_m1;
    __shared__ float final_l1;

    int kv_pair_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int head0 = kv_pair_idx * 2;
    int head1 = head0 + 1;
    if (head1 >= num_heads) return;

    int base0 = (batch_idx * num_heads + head0) * num_splits;
    int base1 = (batch_idx * num_heads + head1) * num_splits;

    if (tid == 0)
    {
        float m0 = -1.0e30f;
        float m1 = -1.0e30f;
        for (int s = 0; s < num_splits; ++s)
        {
            m0 = fmaxf(m0, partial_m[base0 + s]);
            m1 = fmaxf(m1, partial_m[base1 + s]);
        }

        float l0 = 0.0f;
        float l1 = 0.0f;
        if (m0 > -5.0e29f)
        {
            for (int s = 0; s < num_splits; ++s)
            {
                float ms = partial_m[base0 + s];
                if (ms > -5.0e29f) l0 += partial_l[base0 + s] * __expf(ms - m0);
            }
        }
        if (m1 > -5.0e29f)
        {
            for (int s = 0; s < num_splits; ++s)
            {
                float ms = partial_m[base1 + s];
                if (ms > -5.0e29f) l1 += partial_l[base1 + s] * __expf(ms - m1);
            }
        }
        final_m0 = m0;
        final_l0 = l0;
        final_m1 = m1;
        final_l1 = l1;
    }
    __syncthreads();

    if (tid < head_dim)
    {
        float out0 = 0.0f;
        float out1 = 0.0f;
        if (final_l0 > 0.0f)
        {
            for (int s = 0; s < num_splits; ++s)
            {
                float ms = partial_m[base0 + s];
                if (ms > -5.0e29f)
                {
                    out0 += partial_acc[(int64_t)(base0 + s) * head_dim + tid] * __expf(ms - final_m0);
                }
            }
            out0 /= final_l0;
        }
        if (final_l1 > 0.0f)
        {
            for (int s = 0; s < num_splits; ++s)
            {
                float ms = partial_m[base1 + s];
                if (ms > -5.0e29f)
                {
                    out1 += partial_acc[(int64_t)(base1 + s) * head_dim + tid] * __expf(ms - final_m1);
                }
            }
            out1 /= final_l1;
        }

        O[(int64_t)batch_idx * num_heads * head_dim + (int64_t)head0 * head_dim + tid] =
            CudaScalar<scalar_t>::from_float(out0);
        O[(int64_t)batch_idx * num_heads * head_dim + (int64_t)head1 * head_dim + tid] =
            CudaScalar<scalar_t>::from_float(out1);
    }
}

}  // namespace firefly::kernels::attention_detail
