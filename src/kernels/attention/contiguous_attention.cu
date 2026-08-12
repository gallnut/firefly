#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "firefly/kernels/attention/detail/cuda_utils.cuh"
#include "firefly/kernels/attention/detail/launch.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"

namespace firefly::kernels
{
using attention_detail::warp_reduce_sum;
using detail::CudaScalar;
namespace
{
template <typename scalar_t>
__global__ void causal_attention_scalar_kernel(const scalar_t* Q, const scalar_t* K, const scalar_t* V, scalar_t* O,
                                               int query_len, int kv_len, int num_heads, int num_kv_heads,
                                               int head_dim, float scale, bool causal)
{
    extern __shared__ float reduce[];
    __shared__ float        online_softmax[4];

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int64_t stride_b_q = (int64_t)query_len * num_heads * head_dim;
    int64_t stride_b_kv = (int64_t)kv_len * num_kv_heads * head_dim;
    int64_t stride_s_q = num_heads * head_dim;
    int64_t stride_s_kv = num_kv_heads * head_dim;

    const scalar_t* q_ptr = Q + batch_idx * stride_b_q + query_idx * stride_s_q + head_idx * head_dim;
    const scalar_t* k_base = K + batch_idx * stride_b_kv + kv_head_idx * head_dim;
    const scalar_t* v_base = V + batch_idx * stride_b_kv + kv_head_idx * head_dim;
    scalar_t*       o_ptr = O + batch_idx * stride_b_q + query_idx * stride_s_q + head_idx * head_dim;

    float acc = 0.0f;
    if (tid == 0)
    {
        online_softmax[0] = -1.0e30f;
        online_softmax[1] = 0.0f;
    }
    __syncthreads();

    const int key_limit = causal ? query_idx + 1 : kv_len;
    for (int key_idx = 0; key_idx < key_limit; ++key_idx)
    {
        const scalar_t* k_ptr = k_base + key_idx * stride_s_kv;

        float partial_dot = 0.0f;
        if (tid < head_dim)
        {
            partial_dot = CudaScalar<scalar_t>::to_float(q_ptr[tid]) * CudaScalar<scalar_t>::to_float(k_ptr[tid]);
        }

        partial_dot = warp_reduce_sum(partial_dot);
        int lane = tid & 31;
        int warp = tid >> 5;
        if (lane == 0) reduce[warp] = partial_dot;
        __syncthreads();
        if (warp == 0)
        {
            int   warp_count = (blockDim.x + 31) >> 5;
            float dot = lane < warp_count ? reduce[lane] : 0.0f;
            dot = warp_reduce_sum(dot);
            if (lane == 0)
            {
                float score = dot * scale;
                float m_next = fmaxf(online_softmax[0], score);
                float alpha = expf(score - m_next);
                float beta = expf(online_softmax[0] - m_next);
                online_softmax[0] = m_next;
                online_softmax[1] = online_softmax[1] * beta + alpha;
                online_softmax[2] = alpha;
                online_softmax[3] = beta;
            }
        }
        __syncthreads();

        if (tid < head_dim)
        {
            const scalar_t* v_ptr = v_base + key_idx * stride_s_kv;
            acc = acc * online_softmax[3] + CudaScalar<scalar_t>::to_float(v_ptr[tid]) * online_softmax[2];
        }
    }

    if (tid < head_dim)
    {
        o_ptr[tid] = CudaScalar<scalar_t>::from_float(acc / (online_softmax[1] + 1e-6f));
    }
}

template <typename scalar_t>
Status launch_scalar_prefill(Tensor& q, Tensor& k, Tensor& v, Tensor& output, int kv_head_num, int query_len,
                           int kv_len,
                           int batch_size, int num_heads, int head_dim, float scale, bool causal,
                           cudaStream_t stream)
{
    int    threads = FIREFLY_TRY(attention_detail::thread_count(head_dim));
    dim3   block(threads);
    dim3   grid(query_len, num_heads, batch_size);
    size_t smem_size = threads * sizeof(float);

    causal_attention_scalar_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
        static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
        static_cast<const scalar_t*>(v.data()), static_cast<scalar_t*>(output.data()), query_len, kv_len, num_heads,
        kv_head_num, head_dim, scale, causal);
    return {};
}
}  // namespace

Status attention_detail::launch_contiguous(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                                           const AttentionOptions& options, float scale,
                                           const device::Context& context)
{
    int batch_size = query.shape()[0];
    int sequence_length = query.shape()[1];
    int kv_length = key.shape()[1];
    int num_heads = query.shape()[2];
    int head_dim = query.shape()[3];
    if (query.dtype() == DType::BF16)
    {
        return launch_scalar_prefill<__nv_bfloat16>(query, key, value, output, options.kv_head_count,
                                                    sequence_length, kv_length, batch_size, num_heads, head_dim,
                                                    scale, options.causal, context.stream());
    }
    else
    {
        return launch_scalar_prefill<half>(query, key, value, output, options.kv_head_count, sequence_length,
                                           kv_length, batch_size, num_heads, head_dim, scale, options.causal,
                                           context.stream());
    }
}

}  // namespace firefly::kernels
