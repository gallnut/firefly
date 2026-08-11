#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "firefly/kernels/attention/detail/cuda_utils.cuh"
#include "firefly/kernels/attention/detail/decode_reduction.cuh"
#include "firefly/kernels/attention/detail/launch.h"
#include "firefly/kernels/attention/detail/workspace.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"

namespace firefly::kernels
{
using attention_detail::paged_decode_split_dual_reduce_kernel;
using attention_detail::paged_decode_split_reduce_kernel;
using attention_detail::warp_reduce_sum;
using detail::CudaScalar;
namespace
{
template <typename scalar_t, int SPLIT_SIZE>
__global__ void paged_decode_split_partial_kernel(const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K_Cache,
                                                  const scalar_t* __restrict__ V_Cache,
                                                  const int* __restrict__ Block_Table, const int* context_lens,
                                                  float* __restrict__ partial_m, float* __restrict__ partial_l,
                                                  float* __restrict__ partial_acc, int num_heads, int num_kv_heads,
                                                  int head_dim, int max_num_blocks, int num_splits, float scale)
{
    extern __shared__ float reduce[];

    int split_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    int kv_group = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group;
    int context_len = context_lens[batch_idx];
    int visible_tokens = context_len + 1;
    int key_start = split_idx * SPLIT_SIZE;
    int key_end = min(key_start + SPLIT_SIZE, visible_tokens);

    int   partial_idx = (batch_idx * num_heads + head_idx) * num_splits + split_idx;
    float acc = 0.0f;
    float m_i = -1.0e30f;
    float l_i = 0.0f;

    const scalar_t* q_ptr = Q + (int64_t)batch_idx * num_heads * head_dim + (int64_t)head_idx * head_dim;

    for (int key_idx = key_start; key_idx < key_end; ++key_idx)
    {
        int     block_idx = key_idx >> 4;
        int     block_offset = key_idx & 15;
        int     phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
        int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                             (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim;

        float partial_dot = 0.0f;
        if (tid < head_dim)
        {
            partial_dot =
                CudaScalar<scalar_t>::to_float(q_ptr[tid]) * CudaScalar<scalar_t>::to_float(K_Cache[token_addr + tid]);
        }

        reduce[tid] = partial_dot;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }

        float score = reduce[0] * scale;
        float m_next = fmaxf(m_i, score);
        float alpha = __expf(score - m_next);
        float beta = (m_i <= -5.0e29f) ? 0.0f : __expf(m_i - m_next);

        if (tid < head_dim)
        {
            acc = acc * beta + CudaScalar<scalar_t>::to_float(V_Cache[token_addr + tid]) * alpha;
        }

        l_i = l_i * beta + alpha;
        m_i = m_next;
        __syncthreads();
    }

    if (tid == 0)
    {
        partial_m[partial_idx] = m_i;
        partial_l[partial_idx] = l_i;
    }
    if (tid < head_dim)
    {
        partial_acc[(int64_t)partial_idx * head_dim + tid] = acc;
    }
}

__device__ __forceinline__ void block_reduce_sum2(float& value0, float& value1, float* shared0, float* shared1)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warp_count = (blockDim.x + 31) >> 5;

    value0 = warp_reduce_sum(value0);
    value1 = warp_reduce_sum(value1);

    if (lane == 0)
    {
        shared0[warp] = value0;
        shared1[warp] = value1;
    }
    __syncthreads();

    value0 = threadIdx.x < warp_count ? shared0[lane] : 0.0f;
    value1 = threadIdx.x < warp_count ? shared1[lane] : 0.0f;
    if (warp == 0)
    {
        value0 = warp_reduce_sum(value0);
        value1 = warp_reduce_sum(value1);
        if (lane == 0)
        {
            shared0[0] = value0;
            shared1[0] = value1;
        }
    }
    __syncthreads();

    value0 = shared0[0];
    value1 = shared1[0];
}

template <typename scalar_t, int SPLIT_SIZE, int HEAD_DIM>
__global__ void paged_decode_split_dual_partial_kernel(const scalar_t* __restrict__ Q,
                                                       const scalar_t* __restrict__ K_Cache,
                                                       const scalar_t* __restrict__ V_Cache,
                                                       const int* __restrict__ Block_Table, const int* context_lens,
                                                       float* __restrict__ partial_m, float* __restrict__ partial_l,
                                                       float* __restrict__ partial_acc, int num_heads, int num_kv_heads,
                                                       int max_num_blocks, int num_splits, float scale)
{
    extern __shared__ float reduce[];
    float*                  reduce0 = reduce;
    float*                  reduce1 = reduce + blockDim.x;

    int split_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    int head0 = kv_head_idx * 2;
    int head1 = head0 + 1;
    if (head1 >= num_heads) return;

    int context_len = context_lens[batch_idx];
    int visible_tokens = context_len + 1;
    int key_start = split_idx * SPLIT_SIZE;
    int key_end = min(key_start + SPLIT_SIZE, visible_tokens);

    int partial0 = (batch_idx * num_heads + head0) * num_splits + split_idx;
    int partial1 = (batch_idx * num_heads + head1) * num_splits + split_idx;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m0 = -1.0e30f;
    float m1 = -1.0e30f;
    float l0 = 0.0f;
    float l1 = 0.0f;

    const scalar_t* q0 = Q + (int64_t)batch_idx * num_heads * HEAD_DIM + (int64_t)head0 * HEAD_DIM;
    const scalar_t* q1 = Q + (int64_t)batch_idx * num_heads * HEAD_DIM + (int64_t)head1 * HEAD_DIM;
    float           q0_val = CudaScalar<scalar_t>::to_float(q0[tid]);
    float           q1_val = CudaScalar<scalar_t>::to_float(q1[tid]);

    for (int key_idx = key_start; key_idx < key_end; ++key_idx)
    {
        int     block_idx = key_idx >> 4;
        int     block_offset = key_idx & 15;
        int     phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
        int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * HEAD_DIM +
                             (int64_t)block_offset * num_kv_heads * HEAD_DIM + (int64_t)kv_head_idx * HEAD_DIM;

        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float k_val = CudaScalar<scalar_t>::to_float(K_Cache[token_addr + tid]);
        dot0 = q0_val * k_val;
        dot1 = q1_val * k_val;

        block_reduce_sum2(dot0, dot1, reduce0, reduce1);

        float score0 = dot0 * scale;
        float score1 = dot1 * scale;

        float next_m0 = fmaxf(m0, score0);
        float next_m1 = fmaxf(m1, score1);
        float alpha0 = __expf(score0 - next_m0);
        float alpha1 = __expf(score1 - next_m1);
        float beta0 = (m0 <= -5.0e29f) ? 0.0f : __expf(m0 - next_m0);
        float beta1 = (m1 <= -5.0e29f) ? 0.0f : __expf(m1 - next_m1);

        float v_val = CudaScalar<scalar_t>::to_float(V_Cache[token_addr + tid]);
        acc0 = acc0 * beta0 + v_val * alpha0;
        acc1 = acc1 * beta1 + v_val * alpha1;

        l0 = l0 * beta0 + alpha0;
        l1 = l1 * beta1 + alpha1;
        m0 = next_m0;
        m1 = next_m1;
        __syncthreads();
    }

    if (tid == 0)
    {
        partial_m[partial0] = m0;
        partial_l[partial0] = l0;
        partial_m[partial1] = m1;
        partial_l[partial1] = l1;
    }
    partial_acc[(int64_t)partial0 * HEAD_DIM + tid] = acc0;
    partial_acc[(int64_t)partial1 * HEAD_DIM + tid] = acc1;
}

template <typename scalar_t>
void launch_paged_decode_split(Tensor& q, Tensor& k, Tensor& v, Tensor& output, const int* kv_cache_block_table,
                               int kv_head_num, int max_context_blocks, const int* context_lens, int batch_size,
                               int num_heads, int head_dim, float scale, int max_decode_context_len, int split_size,
                               cudaStream_t stream)
{
    auto launch = [&](auto split_tag)
    {
        constexpr int split_size = decltype(split_tag)::value;
        int split_token_limit = max_decode_context_len > 0 ? max_decode_context_len + 1 : max_context_blocks * 16;
        split_token_limit = std::min(split_token_limit, max_context_blocks * 16);
        int num_splits = std::max(1, (split_token_limit + split_size - 1) / split_size);
        attention_detail::ensure_decode_workspace(batch_size, num_heads, num_splits, head_dim);
        auto& scratch = attention_detail::decode_workspace();

        int  threads = attention_detail::thread_count(head_dim);
        bool use_dual_gqa_decode = num_heads == kv_head_num * 2 && head_dim == 128;
        if (use_dual_gqa_decode)
        {
            dim3 partial_grid(num_splits, kv_head_num, batch_size);
            paged_decode_split_dual_partial_kernel<scalar_t, split_size, 128>
                <<<partial_grid, threads, 2 * threads * sizeof(float), stream>>>(
                    static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
                    static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table), context_lens,
                    static_cast<float*>(scratch.partial_m.data()), static_cast<float*>(scratch.partial_l.data()),
                    static_cast<float*>(scratch.partial_acc.data()), num_heads, kv_head_num, max_context_blocks,
                    num_splits, scale);
        }
        else
        {
            dim3 partial_grid(num_splits, num_heads, batch_size);
            paged_decode_split_partial_kernel<scalar_t, split_size>
                <<<partial_grid, threads, threads * sizeof(float), stream>>>(
                    static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
                    static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table), context_lens,
                    static_cast<float*>(scratch.partial_m.data()), static_cast<float*>(scratch.partial_l.data()),
                    static_cast<float*>(scratch.partial_acc.data()), num_heads, kv_head_num, head_dim,
                    max_context_blocks, num_splits, scale);
        }

        if (use_dual_gqa_decode)
        {
            dim3 reduce_grid(kv_head_num, batch_size);
            paged_decode_split_dual_reduce_kernel<scalar_t><<<reduce_grid, threads, 0, stream>>>(
                static_cast<const float*>(scratch.partial_m.data()),
                static_cast<const float*>(scratch.partial_l.data()),
                static_cast<const float*>(scratch.partial_acc.data()), static_cast<scalar_t*>(output.data()), num_heads,
                head_dim, num_splits);
        }
        else
        {
            dim3 reduce_grid(num_heads, batch_size);
            paged_decode_split_reduce_kernel<scalar_t><<<reduce_grid, threads, 0, stream>>>(
                static_cast<const float*>(scratch.partial_m.data()),
                static_cast<const float*>(scratch.partial_l.data()),
                static_cast<const float*>(scratch.partial_acc.data()), static_cast<scalar_t*>(output.data()), num_heads,
                head_dim, num_splits);
        }
    };

    switch (split_size)
    {
        case 128:
            launch(std::integral_constant<int, 128>{});
            break;
        case 512:
            launch(std::integral_constant<int, 512>{});
            break;
        default:
            launch(std::integral_constant<int, 256>{});
            break;
    }
}

template <typename Scalar>
void launch_paged_decode_typed(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                               const AttentionOptions& options, const attention_detail::DecodeConfig& decode_config,
                               float scale, cudaStream_t stream)
{
    launch_paged_decode_split<Scalar>(query, key, value, output, options.block_table, options.kv_head_count,
                                      options.max_context_blocks, options.context_lengths, query.shape()[0],
                                      query.shape()[2], query.shape()[3], scale, options.max_decode_context_length,
                                      decode_config.split_size, stream);
}
}  // namespace

void attention_detail::launch_paged_decode(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                                           const AttentionOptions& options, const DecodeConfig& decode_config,
                                           float scale, const device::Context& context)
{
    if (query.dtype() == DType::BF16)
    {
        launch_paged_decode_typed<__nv_bfloat16>(query, key, value, output, options, decode_config, scale,
                                                 context.stream());
    }
    else
    {
        launch_paged_decode_typed<half>(query, key, value, output, options, decode_config, scale, context.stream());
    }
}
}  // namespace firefly::kernels
