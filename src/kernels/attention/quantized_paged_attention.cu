#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

#include "firefly/kernels/attention/detail/cuda_utils.cuh"
#include "firefly/kernels/attention/detail/decode_reduction.cuh"
#include "firefly/kernels/attention/detail/launch.h"
#include "firefly/kernels/attention/detail/workspace.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"

namespace firefly::kernels
{
using attention_detail::paged_decode_split_dual_reduce_kernel;
using attention_detail::paged_decode_split_reduce_kernel;
using attention_detail::warp_reduce_sum;
using detail::CudaScalar;
namespace
{
__device__ inline void cute_int8_mma_tile(const int8_t* q_data, const int8_t* k_data, int32_t* c_data)
{
    using namespace cute;
    using MmaAtom = SM80_16x8x32_S32S8S8S32_TN;
    auto mma = make_tiled_mma(MmaAtom{});
    auto thr_mma = mma.get_slice(threadIdx.x & 31);
    auto gC = make_tensor(make_smem_ptr(c_data),
                          make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);
    Copy_Atom<UniversalCopy<int8_t>, int8_t> s2r_atom_a;
    Copy_Atom<UniversalCopy<int8_t>, int8_t> s2r_atom_b;
    auto                                     s2r_a = make_tiled_copy_A(s2r_atom_a, mma);
    auto                                     s2r_b = make_tiled_copy_B(s2r_atom_b, mma);
    auto                                     thr_copy_a = s2r_a.get_slice(threadIdx.x & 31);
    auto                                     thr_copy_b = s2r_b.get_slice(threadIdx.x & 31);

    for (int k_offset = 0; k_offset < 128; k_offset += 32)
    {
        auto sA = make_tensor(
            make_smem_ptr(const_cast<int8_t*>(k_data + k_offset)),
            make_layout(make_shape(Int<16>{}, Int<32>{}, Int<1>{}), make_stride(Int<128>{}, Int<1>{}, Int<0>{})));
        auto sB = make_tensor(
            make_smem_ptr(const_cast<int8_t*>(q_data + k_offset)),
            make_layout(make_shape(Int<8>{}, Int<32>{}, Int<1>{}), make_stride(Int<128>{}, Int<1>{}, Int<0>{})));
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
        auto tXsA = thr_copy_a.partition_S(sA(_, _, Int<0>{}));
        auto tXsB = thr_copy_b.partition_S(sB(_, _, Int<0>{}));
        auto tXrA = thr_copy_a.retile_D(tCrA);
        auto tXrB = thr_copy_b.retile_D(tCrB);
        copy(s2r_atom_a, tXsA, tXrA);
        copy(s2r_atom_b, tXsB, tXrB);
        gemm(mma, tCrA, tCrB, tCrC);
    }
    copy(tCrC, tCgC);
}

__device__ __forceinline__ void async_copy_16(int8_t* destination, const int8_t* source, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t shared_address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    int      source_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(shared_address), "l"(source),
                 "r"(source_bytes));
#else
    *reinterpret_cast<int4*>(destination) = valid ? *reinterpret_cast<const int4*>(source) : make_int4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void async_copy_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int pending_groups>
__device__ __forceinline__ void async_copy_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(pending_groups));
#endif
}

template <typename scalar_t>
__global__ void quantize_gqa_queries_kernel(const scalar_t* __restrict__ Q, int8_t* __restrict__ quantized_q,
                                            float* __restrict__ quantized_q_scales, int num_heads, int num_kv_heads)
{
    constexpr int    head_dim = 128;
    __shared__ float maxima[2 * head_dim];
    __shared__ float query_scales[2];
    int              kv_head_idx = blockIdx.x;
    int              batch_idx = blockIdx.y;
    int              tid = threadIdx.x;
    int              head0 = kv_head_idx * 2;
    int              head1 = head0 + 1;
    const scalar_t*  q0 = Q + ((int64_t)batch_idx * num_heads + head0) * head_dim;
    const scalar_t*  q1 = Q + ((int64_t)batch_idx * num_heads + head1) * head_dim;

    maxima[tid] = fabsf(CudaScalar<scalar_t>::to_float(q0[tid]));
    maxima[head_dim + tid] = fabsf(CudaScalar<scalar_t>::to_float(q1[tid]));
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
            maxima[head_dim + tid] = fmaxf(maxima[head_dim + tid], maxima[head_dim + tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        query_scales[0] = fmaxf(maxima[0] / 127.0f, 1.0e-8f);
        query_scales[1] = fmaxf(maxima[head_dim] / 127.0f, 1.0e-8f);
        quantized_q_scales[batch_idx * num_heads + head0] = query_scales[0];
        quantized_q_scales[batch_idx * num_heads + head1] = query_scales[1];
    }
    __syncthreads();

    int64_t destination = ((int64_t)batch_idx * num_kv_heads + kv_head_idx) * 2 * head_dim;
    int8_t  q0_value = static_cast<int8_t>(rintf(CudaScalar<scalar_t>::to_float(q0[tid]) / query_scales[0]));
    int8_t  q1_value = static_cast<int8_t>(rintf(CudaScalar<scalar_t>::to_float(q1[tid]) / query_scales[1]));
    quantized_q[destination + tid] = q0_value;
    quantized_q[destination + head_dim + tid] = q1_value;
}

template <typename scalar_t>
__global__ void paged_quantized_decode_kernel(const scalar_t* __restrict__ Q, const int8_t* __restrict__ K_Cache,
                                              const int8_t* __restrict__ V_Cache, const float* __restrict__ scales,
                                              const int* __restrict__ Block_Table, scalar_t* __restrict__ O,
                                              const int* __restrict__ context_lens, int num_heads, int num_kv_heads,
                                              int head_dim, int max_num_blocks, float scale)
{
    extern __shared__ float reduce[];
    __shared__ float        online_softmax[4];
    int                     query_idx = blockIdx.x;
    int                     head_idx = blockIdx.y;
    int                     batch_idx = blockIdx.z;
    int                     tid = threadIdx.x;
    int                     kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int64_t                 q_offset =
        ((int64_t)batch_idx * gridDim.x + query_idx) * num_heads * head_dim + (int64_t)head_idx * head_dim;
    float acc = 0.0f;
    if (tid == 0)
    {
        online_softmax[0] = -1.0e30f;
        online_softmax[1] = 0.0f;
    }
    __syncthreads();
    int visible_tokens = context_lens[batch_idx] + query_idx + 1;
    for (int token = 0; token < visible_tokens; ++token)
    {
        int     logical_block = token / 16;
        int     block_offset = token % 16;
        int     physical_block = Block_Table[batch_idx * max_num_blocks + logical_block];
        int64_t token_addr = (int64_t)physical_block * 16 * num_kv_heads * head_dim +
                             (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim;
        int64_t scale_addr = ((int64_t)physical_block * num_kv_heads + kv_head_idx) * 16 * 2 + block_offset * 2;
        float   partial = tid < head_dim ? CudaScalar<scalar_t>::to_float(Q[q_offset + tid]) *
                                               static_cast<float>(K_Cache[token_addr + tid]) * scales[scale_addr]
                                         : 0.0f;
        partial = warp_reduce_sum(partial);
        int lane = tid & 31;
        int warp = tid >> 5;
        if (lane == 0) reduce[warp] = partial;
        __syncthreads();
        if (warp == 0)
        {
            int   warp_count = (blockDim.x + 31) >> 5;
            float dot = lane < warp_count ? reduce[lane] : 0.0f;
            dot = warp_reduce_sum(dot);
            if (lane == 0)
            {
                float score = dot * scale;
                float next_m = fmaxf(online_softmax[0], score);
                float alpha = expf(score - next_m);
                float beta = expf(online_softmax[0] - next_m);
                online_softmax[0] = next_m;
                online_softmax[1] = online_softmax[1] * beta + alpha;
                online_softmax[2] = alpha;
                online_softmax[3] = beta;
            }
        }
        __syncthreads();
        if (tid < head_dim)
        {
            acc = acc * online_softmax[3] +
                  static_cast<float>(V_Cache[token_addr + tid]) * scales[scale_addr + 1] * online_softmax[2];
        }
    }
    if (tid < head_dim) O[q_offset + tid] = CudaScalar<scalar_t>::from_float(acc / (online_softmax[1] + 1.0e-6f));
}

template <typename scalar_t, int SPLIT_SIZE>
__global__ void paged_decode_split_quantized_partial_kernel(
    const scalar_t* __restrict__ Q, const int8_t* __restrict__ K_Cache, const int8_t* __restrict__ V_Cache,
    const float* __restrict__ scales, const int* __restrict__ Block_Table, const int* context_lens,
    float* __restrict__ partial_m, float* __restrict__ partial_l, float* __restrict__ partial_acc, int num_heads,
    int num_kv_heads, int head_dim, int max_num_blocks, int num_splits, float scale)
{
    extern __shared__ float reduce[];
    int                     split_idx = blockIdx.x;
    int                     head_idx = blockIdx.y;
    int                     batch_idx = blockIdx.z;
    int                     tid = threadIdx.x;
    int                     kv_group = num_heads / num_kv_heads;
    int                     kv_head_idx = head_idx / kv_group;
    int                     visible_tokens = context_lens[batch_idx] + 1;
    int                     key_start = split_idx * SPLIT_SIZE;
    int                     key_end = min(key_start + SPLIT_SIZE, visible_tokens);
    int                     partial_idx = (batch_idx * num_heads + head_idx) * num_splits + split_idx;
    const scalar_t*         q_ptr = Q + (int64_t)batch_idx * num_heads * head_dim + (int64_t)head_idx * head_dim;
    float                   acc = 0.0f;
    float                   m_i = -1.0e30f;
    float                   l_i = 0.0f;

    for (int key_idx = key_start; key_idx < key_end; ++key_idx)
    {
        int     physical_block = Block_Table[batch_idx * max_num_blocks + (key_idx >> 4)];
        int     block_offset = key_idx & 15;
        int64_t token_addr = (int64_t)physical_block * 16 * num_kv_heads * head_dim +
                             (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim;
        int64_t scale_addr = ((int64_t)physical_block * num_kv_heads + kv_head_idx) * 16 * 2 + block_offset * 2;
        float   dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x)
        {
            dot += CudaScalar<scalar_t>::to_float(q_ptr[d]) * static_cast<float>(K_Cache[token_addr + d]);
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        float score = reduce[0] * scales[scale_addr] * scale;
        float next_m = fmaxf(m_i, score);
        float alpha = __expf(score - next_m);
        float beta = m_i <= -5.0e29f ? 0.0f : __expf(m_i - next_m);
        if (tid < head_dim)
        {
            acc = acc * beta + static_cast<float>(V_Cache[token_addr + tid]) * scales[scale_addr + 1] * alpha;
        }
        l_i = l_i * beta + alpha;
        m_i = next_m;
        __syncthreads();
    }
    if (tid == 0)
    {
        partial_m[partial_idx] = m_i;
        partial_l[partial_idx] = l_i;
    }
    if (tid < head_dim) partial_acc[(int64_t)partial_idx * head_dim + tid] = acc;
}

template <typename scalar_t, int SPLIT_SIZE>
__global__ void paged_decode_split_quantized_dual_tiled_kernel(
    const int8_t* __restrict__ quantized_q, const float* __restrict__ quantized_q_scales,
    const int8_t* __restrict__ K_Cache, const int8_t* __restrict__ V_Cache, const float* __restrict__ scales,
    const int* __restrict__ Block_Table, const int* __restrict__ context_lens, float* __restrict__ partial_m,
    float* __restrict__ partial_l, float* __restrict__ partial_acc, int num_heads, int num_kv_heads, int max_num_blocks,
    int num_splits, float attention_scale)
{
    constexpr int                 head_dim = 128;
    constexpr int                 tile_tokens = 16;
    __shared__ alignas(16) int8_t k_tile[2][tile_tokens * head_dim];
    __shared__ alignas(16) int8_t v_tile[tile_tokens * head_dim];
    __shared__ alignas(16) int8_t q_tile[8 * head_dim];
    __shared__ float              tile_scales[tile_tokens * 2];
    __shared__ float              scores[tile_tokens * 2];
    __shared__ int32_t            qk_c[tile_tokens * 8];

    int           split_idx = blockIdx.x;
    int           kv_head_idx = blockIdx.y;
    int           batch_idx = blockIdx.z;
    int           tid = threadIdx.x;
    int           head0 = kv_head_idx * 2;
    int           head1 = head0 + 1;
    const int8_t* quantized_q_tile = quantized_q + ((int64_t)batch_idx * num_kv_heads + kv_head_idx) * 2 * head_dim;
    for (int vector_idx = tid; vector_idx < 2 * head_dim / 16; vector_idx += blockDim.x)
    {
        int  row = vector_idx / (head_dim / 16);
        int  column = vector_idx % (head_dim / 16);
        int4 value = reinterpret_cast<const int4*>(quantized_q_tile + row * head_dim)[column];
        for (int destination_row = row; destination_row < 8; destination_row += 2)
        {
            reinterpret_cast<int4*>(q_tile + destination_row * head_dim)[column] = value;
        }
    }
    __syncthreads();
    float q0_scale = quantized_q_scales[batch_idx * num_heads + head0];
    float q1_scale = quantized_q_scales[batch_idx * num_heads + head1];

    int   visible_tokens = context_lens[batch_idx] + 1;
    int   key_start = split_idx * SPLIT_SIZE;
    int   key_end = min(key_start + SPLIT_SIZE, visible_tokens);
    int   partial0 = (batch_idx * num_heads + head0) * num_splits + split_idx;
    int   partial1 = (batch_idx * num_heads + head1) * num_splits + split_idx;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float m0 = -1.0e30f;
    float m1 = -1.0e30f;
    float l0 = 0.0f;
    float l1 = 0.0f;

    constexpr int vectors_per_token = head_dim / 16;
    constexpr int vectors_per_tile = tile_tokens * vectors_per_token;
    int           k_stage = 0;
    int           first_physical_block = Block_Table[batch_idx * max_num_blocks + (key_start >> 4)];
    for (int vector = tid; vector < vectors_per_tile; vector += blockDim.x)
    {
        int     token = vector / vectors_per_token;
        int     vector_idx = vector % vectors_per_token;
        int     block_offset = (key_start + token) & 15;
        int64_t token_address = (int64_t)first_physical_block * 16 * num_kv_heads * head_dim +
                                (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim +
                                vector_idx * 16;
        async_copy_16(k_tile[k_stage] + token * head_dim + vector_idx * 16, K_Cache + token_address,
                      key_start + token < key_end);
    }
    async_copy_commit();
    async_copy_wait<0>();
    __syncthreads();

    for (int tile_start = key_start; tile_start < key_end; tile_start += tile_tokens)
    {
        int  tile_count = min(tile_tokens, key_end - tile_start);
        int  physical_block = Block_Table[batch_idx * max_num_blocks + (tile_start >> 4)];
        int  next_tile_start = tile_start + tile_tokens;
        bool has_next_tile = next_tile_start < key_end;
        int  next_physical_block = has_next_tile ? Block_Table[batch_idx * max_num_blocks + (next_tile_start >> 4)] : 0;
        int  next_tile_count = has_next_tile ? min(tile_tokens, key_end - next_tile_start) : 0;

        for (int vector = tid; vector < vectors_per_tile; vector += blockDim.x)
        {
            int     token = vector / vectors_per_token;
            int     vector_idx = vector % vectors_per_token;
            int     key_idx = tile_start + token;
            int8_t* v_destination = v_tile + token * head_dim + vector_idx * 16;
            int     block_offset = key_idx & 15;
            int64_t token_address = (int64_t)physical_block * 16 * num_kv_heads * head_dim +
                                    (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim +
                                    vector_idx * 16;
            async_copy_16(v_destination, V_Cache + token_address, token < tile_count);
        }
        async_copy_commit();
        if (has_next_tile)
        {
            for (int vector = tid; vector < vectors_per_tile; vector += blockDim.x)
            {
                int     token = vector / vectors_per_token;
                int     vector_idx = vector % vectors_per_token;
                int     next_block_offset = (next_tile_start + token) & 15;
                int64_t next_token_address = (int64_t)next_physical_block * 16 * num_kv_heads * head_dim +
                                             (int64_t)next_block_offset * num_kv_heads * head_dim +
                                             (int64_t)kv_head_idx * head_dim + vector_idx * 16;
                async_copy_16(k_tile[k_stage ^ 1] + token * head_dim + vector_idx * 16, K_Cache + next_token_address,
                              token < next_tile_count);
            }
            async_copy_commit();
        }
        for (int token = tid; token < tile_count; token += blockDim.x)
        {
            int     key_idx = tile_start + token;
            int     block_offset = key_idx & 15;
            int64_t scale_addr = ((int64_t)physical_block * num_kv_heads + kv_head_idx) * 16 * 2 + block_offset * 2;
            tile_scales[token * 2] = scales[scale_addr];
            tile_scales[token * 2 + 1] = scales[scale_addr + 1];
        }
        __syncthreads();

        int warp = tid >> 5;
        int lane = tid & 31;
        if (tile_count == tile_tokens)
        {
            if (warp == 0) cute_int8_mma_tile(q_tile, k_tile[k_stage], qk_c);
            __syncthreads();
            if (tid < tile_tokens)
            {
                scores[tid] = static_cast<float>(qk_c[tid * 8]) * q0_scale * tile_scales[tid * 2] * attention_scale;
                scores[tile_tokens + tid] =
                    static_cast<float>(qk_c[tid * 8 + 1]) * q1_scale * tile_scales[tid * 2] * attention_scale;
            }
        }
        else
            for (int local_token = 0; local_token < 8; ++local_token)
            {
                int token = warp * 8 + local_token;
                if (token < tile_count)
                {
                    char4 k_vector = reinterpret_cast<const char4*>(k_tile[k_stage] + token * head_dim)[lane];
                    char4 q0_vector = reinterpret_cast<const char4*>(q_tile)[lane];
                    char4 q1_vector = reinterpret_cast<const char4*>(q_tile + head_dim)[lane];
                    int   dot0 = __dp4a(k_vector, q0_vector, 0);
                    int   dot1 = __dp4a(k_vector, q1_vector, 0);
                    for (int offset = 16; offset > 0; offset >>= 1)
                    {
                        dot0 += __shfl_down_sync(0xffffffffu, dot0, offset);
                        dot1 += __shfl_down_sync(0xffffffffu, dot1, offset);
                    }
                    if (lane == 0)
                    {
                        scores[token] = static_cast<float>(dot0) * q0_scale * tile_scales[token * 2] * attention_scale;
                        scores[tile_tokens + token] =
                            static_cast<float>(dot1) * q1_scale * tile_scales[token * 2] * attention_scale;
                    }
                }
            }
        __syncthreads();

        if (has_next_tile)
            async_copy_wait<1>();
        else
            async_copy_wait<0>();
        __syncthreads();

        for (int token = 0; token < tile_count; ++token)
        {
            float score0 = scores[token];
            float score1 = scores[tile_tokens + token];
            float next_m0;
            float alpha0;
            float beta0;
            if (score0 > m0)
            {
                next_m0 = score0;
                alpha0 = 1.0f;
                beta0 = m0 <= -5.0e29f ? 0.0f : __expf(m0 - score0);
            }
            else
            {
                next_m0 = m0;
                alpha0 = __expf(score0 - m0);
                beta0 = 1.0f;
            }
            float next_m1;
            float alpha1;
            float beta1;
            if (score1 > m1)
            {
                next_m1 = score1;
                alpha1 = 1.0f;
                beta1 = m1 <= -5.0e29f ? 0.0f : __expf(m1 - score1);
            }
            else
            {
                next_m1 = m1;
                alpha1 = __expf(score1 - m1);
                beta1 = 1.0f;
            }
            float value0 = static_cast<float>(v_tile[token * head_dim + tid]) * tile_scales[token * 2 + 1];
            float value1 = static_cast<float>(v_tile[token * head_dim + tid + blockDim.x]) * tile_scales[token * 2 + 1];
            acc00 = acc00 * beta0 + value0 * alpha0;
            acc01 = acc01 * beta0 + value1 * alpha0;
            acc10 = acc10 * beta1 + value0 * alpha1;
            acc11 = acc11 * beta1 + value1 * alpha1;
            l0 = l0 * beta0 + alpha0;
            l1 = l1 * beta1 + alpha1;
            m0 = next_m0;
            m1 = next_m1;
        }
        __syncthreads();
        if (has_next_tile)
        {
            async_copy_wait<0>();
            __syncthreads();
            k_stage ^= 1;
        }
    }

    if (tid == 0)
    {
        partial_m[partial0] = m0;
        partial_l[partial0] = l0;
        partial_m[partial1] = m1;
        partial_l[partial1] = l1;
    }
    partial_acc[(int64_t)partial0 * head_dim + tid] = acc00;
    partial_acc[(int64_t)partial0 * head_dim + tid + blockDim.x] = acc01;
    partial_acc[(int64_t)partial1 * head_dim + tid] = acc10;
    partial_acc[(int64_t)partial1 * head_dim + tid + blockDim.x] = acc11;
}

template <typename scalar_t>
void launch_quantized_paged_decode_split(Tensor& q, Tensor& k, Tensor& v, Tensor& output, const Tensor& scales,
                                         const int* block_table, int kv_head_num, int max_context_blocks,
                                         const int* context_lens, int batch_size, int num_heads, int head_dim,
                                         float scale, int max_decode_context_len, cudaStream_t stream)
{
    int split_token_limit = max_decode_context_len > 0 ? max_decode_context_len + 1 : max_context_blocks * 16;
    split_token_limit = std::min(split_token_limit, max_context_blocks * 16);
    int num_splits = std::max(1, (split_token_limit + 255) / 256);
    attention_detail::ensure_decode_workspace(batch_size, num_heads, num_splits, head_dim);
    auto& scratch = attention_detail::decode_workspace();
    int   threads = attention_detail::thread_count(head_dim);
    dim3  grid(num_splits, num_heads, batch_size);
    paged_decode_split_quantized_partial_kernel<scalar_t, 256><<<grid, threads, threads * sizeof(float), stream>>>(
        static_cast<const scalar_t*>(q.data()), static_cast<const int8_t*>(k.data()),
        static_cast<const int8_t*>(v.data()), static_cast<const float*>(scales.data()),
        static_cast<const int*>(block_table), context_lens, static_cast<float*>(scratch.partial_m.data()),
        static_cast<float*>(scratch.partial_l.data()), static_cast<float*>(scratch.partial_acc.data()), num_heads,
        kv_head_num, head_dim, max_context_blocks, num_splits, scale);
    dim3 reduce_grid(num_heads, batch_size);
    paged_decode_split_reduce_kernel<scalar_t><<<reduce_grid, threads, 0, stream>>>(
        static_cast<const float*>(scratch.partial_m.data()), static_cast<const float*>(scratch.partial_l.data()),
        static_cast<const float*>(scratch.partial_acc.data()), static_cast<scalar_t*>(output.data()), num_heads,
        head_dim, num_splits);
}

template <typename scalar_t>
void launch_quantized_paged_decode_dual(Tensor& q, Tensor& k, Tensor& v, Tensor& output, const Tensor& scales,
                                        const int* block_table, int kv_head_num, int max_context_blocks,
                                        const int* context_lens, int batch_size, int num_heads, int head_dim,
                                        float scale, int max_decode_context_len, QuantizedQuery quantized_query,
                                        int split_size, cudaStream_t stream)
{
    int split_token_limit = max_decode_context_len > 0 ? max_decode_context_len + 1 : max_context_blocks * 16;
    split_token_limit = std::min(split_token_limit, max_context_blocks * 16);
    auto launch = [&](auto split_tag)
    {
        constexpr int split_size = decltype(split_tag)::value;
        int           num_splits = std::max(1, (split_token_limit + split_size - 1) / split_size);
        attention_detail::ensure_decode_workspace(batch_size, num_heads, num_splits, head_dim);
        auto&         scratch = attention_detail::decode_workspace();
        const Tensor* quantized_values = quantized_query.values;
        const Tensor* quantized_scales = quantized_query.scales;
        if (!quantized_query.valid())
        {
            quantize_gqa_queries_kernel<scalar_t><<<dim3(kv_head_num, batch_size), 128, 0, stream>>>(
                static_cast<const scalar_t*>(q.data()), static_cast<int8_t*>(scratch.quantized_query.data()),
                static_cast<float*>(scratch.quantized_query_scales.data()), num_heads, kv_head_num);
            quantized_values = &scratch.quantized_query;
            quantized_scales = &scratch.quantized_query_scales;
        }
        dim3 partial_grid(num_splits, kv_head_num, batch_size);
        paged_decode_split_quantized_dual_tiled_kernel<scalar_t, split_size><<<partial_grid, 64, 0, stream>>>(
            static_cast<const int8_t*>(quantized_values->data()), static_cast<const float*>(quantized_scales->data()),
            static_cast<const int8_t*>(k.data()), static_cast<const int8_t*>(v.data()),
            static_cast<const float*>(scales.data()), static_cast<const int*>(block_table), context_lens,
            static_cast<float*>(scratch.partial_m.data()), static_cast<float*>(scratch.partial_l.data()),
            static_cast<float*>(scratch.partial_acc.data()), num_heads, kv_head_num, max_context_blocks, num_splits,
            scale);
        dim3 reduce_grid(kv_head_num, batch_size);
        paged_decode_split_dual_reduce_kernel<scalar_t><<<reduce_grid, 128, 0, stream>>>(
            static_cast<const float*>(scratch.partial_m.data()), static_cast<const float*>(scratch.partial_l.data()),
            static_cast<const float*>(scratch.partial_acc.data()), static_cast<scalar_t*>(output.data()), num_heads,
            head_dim, num_splits);
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

}  // namespace

void attention_detail::launch_quantized_paged(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                                              const AttentionOptions& options, const DecodeConfig& decode_config,
                                              float scale, const device::Context& context)
{
    int           batch_size = query.shape()[0];
    int           sequence_length = query.shape()[1];
    int           num_heads = query.shape()[2];
    int           head_dim = query.shape()[3];
    int           threads = attention_detail::thread_count(head_dim);
    const Tensor& scales = *options.kv_scales;

    if (sequence_length == 1)
    {
        auto launch = [&]<typename Scalar>()
        {
            if (num_heads == options.kv_head_count * 2 && head_dim == 128)
            {
                launch_quantized_paged_decode_dual<Scalar>(
                    query, key, value, output, scales, options.block_table, options.kv_head_count,
                    options.max_context_blocks, options.context_lengths, batch_size, num_heads, head_dim, scale,
                    options.max_decode_context_length, options.quantized_query, decode_config.split_size,
                    context.stream());
            }
            else
            {
                launch_quantized_paged_decode_split<Scalar>(query, key, value, output, scales, options.block_table,
                                                            options.kv_head_count, options.max_context_blocks,
                                                            options.context_lengths, batch_size, num_heads, head_dim,
                                                            scale, options.max_decode_context_length, context.stream());
            }
        };
        if (query.dtype() == DType::BF16)
            launch.template operator()<__nv_bfloat16>();
        else
            launch.template operator()<half>();
        return;
    }

    dim3   grid(sequence_length, num_heads, batch_size);
    size_t shared_memory = threads * sizeof(float);
    if (query.dtype() == DType::BF16)
    {
        paged_quantized_decode_kernel<__nv_bfloat16><<<grid, threads, shared_memory, context.stream()>>>(
            static_cast<const __nv_bfloat16*>(query.data()), static_cast<const int8_t*>(key.data()),
            static_cast<const int8_t*>(value.data()), static_cast<const float*>(scales.data()), options.block_table,
            static_cast<__nv_bfloat16*>(output.data()), options.context_lengths, num_heads, options.kv_head_count,
            head_dim, options.max_context_blocks, scale);
    }
    else
    {
        paged_quantized_decode_kernel<half><<<grid, threads, shared_memory, context.stream()>>>(
            static_cast<const half*>(query.data()), static_cast<const int8_t*>(key.data()),
            static_cast<const int8_t*>(value.data()), static_cast<const float*>(scales.data()), options.block_table,
            static_cast<half*>(output.data()), options.context_lengths, num_heads, options.kv_head_count, head_dim,
            options.max_context_blocks, scale);
    }
}

}  // namespace firefly::kernels
