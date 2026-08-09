#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

#include "firefly/kernels/attention/detail/cuda_utils.cuh"
#include "firefly/kernels/attention/detail/launch.h"
#include "firefly/kernels/detail/cuda_scalar.cuh"

using namespace nvcuda;

namespace firefly::kernels
{
using detail::CudaScalar;
namespace
{
template <typename scalar_t>
__global__ void paged_attention_kernel(const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K_Cache,
                                       const scalar_t* __restrict__ V_Cache, const int* __restrict__ Block_Table,
                                       scalar_t* __restrict__ O, const int             batch_size,
                                       const int* __restrict__ context_lens, const int num_heads,
                                       const int num_kv_heads, const int seq_len, const int head_dim,
                                       const int max_num_blocks, const float scale)
{
    extern __shared__ float reduce[];

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    if (head_idx >= num_heads) return;

    int     kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int64_t q_offset = (int64_t)batch_idx * seq_len * num_heads * head_dim + (int64_t)query_idx * num_heads * head_dim +
                       (int64_t)head_idx * head_dim;

    const scalar_t* q_ptr = Q + q_offset;
    scalar_t*       o_ptr = O + q_offset;

    // K/V for the current chunk have already been appended, so every query can
    // see all prefix tokens plus current chunk tokens up to its causal position.
    int visible_tokens = context_lens[batch_idx] + query_idx + 1;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc = 0.0f;

    for (int key_idx = 0; key_idx < visible_tokens; ++key_idx)
    {
        int block_idx = key_idx / 16;
        int block_offset = key_idx % 16;
        int phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];

        int64_t base_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim;
        int64_t token_addr =
            base_addr + (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim;

        const scalar_t* k_ptr = K_Cache + token_addr;

        float partial_dot = 0.0f;
        if (tid < head_dim)
        {
            partial_dot = CudaScalar<scalar_t>::to_float(q_ptr[tid]) * CudaScalar<scalar_t>::to_float(k_ptr[tid]);
        }

        reduce[tid] = partial_dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }

        float score = reduce[0] * scale;
        float m_next = fmaxf(m_i, score);
        float alpha = expf(score - m_next);
        float beta = expf(m_i - m_next);

        if (tid < head_dim)
        {
            const scalar_t* v_ptr = V_Cache + token_addr;
            acc = acc * beta + CudaScalar<scalar_t>::to_float(v_ptr[tid]) * alpha;
        }

        l_i = l_i * beta + alpha;
        m_i = m_next;
    }

    if (tid < head_dim)
    {
        o_ptr[tid] = CudaScalar<scalar_t>::from_float(acc / (l_i + 1e-6f));
    }
}

template <typename scalar_t, int Q_BLOCK, int K_BLOCK, int MAX_HEAD_DIM>
__global__ void paged_attention_blockwise_kernel(const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K_Cache,
                                                 const scalar_t* __restrict__ V_Cache,
                                                 const int* __restrict__ Block_Table, scalar_t* __restrict__ O,
                                                 const int* __restrict__ context_lens, const int num_heads,
                                                 const int num_kv_heads, const int seq_len, const int head_dim,
                                                 const int max_num_blocks, const float scale)
{
    __shared__ scalar_t q_tile[Q_BLOCK][MAX_HEAD_DIM];
    __shared__ scalar_t k_tile[K_BLOCK][MAX_HEAD_DIM];
    __shared__ scalar_t v_tile[K_BLOCK][MAX_HEAD_DIM];
    __shared__ float    scores[Q_BLOCK][K_BLOCK];
    __shared__ float    acc[Q_BLOCK][MAX_HEAD_DIM];
    __shared__ float    m_val[Q_BLOCK];
    __shared__ float    l_val[Q_BLOCK];
    __shared__ float    m_next[Q_BLOCK];
    __shared__ float    l_next[Q_BLOCK];

    constexpr float kNeg = -1.0e30f;

    int q_block_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    int q_start = q_block_idx * Q_BLOCK;
    int q_count = seq_len - q_start;
    q_count = q_count > Q_BLOCK ? Q_BLOCK : q_count;
    if (q_count <= 0 || head_idx >= num_heads) return;

    int kv_group = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group;
    int context_len = context_lens[batch_idx];
    int total_tokens = context_len + seq_len;
    int max_visible = context_len + q_start + q_count;

    int64_t q_batch_base = (int64_t)batch_idx * seq_len * num_heads * head_dim;
    int64_t q_head_base = (int64_t)head_idx * head_dim;

    for (int idx = tid; idx < Q_BLOCK * head_dim; idx += blockDim.x)
    {
        int qi = idx / head_dim;
        int d = idx - qi * head_dim;
        int q_idx = q_start + qi;
        if (qi < q_count)
        {
            q_tile[qi][d] = Q[q_batch_base + (int64_t)q_idx * num_heads * head_dim + q_head_base + d];
        }
        acc[qi][d] = 0.0f;
    }
    for (int qi = tid; qi < Q_BLOCK; qi += blockDim.x)
    {
        m_val[qi] = kNeg;
        l_val[qi] = 0.0f;
    }
    __syncthreads();

    for (int tile_start = 0; tile_start < max_visible; tile_start += K_BLOCK)
    {
        for (int idx = tid; idx < K_BLOCK * head_dim; idx += blockDim.x)
        {
            int kj = idx / head_dim;
            int d = idx - kj * head_dim;
            int key_idx = tile_start + kj;

            scalar_t k_zero = CudaScalar<scalar_t>::from_float(0.0f);
            scalar_t v_zero = CudaScalar<scalar_t>::from_float(0.0f);
            if (key_idx < total_tokens)
            {
                int     block_idx = key_idx >> 4;
                int     block_offset = key_idx & 15;
                int     phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
                int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                                     (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim +
                                     d;
                k_tile[kj][d] = K_Cache[token_addr];
                v_tile[kj][d] = V_Cache[token_addr];
            }
            else
            {
                k_tile[kj][d] = k_zero;
                v_tile[kj][d] = v_zero;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < Q_BLOCK * K_BLOCK; idx += blockDim.x)
        {
            int   qi = idx / K_BLOCK;
            int   kj = idx - qi * K_BLOCK;
            int   q_idx = q_start + qi;
            int   key_idx = tile_start + kj;
            float score = kNeg;

            if (qi < q_count && key_idx < context_len + q_idx + 1)
            {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot +=
                        CudaScalar<scalar_t>::to_float(q_tile[qi][d]) * CudaScalar<scalar_t>::to_float(k_tile[kj][d]);
                }
                score = dot * scale;
            }
            scores[qi][kj] = score;
        }
        __syncthreads();

        for (int qi = tid; qi < Q_BLOCK; qi += blockDim.x)
        {
            if (qi < q_count)
            {
                float tile_m = kNeg;
                for (int kj = 0; kj < K_BLOCK; ++kj)
                {
                    tile_m = fmaxf(tile_m, scores[qi][kj]);
                }

                float old_m = m_val[qi];
                float new_m = fmaxf(old_m, tile_m);
                float old_scale = old_m <= kNeg * 0.5f ? 0.0f : __expf(old_m - new_m);
                float tile_l = 0.0f;
                if (tile_m > kNeg * 0.5f)
                {
                    for (int kj = 0; kj < K_BLOCK; ++kj)
                    {
                        float score = scores[qi][kj];
                        if (score > kNeg * 0.5f)
                        {
                            tile_l += __expf(score - new_m);
                        }
                    }
                }

                m_next[qi] = new_m;
                l_next[qi] = l_val[qi] * old_scale + tile_l;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < Q_BLOCK * head_dim; idx += blockDim.x)
        {
            int qi = idx / head_dim;
            int d = idx - qi * head_dim;
            if (qi < q_count)
            {
                float old_m = m_val[qi];
                float new_m = m_next[qi];
                float beta = old_m <= kNeg * 0.5f ? 0.0f : __expf(old_m - new_m);
                float tile_acc = 0.0f;

                for (int kj = 0; kj < K_BLOCK; ++kj)
                {
                    float score = scores[qi][kj];
                    if (score > kNeg * 0.5f)
                    {
                        tile_acc += __expf(score - new_m) * CudaScalar<scalar_t>::to_float(v_tile[kj][d]);
                    }
                }
                acc[qi][d] = acc[qi][d] * beta + tile_acc;
            }
        }
        __syncthreads();

        for (int qi = tid; qi < Q_BLOCK; qi += blockDim.x)
        {
            if (qi < q_count)
            {
                m_val[qi] = m_next[qi];
                l_val[qi] = l_next[qi];
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Q_BLOCK * head_dim; idx += blockDim.x)
    {
        int qi = idx / head_dim;
        int d = idx - qi * head_dim;
        if (qi < q_count)
        {
            int     q_idx = q_start + qi;
            int64_t out_offset = (int64_t)batch_idx * seq_len * num_heads * head_dim +
                                 (int64_t)q_idx * num_heads * head_dim + (int64_t)head_idx * head_dim + d;
            O[out_offset] = CudaScalar<scalar_t>::from_float(acc[qi][d] / (l_val[qi] + 1e-6f));
        }
    }
}

template <typename scalar_t>
__global__ void paged_prefill_wmma_qk_kernel(const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K_Cache,
                                             const scalar_t* __restrict__ V_Cache, const int* __restrict__ Block_Table,
                                             scalar_t* __restrict__ O, const int* __restrict__ context_lens,
                                             const int num_heads, const int num_kv_heads, const int seq_len,
                                             const int max_num_blocks, const float scale)
{
    constexpr int   q_tile = 16;
    constexpr int   k_tile = 16;
    constexpr int   head_dim = 128;
    constexpr int   dim_tiles = head_dim / 16;
    constexpr float kNeg = -1.0e30f;

    __shared__ scalar_t k_smem[k_tile * 16];
    __shared__ scalar_t q_smem[q_tile][head_dim];
    __shared__ scalar_t v_smem[k_tile][head_dim];
    __shared__ float    scores[q_tile][k_tile];
    __shared__ float    probs[q_tile][k_tile];
    __shared__ float    beta_val[q_tile];
    __shared__ float    acc[q_tile][head_dim];
    __shared__ float    m_val[q_tile];
    __shared__ float    l_val[q_tile];
    __shared__ float    m_next[q_tile];
    __shared__ float    l_next[q_tile];

    int q_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    int q_start = q_tile_idx * q_tile;
    int q_count = seq_len - q_start;
    q_count = q_count > q_tile ? q_tile : q_count;
    if (q_count <= 0 || head_idx >= num_heads) return;

    int context_len = context_lens[batch_idx];
    int total_tokens = context_len + seq_len;
    int max_visible = context_len + q_start + q_count;

    int kv_group = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group;

    int64_t q_batch_base = (int64_t)batch_idx * seq_len * num_heads * head_dim;
    int64_t q_head_base = (int64_t)head_idx * head_dim;

    for (int idx = tid; idx < q_tile * head_dim; idx += blockDim.x)
    {
        int      row = idx / head_dim;
        int      d = idx - row * head_dim;
        scalar_t zero = CudaScalar<scalar_t>::from_float(0.0f);
        q_smem[row][d] =
            row < q_count ? Q[q_batch_base + (int64_t)(q_start + row) * num_heads * head_dim + q_head_base + d] : zero;
        acc[row][d] = 0.0f;
    }
    for (int row = tid; row < q_tile; row += blockDim.x)
    {
        m_val[row] = kNeg;
        l_val[row] = 0.0f;
    }
    __syncthreads();

    for (int key_start = 0; key_start < max_visible; key_start += k_tile)
    {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);

#pragma unroll
        for (int d_tile = 0; d_tile < dim_tiles; ++d_tile)
        {
            for (int idx = tid; idx < k_tile * 16; idx += blockDim.x)
            {
                int      kr = idx / 16;
                int      dc = idx - kr * 16;
                int      key_idx = key_start + kr;
                scalar_t zero = CudaScalar<scalar_t>::from_float(0.0f);

                if (key_idx < total_tokens)
                {
                    int     block_idx = key_idx >> 4;
                    int     block_offset = key_idx & 15;
                    int     phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
                    int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                                         (int64_t)block_offset * num_kv_heads * head_dim +
                                         (int64_t)kv_head_idx * head_dim + d_tile * 16 + dc;
                    k_smem[idx] = K_Cache[token_addr];
                }
                else
                {
                    k_smem[idx] = zero;
                }
            }
            __syncthreads();

            if (tid < 32)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, scalar_t, wmma::row_major> frag_q;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, scalar_t, wmma::col_major> frag_k;

                const scalar_t* q_ptr = &q_smem[0][d_tile * 16];
                wmma::load_matrix_sync(frag_q, q_ptr, head_dim);
                wmma::load_matrix_sync(frag_k, k_smem, 16);
                wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
            }
            __syncthreads();
        }

        if (tid < 32)
        {
            wmma::store_matrix_sync(&scores[0][0], frag_s, k_tile, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = tid; idx < q_tile * k_tile; idx += blockDim.x)
        {
            int   row = idx / k_tile;
            int   col = idx - row * k_tile;
            int   key_idx = key_start + col;
            float val = scores[row][col] * scale;
            if (row >= q_count || key_idx >= total_tokens || key_idx > context_len + q_start + row)
            {
                val = kNeg;
            }
            scores[row][col] = val;
        }
        __syncthreads();

        for (int row = tid; row < q_tile; row += blockDim.x)
        {
            float tile_m = kNeg;
            for (int col = 0; col < k_tile; ++col)
            {
                tile_m = fmaxf(tile_m, scores[row][col]);
            }

            float old_m = m_val[row];
            float new_m = fmaxf(old_m, tile_m);
            float old_scale = old_m <= kNeg * 0.5f ? 0.0f : __expf(old_m - new_m);
            float tile_l = 0.0f;
            if (tile_m > kNeg * 0.5f)
            {
                for (int col = 0; col < k_tile; ++col)
                {
                    float score = scores[row][col];
                    if (score > kNeg * 0.5f)
                    {
                        float prob = __expf(score - new_m);
                        probs[row][col] = prob;
                        tile_l += prob;
                    }
                    else
                    {
                        probs[row][col] = 0.0f;
                    }
                }
            }
            else
            {
                for (int col = 0; col < k_tile; ++col)
                {
                    probs[row][col] = 0.0f;
                }
            }

            m_next[row] = new_m;
            l_next[row] = l_val[row] * old_scale + tile_l;
            beta_val[row] = old_scale;
        }
        __syncthreads();

        for (int idx = tid; idx < k_tile * head_dim; idx += blockDim.x)
        {
            int      kr = idx / head_dim;
            int      d = idx - kr * head_dim;
            int      key_idx = key_start + kr;
            scalar_t zero = CudaScalar<scalar_t>::from_float(0.0f);

            if (key_idx < total_tokens)
            {
                int     block_idx = key_idx >> 4;
                int     block_offset = key_idx & 15;
                int     phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
                int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                                     (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim +
                                     d;
                v_smem[kr][d] = V_Cache[token_addr];
            }
            else
            {
                v_smem[kr][d] = zero;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < q_tile * head_dim; idx += blockDim.x)
        {
            int row = idx / head_dim;
            int d = idx - row * head_dim;
            if (row >= q_count) continue;
            float tile_acc = 0.0f;

            for (int col = 0; col < k_tile; ++col)
            {
                float prob = probs[row][col];
                if (prob > 0.0f)
                {
                    tile_acc += prob * CudaScalar<scalar_t>::to_float(v_smem[col][d]);
                }
            }
            acc[row][d] = acc[row][d] * beta_val[row] + tile_acc;
        }
        __syncthreads();

        for (int row = tid; row < q_tile; row += blockDim.x)
        {
            m_val[row] = m_next[row];
            l_val[row] = l_next[row];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < q_tile * head_dim; idx += blockDim.x)
    {
        int row = idx / head_dim;
        int d = idx - row * head_dim;
        int q_idx = q_start + row;
        if (row >= q_count) continue;
        int64_t out_offset = (int64_t)batch_idx * seq_len * num_heads * head_dim +
                             (int64_t)q_idx * num_heads * head_dim + (int64_t)head_idx * head_dim + d;
        O[out_offset] = CudaScalar<scalar_t>::from_float(acc[row][d] / (l_val[row] + 1e-6f));
    }
}

template <typename Scalar>
void launch_paged_prefill_typed(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                                const AttentionOptions& options, float scale, cudaStream_t stream)
{
    int batch_size = query.shape()[0];
    int sequence_length = query.shape()[1];
    int num_heads = query.shape()[2];
    int head_dim = query.shape()[3];

    if (sequence_length > 1 && head_dim == 128)
    {
        dim3 grid((sequence_length + 15) / 16, num_heads, batch_size);
        paged_prefill_wmma_qk_kernel<Scalar>
            <<<grid, 128, 0, stream>>>(static_cast<const Scalar*>(query.data()), static_cast<const Scalar*>(key.data()),
                                       static_cast<const Scalar*>(value.data()), options.block_table,
                                       static_cast<Scalar*>(output.data()), options.context_lengths, num_heads,
                                       options.kv_head_count, sequence_length, options.max_context_blocks, scale);
        return;
    }

    constexpr int query_block = 8;
    constexpr int key_block = 64;
    constexpr int max_blockwise_head_dim = 128;
    if (head_dim <= max_blockwise_head_dim)
    {
        dim3 grid((sequence_length + query_block - 1) / query_block, num_heads, batch_size);
        paged_attention_blockwise_kernel<Scalar, query_block, key_block, max_blockwise_head_dim>
            <<<grid, 256, 0, stream>>>(static_cast<const Scalar*>(query.data()), static_cast<const Scalar*>(key.data()),
                                       static_cast<const Scalar*>(value.data()), options.block_table,
                                       static_cast<Scalar*>(output.data()), options.context_lengths, num_heads,
                                       options.kv_head_count, sequence_length, head_dim, options.max_context_blocks,
                                       scale);
        return;
    }

    int  threads = attention_detail::thread_count(head_dim);
    dim3 grid(sequence_length, num_heads, batch_size);
    paged_attention_kernel<Scalar><<<grid, threads, threads * sizeof(float), stream>>>(
        static_cast<const Scalar*>(query.data()), static_cast<const Scalar*>(key.data()),
        static_cast<const Scalar*>(value.data()), options.block_table, static_cast<Scalar*>(output.data()), batch_size,
        options.context_lengths, num_heads, options.kv_head_count, sequence_length, head_dim,
        options.max_context_blocks, scale);
}
}  // namespace

void attention_detail::launch_paged_prefill(Tensor& query, Tensor& key, Tensor& value, Tensor& output,
                                            const AttentionOptions& options, float scale,
                                            const device::Context& context)
{
    if (query.dtype() == DType::BF16)
    {
        launch_paged_prefill_typed<__nv_bfloat16>(query, key, value, output, options, scale, context.stream());
    }
    else
    {
        launch_paged_prefill_typed<half>(query, key, value, output, options, scale, context.stream());
    }
}
}  // namespace firefly::kernels
