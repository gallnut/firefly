#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "firefly/cuda_dtype.cuh"
#include "firefly/kernels.h"

using namespace nvcuda;

namespace firefly::kernels
{
namespace
{
struct DecodeSplitScratch
{
    Tensor  partial_m;
    Tensor  partial_l;
    Tensor  partial_acc;
    int64_t partial_m_capacity = 0;
    int64_t partial_l_capacity = 0;
    int64_t partial_acc_capacity = 0;
};

DecodeSplitScratch& decode_split_scratch()
{
    static thread_local DecodeSplitScratch* scratch = new DecodeSplitScratch();
    return *scratch;
}

void ensure_decode_split_scratch(int batch_size, int num_heads, int num_splits, int head_dim)
{
    auto& scratch = decode_split_scratch();
    int64_t partial_count = (int64_t)batch_size * num_heads * num_splits;
    int64_t acc_count = partial_count * head_dim;

    if (scratch.partial_m_capacity < partial_count)
    {
        scratch.partial_m = Tensor({partial_count}, DType::F32, Device::CUDA);
        scratch.partial_m_capacity = partial_count;
    }
    if (scratch.partial_l_capacity < partial_count)
    {
        scratch.partial_l = Tensor({partial_count}, DType::F32, Device::CUDA);
        scratch.partial_l_capacity = partial_count;
    }
    if (scratch.partial_acc_capacity < acc_count)
    {
        scratch.partial_acc = Tensor({acc_count}, DType::F32, Device::CUDA);
        scratch.partial_acc_capacity = acc_count;
    }
}

bool force_single_decode()
{
    const char* value = std::getenv("FIREFLY_DECODE_BACKEND");
    return value != nullptr && std::string(value) == "single";
}

bool force_split_decode()
{
    const char* value = std::getenv("FIREFLY_DECODE_BACKEND");
    return value != nullptr && std::string(value) == "split";
}

int decode_split_size()
{
    const char* value = std::getenv("FIREFLY_DECODE_SPLIT_SIZE");
    if (value == nullptr || std::string(value).empty()) return 256;
    int split_size = std::atoi(value);
    if (split_size == 128 || split_size == 256 || split_size == 512) return split_size;
    std::cerr << "Unsupported FIREFLY_DECODE_SPLIT_SIZE='" << value << "', using 256" << std::endl;
    return 256;
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
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
}  // namespace

const char* attention_backend_name(AttentionBackend backend)
{
    switch (backend)
    {
        case AttentionBackend::Auto:
            return "auto";
        case AttentionBackend::Paged:
            return "paged";
        case AttentionBackend::Contiguous:
            return "contiguous";
        case AttentionBackend::External:
            return "external";
    }
    return "unknown";
}

AttentionBackend get_attention_backend()
{
    static AttentionBackend backend = []()
    {
        const char* env = std::getenv("FIREFLY_ATTENTION_BACKEND");
        if (env == nullptr || std::string(env).empty() || std::string(env) == "auto")
        {
            return AttentionBackend::Auto;
        }
        std::string value(env);
        if (value == "paged") return AttentionBackend::Paged;
        if (value == "contiguous") return AttentionBackend::Contiguous;
        if (value == "external") return AttentionBackend::External;
        std::cerr << "Unknown FIREFLY_ATTENTION_BACKEND='" << value << "', using auto" << std::endl;
        return AttentionBackend::Auto;
    }();
    return backend;
}

// -----------------------------------------------------------------------------
// Correct generic attention kernels.
// One block owns one attention row. The block size is selected so every head
// element has a lane, avoiding hidden 64/128/8-wide assumptions.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void causal_attention_scalar_kernel(const scalar_t* Q, const scalar_t* K, const scalar_t* V, scalar_t* O,
                                               int seq_len, int num_heads, int num_kv_heads, int head_dim, float scale)
{
    extern __shared__ float reduce[];

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int64_t stride_b_q = (int64_t)seq_len * num_heads * head_dim;
    int64_t stride_b_kv = (int64_t)seq_len * num_kv_heads * head_dim;
    int64_t stride_s_q = num_heads * head_dim;
    int64_t stride_s_kv = num_kv_heads * head_dim;

    const scalar_t* q_ptr = Q + batch_idx * stride_b_q + query_idx * stride_s_q + head_idx * head_dim;
    const scalar_t* k_base = K + batch_idx * stride_b_kv + kv_head_idx * head_dim;
    const scalar_t* v_base = V + batch_idx * stride_b_kv + kv_head_idx * head_dim;
    scalar_t*       o_ptr = O + batch_idx * stride_b_q + query_idx * stride_s_q + head_idx * head_dim;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc = 0.0f;

    for (int key_idx = 0; key_idx <= query_idx; ++key_idx)
    {
        const scalar_t* k_ptr = k_base + key_idx * stride_s_kv;

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
            const scalar_t* v_ptr = v_base + key_idx * stride_s_kv;
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
__global__ void paged_attention_blockwise_kernel(const scalar_t* __restrict__ Q,
                                                 const scalar_t* __restrict__ K_Cache,
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
                int block_idx = key_idx >> 4;
                int block_offset = key_idx & 15;
                int phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
                int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                                     (int64_t)block_offset * num_kv_heads * head_dim +
                                     (int64_t)kv_head_idx * head_dim + d;
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
            int qi = idx / K_BLOCK;
            int kj = idx - qi * K_BLOCK;
            int q_idx = q_start + qi;
            int key_idx = tile_start + kj;
            float score = kNeg;

            if (qi < q_count && key_idx < context_len + q_idx + 1)
            {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += CudaScalar<scalar_t>::to_float(q_tile[qi][d]) *
                           CudaScalar<scalar_t>::to_float(k_tile[kj][d]);
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
            int q_idx = q_start + qi;
            int64_t out_offset = (int64_t)batch_idx * seq_len * num_heads * head_dim +
                                 (int64_t)q_idx * num_heads * head_dim + (int64_t)head_idx * head_dim + d;
            O[out_offset] = CudaScalar<scalar_t>::from_float(acc[qi][d] / (l_val[qi] + 1e-6f));
        }
    }
}

template <typename scalar_t>
__global__ void paged_prefill_wmma_qk_kernel(const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K_Cache,
                                             const scalar_t* __restrict__ V_Cache,
                                             const int* __restrict__ Block_Table, scalar_t* __restrict__ O,
                                             const int* __restrict__ context_lens, const int num_heads,
                                             const int num_kv_heads, const int seq_len, const int max_num_blocks,
                                             const float scale)
{
    constexpr int q_tile = 16;
    constexpr int k_tile = 16;
    constexpr int head_dim = 128;
    constexpr int dim_tiles = head_dim / 16;
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
        int row = idx / head_dim;
        int d = idx - row * head_dim;
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
                int kr = idx / 16;
                int dc = idx - kr * 16;
                int key_idx = key_start + kr;
                scalar_t zero = CudaScalar<scalar_t>::from_float(0.0f);

                if (key_idx < total_tokens)
                {
                    int block_idx = key_idx >> 4;
                    int block_offset = key_idx & 15;
                    int phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
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
            int row = idx / k_tile;
            int col = idx - row * k_tile;
            int key_idx = key_start + col;
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
            int kr = idx / head_dim;
            int d = idx - kr * head_dim;
            int key_idx = key_start + kr;
            scalar_t zero = CudaScalar<scalar_t>::from_float(0.0f);

            if (key_idx < total_tokens)
            {
                int block_idx = key_idx >> 4;
                int block_offset = key_idx & 15;
                int phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
                int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                                     (int64_t)block_offset * num_kv_heads * head_dim +
                                     (int64_t)kv_head_idx * head_dim + d;
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

template <typename scalar_t, int SPLIT_SIZE>
__global__ void paged_decode_split_partial_kernel(const scalar_t* __restrict__ Q,
                                                  const scalar_t* __restrict__ K_Cache,
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

    int partial_idx = (batch_idx * num_heads + head_idx) * num_splits + split_idx;
    float acc = 0.0f;
    float m_i = -1.0e30f;
    float l_i = 0.0f;

    const scalar_t* q_ptr =
        Q + (int64_t)batch_idx * num_heads * head_dim + (int64_t)head_idx * head_dim;

    for (int key_idx = key_start; key_idx < key_end; ++key_idx)
    {
        int block_idx = key_idx >> 4;
        int block_offset = key_idx & 15;
        int phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
        int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                             (int64_t)block_offset * num_kv_heads * head_dim +
                             (int64_t)kv_head_idx * head_dim;

        float partial_dot = 0.0f;
        if (tid < head_dim)
        {
            partial_dot = CudaScalar<scalar_t>::to_float(q_ptr[tid]) *
                          CudaScalar<scalar_t>::to_float(K_Cache[token_addr + tid]);
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

template <typename scalar_t, int SPLIT_SIZE, int HEAD_DIM>
__global__ void paged_decode_split_dual_partial_kernel(const scalar_t* __restrict__ Q,
                                                       const scalar_t* __restrict__ K_Cache,
                                                       const scalar_t* __restrict__ V_Cache,
                                                       const int* __restrict__ Block_Table, const int* context_lens,
                                                       float* __restrict__ partial_m,
                                                       float* __restrict__ partial_l,
                                                       float* __restrict__ partial_acc, int num_heads,
                                                       int num_kv_heads, int max_num_blocks,
                                                       int num_splits, float scale)
{
    extern __shared__ float reduce[];
    float* reduce0 = reduce;
    float* reduce1 = reduce + blockDim.x;

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
    float q0_val = CudaScalar<scalar_t>::to_float(q0[tid]);
    float q1_val = CudaScalar<scalar_t>::to_float(q1[tid]);

    for (int key_idx = key_start; key_idx < key_end; ++key_idx)
    {
        int block_idx = key_idx >> 4;
        int block_offset = key_idx & 15;
        int phys_block = Block_Table[batch_idx * max_num_blocks + block_idx];
        int64_t token_addr = (int64_t)phys_block * 16 * num_kv_heads * HEAD_DIM +
                             (int64_t)block_offset * num_kv_heads * HEAD_DIM +
                             (int64_t)kv_head_idx * HEAD_DIM;

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
__global__ void paged_decode_split_reduce_kernel(const float* __restrict__ partial_m,
                                                 const float* __restrict__ partial_l,
                                                 const float* __restrict__ partial_acc,
                                                 scalar_t* __restrict__ O, int num_heads, int head_dim,
                                                 int num_splits)
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
                                                      const float* __restrict__ partial_acc,
                                                      scalar_t* __restrict__ O, int num_heads, int head_dim,
                                                      int num_splits)
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

// -----------------------------------------------------------------------------
// Host Dispatcher
// -----------------------------------------------------------------------------
int attention_thread_count(int head_dim)
{
    if (head_dim <= 0 || head_dim > 1024)
    {
        throw std::runtime_error("attention currently supports 1 <= head_dim <= 1024");
    }

    int threads = 32;
    while (threads < head_dim)
    {
        threads <<= 1;
    }
    return threads;
}

template <typename scalar_t>
void launch_scalar_prefill(Tensor& q, Tensor& k, Tensor& v, Tensor& output, int kv_head_num, int seq_len,
                           int batch_size, int num_heads, int head_dim, float scale)
{
    int    threads = attention_thread_count(head_dim);
    dim3   block(threads);
    dim3   grid(seq_len, num_heads, batch_size);
    size_t smem_size = threads * sizeof(float);

    causal_attention_scalar_kernel<scalar_t><<<grid, block, smem_size, get_default_stream()>>>(
        static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
        static_cast<const scalar_t*>(v.data()), static_cast<scalar_t*>(output.data()), seq_len, num_heads, kv_head_num,
        head_dim, scale);
}

template <typename scalar_t>
void launch_paged_decode_split(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* kv_cache_block_table,
                               int kv_head_num, int max_context_blocks, const int* context_lens, int batch_size,
                               int num_heads, int head_dim, float scale, int max_decode_context_len)
{
    auto launch = [&](auto split_tag)
    {
        constexpr int split_size = decltype(split_tag)::value;
        int split_token_limit = max_decode_context_len > 0 ? max_decode_context_len + 1 : max_context_blocks * 16;
        split_token_limit = std::min(split_token_limit, max_context_blocks * 16);
        int num_splits = std::max(1, (split_token_limit + split_size - 1) / split_size);
        ensure_decode_split_scratch(batch_size, num_heads, num_splits, head_dim);
        auto& scratch = decode_split_scratch();

        int  threads = attention_thread_count(head_dim);
        bool use_dual_gqa_decode = num_heads == kv_head_num * 2 && head_dim == 128;
        if (use_dual_gqa_decode)
        {
            dim3 partial_grid(num_splits, kv_head_num, batch_size);
            paged_decode_split_dual_partial_kernel<scalar_t, split_size, 128>
                <<<partial_grid, threads, 2 * threads * sizeof(float), get_default_stream()>>>(
                    static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
                    static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table),
                    context_lens, static_cast<float*>(scratch.partial_m.data()),
                    static_cast<float*>(scratch.partial_l.data()), static_cast<float*>(scratch.partial_acc.data()),
                    num_heads, kv_head_num, max_context_blocks, num_splits, scale);
        }
        else
        {
            dim3 partial_grid(num_splits, num_heads, batch_size);
            paged_decode_split_partial_kernel<scalar_t, split_size>
                <<<partial_grid, threads, threads * sizeof(float), get_default_stream()>>>(
                    static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
                    static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table),
                    context_lens, static_cast<float*>(scratch.partial_m.data()),
                    static_cast<float*>(scratch.partial_l.data()), static_cast<float*>(scratch.partial_acc.data()),
                    num_heads, kv_head_num, head_dim, max_context_blocks, num_splits, scale);
        }

        if (use_dual_gqa_decode)
        {
            dim3 reduce_grid(kv_head_num, batch_size);
            paged_decode_split_dual_reduce_kernel<scalar_t><<<reduce_grid, threads, 0, get_default_stream()>>>(
                static_cast<const float*>(scratch.partial_m.data()),
                static_cast<const float*>(scratch.partial_l.data()),
                static_cast<const float*>(scratch.partial_acc.data()), static_cast<scalar_t*>(output.data()),
                num_heads, head_dim, num_splits);
        }
        else
        {
            dim3 reduce_grid(num_heads, batch_size);
            paged_decode_split_reduce_kernel<scalar_t><<<reduce_grid, threads, 0, get_default_stream()>>>(
                static_cast<const float*>(scratch.partial_m.data()),
                static_cast<const float*>(scratch.partial_l.data()),
                static_cast<const float*>(scratch.partial_acc.data()), static_cast<scalar_t*>(output.data()),
                num_heads, head_dim, num_splits);
        }
    };

    switch (decode_split_size())
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

template <typename scalar_t>
void launch_paged_attention(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* kv_cache_block_table,
                            int kv_head_num, int max_context_blocks, const int* context_lens, int batch_size,
                            int num_heads, int seq_len, int head_dim, float scale, bool prefer_split_decode,
                            int max_decode_context_len)
{
    bool use_split = prefer_split_decode && seq_len == 1 && head_dim <= 1024;
    if (force_single_decode()) use_split = false;
    if (force_split_decode() && seq_len == 1 && head_dim <= 1024) use_split = true;

    if (use_split)
    {
        launch_paged_decode_split<scalar_t>(q, k, v, output, kv_cache_block_table, kv_head_num, max_context_blocks,
                                            context_lens, batch_size, num_heads, head_dim, scale,
                                            max_decode_context_len);
        return;
    }

    if (seq_len > 1 && head_dim == 128)
    {
        dim3 grid((seq_len + 15) / 16, num_heads, batch_size);
        dim3 block(128);
        paged_prefill_wmma_qk_kernel<scalar_t><<<grid, block, 0, get_default_stream()>>>(
            static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
            static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table),
            static_cast<scalar_t*>(output.data()), context_lens, num_heads, kv_head_num, seq_len, max_context_blocks,
            scale);
        return;
    }

    constexpr int q_block = 8;
    constexpr int k_block = 64;
    constexpr int max_blockwise_head_dim = 128;
    if (head_dim <= max_blockwise_head_dim)
    {
        dim3 grid((seq_len + q_block - 1) / q_block, num_heads, batch_size);
        dim3 block(256);
        paged_attention_blockwise_kernel<scalar_t, q_block, k_block, max_blockwise_head_dim>
            <<<grid, block, 0, get_default_stream()>>>(
                static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
                static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table),
                static_cast<scalar_t*>(output.data()), context_lens, num_heads, kv_head_num, seq_len, head_dim,
                max_context_blocks, scale);
        return;
    }

    int  threads = attention_thread_count(head_dim);
    dim3 block_decode(threads);
    dim3 grid_decode(seq_len, num_heads, batch_size);

    paged_attention_kernel<scalar_t><<<grid_decode, block_decode, threads * sizeof(float), get_default_stream()>>>(
        static_cast<const scalar_t*>(q.data()), static_cast<const scalar_t*>(k.data()),
        static_cast<const scalar_t*>(v.data()), static_cast<const int*>(kv_cache_block_table),
        static_cast<scalar_t*>(output.data()), batch_size, context_lens, num_heads, kv_head_num, seq_len, head_dim,
        max_context_blocks, scale);
}

void attention(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* kv_cache_block_table, int kv_head_num,
               int seq_len, int max_context_blocks, const int* context_lens, bool prefer_split_decode,
               int max_decode_context_len)
{
    attention_ex(q, k, v, output, kv_cache_block_table, kv_head_num, seq_len, max_context_blocks, context_lens,
                 AttentionBackend::Auto, prefer_split_decode, max_decode_context_len);
}

void attention_ex(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* kv_cache_block_table, int kv_head_num,
                  int seq_len, int max_context_blocks, const int* context_lens, AttentionBackend backend,
                  bool prefer_split_decode, int max_decode_context_len, int prefill_context_len)
{
    require_float16_or_bfloat16(q.dtype(), "attention");
    require_same_dtype(q.dtype(), k.dtype(), "attention");
    require_same_dtype(q.dtype(), v.dtype(), "attention");
    require_same_dtype(q.dtype(), output.dtype(), "attention");

    if (q.shape().size() != 4)
    {
        throw std::runtime_error("attention expects rank-4 q tensor");
    }
    if (seq_len <= 0)
    {
        throw std::runtime_error("attention requires seq_len > 0");
    }

    int   batch_size = q.shape()[0];
    int   num_heads = q.shape()[2];
    int   head_dim = q.shape()[3];
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (q.shape()[1] != seq_len)
    {
        throw std::runtime_error("attention seq_len does not match q shape");
    }
    if (kv_head_num <= 0 || num_heads % kv_head_num != 0)
    {
        throw std::runtime_error("attention requires num_heads to be divisible by kv_head_num");
    }
    if (output.numel() != static_cast<int64_t>(batch_size) * seq_len * num_heads * head_dim)
    {
        throw std::runtime_error("attention output element count does not match q shape");
    }

    if (backend == AttentionBackend::Auto)
    {
        backend = kv_cache_block_table == nullptr ? AttentionBackend::Contiguous : AttentionBackend::Paged;
    }
    if (backend == AttentionBackend::External)
    {
        if (kv_cache_block_table == nullptr && torch_flash_attention(q, k, v, output, kv_head_num, seq_len))
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("external attention launch failed: ") + cudaGetErrorString(err));
            }
            return;
        }
        if (kv_cache_block_table != nullptr && seq_len >= 1 && prefill_context_len >= 0 &&
            torch_flash_paged_prefill(q, k, v, output, static_cast<const int*>(kv_cache_block_table), kv_head_num,
                                      seq_len, max_context_blocks, prefill_context_len))
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("external paged prefill attention failed: ") +
                                         cudaGetErrorString(err));
            }
            return;
        }
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "FIREFLY_ATTENTION_BACKEND=external requested, but no compatible external SDPA backend "
                         "handled this shape; falling back to "
                      << (kv_cache_block_table == nullptr ? "contiguous" : "paged") << std::endl;
            warned = true;
        }
        backend = kv_cache_block_table == nullptr ? AttentionBackend::Contiguous : AttentionBackend::Paged;
    }
    if (backend == AttentionBackend::Contiguous && kv_cache_block_table != nullptr)
    {
        backend = AttentionBackend::Paged;
    }

    if (backend == AttentionBackend::Contiguous)
    {
        if (k.shape().size() != 4 || v.shape().size() != 4 || k.shape()[0] != batch_size ||
            v.shape()[0] != batch_size || k.shape()[1] != seq_len || v.shape()[1] != seq_len ||
            k.shape()[2] != kv_head_num || v.shape()[2] != kv_head_num || k.shape()[3] != head_dim ||
            v.shape()[3] != head_dim)
        {
            throw std::runtime_error("prefill attention expects k/v shape [batch, seq_len, kv_heads, head_dim]");
        }

        if (q.dtype() == DType::BF16)
        {
            launch_scalar_prefill<__nv_bfloat16>(q, k, v, output, kv_head_num, seq_len, batch_size, num_heads, head_dim,
                                                 scale);
        }
        else
        {
            launch_scalar_prefill<half>(q, k, v, output, kv_head_num, seq_len, batch_size, num_heads, head_dim, scale);
        }
    }
    else if (backend == AttentionBackend::Paged)
    {
        if (context_lens == nullptr)
        {
            throw std::runtime_error("paged attention requires context_lens");
        }
        if (k.shape().size() != 4 || v.shape().size() != 4 || k.shape()[1] != 16 || v.shape()[1] != 16 ||
            k.shape()[2] != kv_head_num || v.shape()[2] != kv_head_num || k.shape()[3] != head_dim ||
            v.shape()[3] != head_dim)
        {
            throw std::runtime_error("paged attention expects k/v cache shape [blocks, 16, kv_heads, head_dim]");
        }

        if (q.dtype() == DType::BF16)
        {
            launch_paged_attention<__nv_bfloat16>(q, k, v, output, kv_cache_block_table, kv_head_num,
                                                  max_context_blocks, context_lens, batch_size, num_heads, seq_len,
                                                  head_dim, scale, prefer_split_decode, max_decode_context_len);
        }
        else
        {
            launch_paged_attention<half>(q, k, v, output, kv_cache_block_table, kv_head_num, max_context_blocks,
                                         context_lens, batch_size, num_heads, seq_len, head_dim, scale,
                                         prefer_split_decode, max_decode_context_len);
        }
    }
    else
    {
        throw std::runtime_error("unsupported attention backend");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("attention launch failed: ") + cudaGetErrorString(err));
    }
}

}  // namespace firefly::kernels
