#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>

#include "firefly/kernels.h"

using namespace nvcuda;

namespace firefly::kernels
{

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int Q_TILE_SIZE = 16;  // WMMA M

// -----------------------------------------------------------------------------
// Optimized FlashAttention Forward Kernel using Tensor Cores (WMMA)
// -----------------------------------------------------------------------------
__global__ void flash_attn_wmma_kernel(const half* Q, const half* K, const half* V, half* O, int seq_len, int num_heads,
                                       int num_kv_heads, int head_dim, float scale)
{
    // WMMA Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q[8];  // 128 dim / 16 = 8 chunks

    // Output Accumulators: 8 chunks of 16 columns = 128 columns
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_O[8];

    // Scratch fragments
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;  // Transposed K for Q*K^T
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              frag_S;  // Output Scores

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;  // Softmaxed Scores
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;  // V chunk

    // Indices
    int bx = blockIdx.x;  // Q-Tile Index
    int by = blockIdx.y;  // Head Index
    int bz = blockIdx.z;  // Batch Index
    int tid = threadIdx.x;

    int kv_head_idx = by / (num_heads / num_kv_heads);
    int m_start = bx * Q_TILE_SIZE;

    // Bounds check for Q-tile
    if (m_start >= seq_len) return;

    // Strides (in elements)
    int64_t stride_b_q = (int64_t)seq_len * num_heads * head_dim;
    int64_t stride_b_kv = (int64_t)seq_len * num_kv_heads * head_dim;
    int64_t stride_h_q = head_dim;
    int64_t stride_s_q = num_heads * head_dim;
    int64_t stride_s_kv = num_kv_heads * head_dim;  // Seq stride for KV

    // Base Pointers
    const half* q_base = Q + bz * stride_b_q + by * stride_h_q + m_start * stride_s_q;
    const half* k_base = K + bz * stride_b_kv + kv_head_idx * stride_h_q;
    const half* v_base = V + bz * stride_b_kv + kv_head_idx * stride_h_q;
    half*       o_base = O + bz * stride_b_q + by * stride_h_q + m_start * stride_s_q;

    // Shared Memory scratchpad for K transposition and Softmax
    // Needs to hold 16x16 half K-tile = 256 halves = 512 bytes.
    // Needs to hold 16x16 float S-tile = 256 floats = 1024 bytes.
    extern __shared__ char smem_buffer[];
    half*                  k_smem = reinterpret_cast<half*>(smem_buffer);    // [16*16]
    float*                 s_smem = reinterpret_cast<float*>(k_smem + 256);  // [16*16]

// -------------------------------------------------------------------------
// 1. Load Q into registers (Fragments)
// -------------------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        // Load sub-matrix of Q [16, 16] starting at col i*16
        // Address: q_base + i*16 (col offset). LDM = stride_s_q
        wmma::load_matrix_sync(frag_Q[i], q_base + i * 16, stride_s_q);
    }

// Initialize O accumulators to 0
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        wmma::fill_fragment(frag_O[i], 0.0f);
    }

    // Softmax Statistics (per row)
    // Each thread in warp handles specific rows?
    // Actually, we keep stats in registers but since we do softmax via SMEM,
    // we can update a local register 'my_m' and 'my_l' if we map threads to rows.
    // 32 threads, 16 rows.
    // Let's assume thread i (0..15) handles row i. Thread i (16..31) handles row i-16.
    int my_row;
    if (tid < 16)
        my_row = tid;
    else
        my_row = tid - 16;

    float m_i = -1e30f;
    float l_i = 0.0f;

    // -------------------------------------------------------------------------
    // 2. Loop over KV chunks (16 rows at a time)
    // -------------------------------------------------------------------------
    for (int n_start = 0; n_start < seq_len; n_start += 16)
    {
        // -------------------------------------
        // Compute S = Q * K^T
        // -------------------------------------
        wmma::fill_fragment(frag_S, 0.0f);

// Accumulate dot product over Head Dimension (8 chunks of 16)
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            // Step 2a: Load K sub-block [16, 16] into Shared Memory
            // (to enable Col-Major load for transpose effect)
            // K Chunk: rows [n_start, n_start+15], cols [i*16, i*16+15]
            // Global Address: k_base + n_start * stride_s_kv + i * 16
            // But wmma::load wants contiguous memory usually?
            // Shared Mem load handles strides if handled manually?
            // Here `k_chunk_ptr` access is manually strided.
            const half* k_chunk_ptr = k_base + n_start * stride_s_kv + i * 16;

// Collaborative load 256 half elements (8 per thread)
#pragma unroll
            for (int x = 0; x < 8; ++x)
            {
                int idx = tid * 8 + x;  // 0..255
                int r = idx / 16;       // row in tile 0..15
                int c = idx % 16;       // col in tile 0..15

                if ((n_start + r) < seq_len)
                {
                    k_smem[idx] = k_chunk_ptr[r * stride_s_kv + c];
                }
                else
                {
                    k_smem[idx] = __float2half(0.0f);
                }
            }
            __syncwarp();  // Ensure SMEM is ready

            // Step 2b: Load from SMEM as Col Major -> Acts as Transpose
            // frag_K will hold K^T.
            // LDM is 16 (stride of SMEM).
            wmma::load_matrix_sync(frag_K, k_smem, 16);

            // Step 2c: MMA Accumulate
            // S += Q_chunk * K_chunk^T
            wmma::mma_sync(frag_S, frag_Q[i], frag_K, frag_S);
        }

        // -------------------------------------
        // Softmax Update
        // -------------------------------------
        // Store Scores (S) to SMEM to compute exp/max
        wmma::store_matrix_sync(s_smem, frag_S, 16, wmma::mem_row_major);
        __syncwarp();

        // Compute Softmax stats locally
        float row_max = -1e30f;
        // Search max in the row assigned to this thread
        // Note: 2 threads cover same row. Result is same since input is deterministic.
        // It's fine to duplicate work.

        // Use vector load optimization? No, bank conflicts on float access?
        // 16 floats per row.
        for (int c = 0; c < 16; ++c)
        {
            float val = s_smem[my_row * 16 + c] * scale;
            if ((n_start + c) >= seq_len || (n_start + c) > (m_start + my_row))
                val = -1e30f;               // Mask out padding and causal future
            s_smem[my_row * 16 + c] = val;  // Store scaled
            row_max = fmaxf(row_max, val);
        }

        // Update global stats
        float m_prev = m_i;
        m_i = fmaxf(m_i, row_max);

        float e_score = expf(row_max - m_i);      // Contrib of current block max
        float e_correction = expf(m_prev - m_i);  // Correction for previous sum

        // Compute sum of exps for current block
        float row_sum = 0.0f;
        for (int c = 0; c < 16; ++c)
        {
            float val = s_smem[my_row * 16 + c];
            float p_val = expf(val - m_i);
            s_smem[my_row * 16 + c] = p_val;  // Overwrite S with P (unnormalized)
            row_sum += p_val;
        }

        l_i = l_i * e_correction + row_sum;

        // Wait for P to be ready in SMEM
        __syncwarp();

// -------------------------------------
// O Update: O = O * correction + P * V
// -------------------------------------

// 1. Scale existing O accumulator
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            // Iterate fragment elements? No, WMMA accumulator opaque.
            // Standard trick: `frag_O[i] = frag_O[i] * e_correction`?
            // Need to decompose into component-wise mult.
            for (int t = 0; t < frag_O[i].num_elements; ++t)
            {
                frag_O[i].x[t] *= e_correction;
            }
        }

// 2. Load P (calculated in S_smem) into Fragment
// Convert float P -> half P for MMA
// We reuse K_SMEM for P_half? Yes.
#pragma unroll
        for (int x = 0; x < 8; ++x)
        {
            int idx = tid * 8 + x;
            k_smem[idx] = __float2half(s_smem[idx]);
        }
        __syncwarp();

        wmma::load_matrix_sync(frag_P, k_smem, 16);

// 3. Matmul P * V
// V is [16, 128]. We iterate cols in chunks of 16.
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            // Load V sub-block [16, 16]
            // Rows [0..15], Cols [i*16 .. i*16+15]
            // Address: v_base + n_start * stride_s_kv + i * 16
            // Layout: Row Major.
            wmma::load_matrix_sync(frag_V, v_base + n_start * stride_s_kv + i * 16, stride_s_kv);

            // Accumulate
            wmma::mma_sync(frag_O[i], frag_P, frag_V, frag_O[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Final Store (Output)
    // -------------------------------------------------------------------------
    // 1. Normalize O in place (divide by l_i)
    float inv_li = 1.0f / (l_i + 1e-6f);

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        for (int t = 0; t < frag_O[i].num_elements; ++t)
        {
            frag_O[i].x[t] *= inv_li;
        }

        // 2. Store to global memory: Float Fragment -> Float SMEM -> Half Global
        wmma::store_matrix_sync(s_smem, frag_O[i], 16, wmma::mem_row_major);
        __syncwarp();

// 3. Manual copy from SMEM (float) to Global (half)
// 256 elements, 32 threads -> 8 elements per thread
#pragma unroll
        for (int x = 0; x < 8; ++x)
        {
            int idx = tid * 8 + x;
            int r = idx / 16;
            int c = idx % 16;

            // Global Address Calculation:
            // Reference: o_base is start of block (i=0 of head dim)
            // Global Offset = r * stride_s_q + (i * 16 + c)
            size_t global_offset = (size_t)r * stride_s_q + (i * 16 + c);

            if ((m_start + r) < seq_len)
            {
                o_base[global_offset] = __float2half(s_smem[idx]);
            }
        }
        __syncwarp();
    }
}

// -----------------------------------------------------------------------------
// Paged Attention (Decode Phase) - Kept Simple / Baseline for now
// -----------------------------------------------------------------------------
__global__ void paged_attention_kernel(const half* __restrict__ Q, const half* __restrict__ K_Cache,
                                       const half* __restrict__ V_Cache, const int* __restrict__ Block_Table,
                                       half* __restrict__ O, const int batch_size, const int* __restrict__ context_lens,
                                       const int num_heads, const int num_kv_heads, const int head_dim,
                                       const int max_num_blocks, const float scale)
{
    // Keeping the functional scalar implementation for decode
    int bx = blockIdx.x;  // Head Index
    int by = blockIdx.y;  // Batch Index
    int tx = threadIdx.x;

    if (bx >= num_heads) return;

    int     kv_head_idx = bx / (num_heads / num_kv_heads);
    int64_t q_offset = (int64_t)by * num_heads * head_dim + (int64_t)bx * head_dim;

    // 128 threads / 16 threads = 8 tokens processed concurrently per iteration
    int token_idx = tx / 16;
    int vec_idx = tx % 16;

    // Each vec_idx handles 8 halves (1 float4)
    float4 q_vec_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (vec_idx * 8 < head_dim)
    {
        q_vec_f4 = *reinterpret_cast<const float4*>(&Q[q_offset + vec_idx * 8]);
    }
    half* q_vec = reinterpret_cast<half*>(&q_vec_f4);

    // +1 to include the current token's decoded KV, which was just appended
    int context_len = context_lens[by] + 1;
    int num_blocks = (context_len + 15) / 16;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc_val[8] = {0};

    // Shared memory for reduction across the 8 token groups
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_acc[8][128];

    for (int b = 0; b < num_blocks; ++b)
    {
        int phys_block = Block_Table[by * max_num_blocks + b];
        int tokens_in_block = (b == num_blocks - 1) ? (context_len % 16) : 16;
        if (tokens_in_block == 0) tokens_in_block = 16;

        for (int t_start = 0; t_start < tokens_in_block; t_start += 8)
        {
            int t = t_start + token_idx;

            float dot = 0.0f;
            bool  valid_token = (t < tokens_in_block);

            if (valid_token && vec_idx * 8 < head_dim)
            {
                int64_t base_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim;
                int64_t token_addr = base_addr + (int64_t)t * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim;

                float4 k_vec_f4 = *reinterpret_cast<const float4*>(&K_Cache[token_addr + vec_idx * 8]);
                half*  k_vec = reinterpret_cast<half*>(&k_vec_f4);

#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    dot += __half2float(q_vec[i]) * __half2float(k_vec[i]);
                }
            }

            // Warp-level reduction across 16 threads (vec_idx) for this token
            unsigned int active = 0xffffffff;
#pragma unroll
            for (int offset = 8; offset > 0; offset /= 2)
            {
                dot += __shfl_down_sync(active, dot, offset, 16);
            }

            // Broadcast the dot product from lane 0 of the sub-warp to all its lanes
            float score = __shfl_sync(active, dot, 0, 16) * scale;
            if (!valid_token) score = -1e30f;

            // Online softmax update
            float m_prev = m_i;
            m_i = fmaxf(m_i, score);

            float alpha = valid_token ? expf(score - m_i) : 0.0f;
            float beta = expf(m_prev - m_i);

            l_i = l_i * beta + alpha;

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                acc_val[i] *= beta;
            }

            if (valid_token && vec_idx * 8 < head_dim)
            {
                int64_t base_addr = (int64_t)phys_block * 16 * num_kv_heads * head_dim;
                int64_t token_addr = base_addr + (int64_t)t * num_kv_heads * head_dim + (int64_t)kv_head_idx * head_dim;
                float4  v_vec_f4 = *reinterpret_cast<const float4*>(&V_Cache[token_addr + vec_idx * 8]);
                half*   v_vec = reinterpret_cast<half*>(&v_vec_f4);

#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    acc_val[i] += __half2float(v_vec[i]) * alpha;
                }
            }
        }
    }

    // Store localized reduction to shared memory
    if (vec_idx == 0)
    {
        smem_m[token_idx] = m_i;
        smem_l[token_idx] = l_i;
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        smem_acc[token_idx][vec_idx * 8 + i] = acc_val[i];
    }
    __syncthreads();

    // Now all 128 threads do an identical merged final state calculation
    float final_m = -1e30f;
    float final_l = 0.0f;

    for (int g = 0; g < 8; ++g)
    {
        float m_g = smem_m[g];
        final_m = fmaxf(final_m, m_g);
    }

    for (int g = 0; g < 8; ++g)
    {
        float m_g = smem_m[g];
        float l_g = smem_l[g];
        final_l += l_g * expf(m_g - final_m);
    }

    // Each thread takes one output element based on tx
    float final_acc = 0.0f;
    for (int g = 0; g < 8; ++g)
    {
        float m_g = smem_m[g];
        float acc_g = smem_acc[g][tx];  // tx happens to be vec_idx * 8 + i mapping 1-to-1
        final_acc += acc_g * expf(m_g - final_m);
    }

    if (tx < head_dim)
    {
        O[q_offset + tx] = __float2half(final_acc / final_l);
    }
}

// -----------------------------------------------------------------------------
// Host Dispatcher
// -----------------------------------------------------------------------------
void attention(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* kv_cache_block_table, int kv_head_num,
               int seq_len, int max_context_blocks, const int* context_lens)
{
    int   batch_size = q.shape()[0];
    int   num_heads = q.shape()[2];
    int   head_dim = q.shape()[3];
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (kv_cache_block_table == nullptr)
    {
        // Prefill: FlashAttention WMMA
        // Grid: M-tiles, Heads, Batch
        // Block: 32 threads (1 warp)
        dim3 block(32);
        dim3 grid((seq_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE, num_heads, batch_size);

        // Size: 256 half (512B) + 256 float (1024B) = ~1.5KB
        size_t smem_size = 2048;

        flash_attn_wmma_kernel<<<grid, block, smem_size, get_default_stream()>>>(
            (const half*)q.data(), (const half*)k.data(), (const half*)v.data(), (half*)output.data(), seq_len,
            num_heads, kv_head_num, head_dim, scale);
    }
    else
    {
        // Decode: PagedAttention
        dim3 block_decode(128);  // Matches head_dim 128
        dim3 grid_decode(num_heads, batch_size);

        int max_num_blocks = max_context_blocks;

        paged_attention_kernel<<<grid_decode, block_decode, 0, get_default_stream()>>>(
            (const half*)q.data(), (const half*)k.data(), (const half*)v.data(), (const int*)kv_cache_block_table,
            (half*)output.data(), batch_size, context_lens, num_heads, kv_head_num, head_dim, max_num_blocks, scale);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "PagedAttention Launch Failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

}  // namespace firefly::kernels
