#include "firefly/kernels.h"

#ifdef FIREFLY_USE_TORCH_SDPA

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ATen/Functions.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <tuple>

#include "firefly/cuda_dtype.cuh"

namespace firefly::kernels
{
namespace
{
at::ScalarType torch_dtype(DType dtype)
{
    if (dtype == DType::BF16) return at::kBFloat16;
    if (dtype == DType::F16) return at::kHalf;
    throw std::runtime_error("torch flash attention only supports float16/bfloat16");
}

at::Tensor wrap_cuda_tensor(void* data, DType dtype, at::IntArrayRef sizes, at::IntArrayRef strides)
{
    auto options = at::TensorOptions().device(at::kCUDA).dtype(torch_dtype(dtype));
    return at::from_blob(data, sizes, strides, options);
}

template <typename scalar_t>
__global__ void gather_paged_kv_kernel(const scalar_t* __restrict__ k_cache, const scalar_t* __restrict__ v_cache,
                                       scalar_t* __restrict__ k_out, scalar_t* __restrict__ v_out,
                                       const int* __restrict__ block_table, int batch, int total_len,
                                       int num_kv_heads, int head_dim, int max_context_blocks)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)batch * total_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % num_kv_heads;
    int t = (idx / (head_dim * num_kv_heads)) % total_len;
    int b = idx / ((int64_t)total_len * num_kv_heads * head_dim);

    int block_idx = t >> 4;
    int block_offset = t & 15;
    int phys_block = block_table[b * max_context_blocks + block_idx];
    int64_t src = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                  (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)h * head_dim + d;

    k_out[idx] = k_cache[src];
    v_out[idx] = v_cache[src];
}

template <typename scalar_t>
__global__ void copy_bhsd_to_bshd_slice_kernel(const scalar_t* __restrict__ src, scalar_t* __restrict__ dst, int batch,
                                               int dst_seq_len, int src_seq_len, int heads, int head_dim,
                                               int src_seq_offset)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)batch * dst_seq_len * heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int s = (idx / (head_dim * heads)) % dst_seq_len;
    int b = idx / ((int64_t)dst_seq_len * heads * head_dim);

    int src_s = src_seq_offset + s;
    int64_t src_idx = (int64_t)b * heads * src_seq_len * head_dim + (int64_t)h * src_seq_len * head_dim +
                      (int64_t)src_s * head_dim + d;
    dst[idx] = src[src_idx];
}

template <typename scalar_t>
__global__ void copy_bshd_to_bshd_offset_kernel(const scalar_t* __restrict__ src, scalar_t* __restrict__ dst,
                                                int batch, int src_seq_len, int dst_seq_len, int heads, int head_dim,
                                                int dst_seq_offset)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)batch * src_seq_len * heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int s = (idx / (head_dim * heads)) % src_seq_len;
    int b = idx / ((int64_t)src_seq_len * heads * head_dim);

    int dst_s = dst_seq_offset + s;
    int64_t dst_idx = (int64_t)b * dst_seq_len * heads * head_dim + (int64_t)dst_s * heads * head_dim +
                      (int64_t)h * head_dim + d;
    dst[dst_idx] = src[idx];
}

struct TorchPagedPrefillScratch
{
    Tensor  q;
    Tensor  k;
    Tensor  v;
    int64_t capacity = 0;
    int64_t q_capacity = 0;
};

TorchPagedPrefillScratch& torch_paged_prefill_scratch()
{
    static thread_local TorchPagedPrefillScratch scratch;
    return scratch;
}

void ensure_torch_paged_prefill_scratch(DType dtype, int batch, int total_len, int kv_heads, int head_dim)
{
    auto& scratch = torch_paged_prefill_scratch();
    int64_t needed = (int64_t)batch * total_len * kv_heads * head_dim;
    if (scratch.capacity < needed || scratch.k.dtype() != dtype)
    {
        scratch.k = Tensor({needed}, dtype, Device::CUDA);
        scratch.v = Tensor({needed}, dtype, Device::CUDA);
        scratch.capacity = needed;
    }
}

void ensure_torch_paged_q_scratch(DType dtype, int batch, int total_len, int heads, int head_dim)
{
    auto& scratch = torch_paged_prefill_scratch();
    int64_t needed = (int64_t)batch * total_len * heads * head_dim;
    if (scratch.q_capacity < needed || scratch.q.dtype() != dtype)
    {
        scratch.q = Tensor({needed}, dtype, Device::CUDA);
        scratch.q_capacity = needed;
    }
}
}  // namespace

bool torch_flash_attention_available() { return true; }

bool torch_flash_attention(Tensor& q, Tensor& k, Tensor& v, Tensor& output, int kv_head_num, int seq_len)
{
    if (q.dtype() != DType::BF16 && q.dtype() != DType::F16) return false;
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != output.dtype()) return false;
    if (q.shape().size() != 4 || k.shape().size() != 4 || v.shape().size() != 4) return false;
    if (q.shape()[0] != k.shape()[0] || q.shape()[0] != v.shape()[0]) return false;
    if (q.shape()[1] != seq_len || k.shape()[1] != v.shape()[1]) return false;
    if (k.shape()[2] != kv_head_num || v.shape()[2] != kv_head_num) return false;
    if (q.shape()[3] != k.shape()[3] || q.shape()[3] != v.shape()[3]) return false;
    if (output.numel() != q.numel()) return false;

    int64_t batch = q.shape()[0];
    int64_t key_len = k.shape()[1];
    int64_t heads = q.shape()[2];
    int64_t kv_heads = k.shape()[2];
    int64_t head_dim = q.shape()[3];
    if (heads % kv_heads != 0) return false;
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 192 && head_dim != 256) return false;

    try
    {
        int device = 0;
        cudaGetDevice(&device);
        auto firefly_stream = get_default_stream();
        auto torch_stream = c10::cuda::getStreamFromExternal(firefly_stream, device);
        auto previous_stream = c10::cuda::getCurrentCUDAStream(device);
        c10::cuda::setCurrentCUDAStream(torch_stream);

        at::Tensor tq = wrap_cuda_tensor(q.data(), q.dtype(), {batch, heads, seq_len, head_dim},
                                         {seq_len * heads * head_dim, head_dim, heads * head_dim, 1});
        at::Tensor tk = wrap_cuda_tensor(k.data(), k.dtype(), {batch, kv_heads, key_len, head_dim},
                                         {key_len * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1});
        at::Tensor tv = wrap_cuda_tensor(v.data(), v.dtype(), {batch, kv_heads, key_len, head_dim},
                                         {key_len * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1});

        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        auto result = at::_scaled_dot_product_flash_attention(tq, tk, tv, 0.0, true, false, scale);
        at::Tensor out = std::get<0>(result);

        int64_t elems = (int64_t)batch * seq_len * heads * head_dim;
        int threads = 256;
        int blocks = (int)((elems + threads - 1) / threads);
        if (q.dtype() == DType::BF16)
        {
            copy_bhsd_to_bshd_slice_kernel<__nv_bfloat16><<<blocks, threads, 0, firefly_stream>>>(
                static_cast<const __nv_bfloat16*>(out.data_ptr()),
                static_cast<__nv_bfloat16*>(output.data()), batch, seq_len, seq_len, heads, head_dim, 0);
        }
        else
        {
            copy_bhsd_to_bshd_slice_kernel<half><<<blocks, threads, 0, firefly_stream>>>(
                static_cast<const half*>(out.data_ptr()), static_cast<half*>(output.data()), batch, seq_len, seq_len,
                heads, head_dim, 0);
        }
        c10::cuda::setCurrentCUDAStream(previous_stream);
        return true;
    }
    catch (const std::exception& e)
    {
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "Torch flash attention failed once, falling back: " << e.what() << std::endl;
            warned = true;
        }
        return false;
    }
}

bool torch_flash_paged_prefill(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output, const int* block_table,
                               int kv_head_num, int seq_len, int max_context_blocks, int context_len)
{
    if (context_len <= 0) return false;
    if (seq_len <= 1) return false;
    if (q.dtype() != DType::BF16 && q.dtype() != DType::F16) return false;
    if (q.dtype() != k_cache.dtype() || q.dtype() != v_cache.dtype() || q.dtype() != output.dtype()) return false;
    if (q.shape().size() != 4 || k_cache.shape().size() != 4 || v_cache.shape().size() != 4) return false;
    if (k_cache.shape()[1] != 16 || v_cache.shape()[1] != 16) return false;
    if (k_cache.shape()[2] != kv_head_num || v_cache.shape()[2] != kv_head_num) return false;

    int64_t batch = q.shape()[0];
    int64_t heads = q.shape()[2];
    int64_t kv_heads = kv_head_num;
    int64_t head_dim = q.shape()[3];
    int total_len = context_len + seq_len;
    if (q.shape()[1] != seq_len) return false;
    if (k_cache.shape()[3] != head_dim || v_cache.shape()[3] != head_dim) return false;
    if (heads % kv_heads != 0) return false;
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 192 && head_dim != 256) return false;

    ensure_torch_paged_prefill_scratch(q.dtype(), batch, total_len, kv_heads, head_dim);
    ensure_torch_paged_q_scratch(q.dtype(), batch, total_len, heads, head_dim);
    auto& scratch = torch_paged_prefill_scratch();

    int64_t elems = (int64_t)batch * total_len * kv_heads * head_dim;
    int threads = 256;
    int blocks = (int)((elems + threads - 1) / threads);
    if (q.dtype() == DType::BF16)
    {
        gather_paged_kv_kernel<__nv_bfloat16><<<blocks, threads, 0, get_default_stream()>>>(
            static_cast<const __nv_bfloat16*>(k_cache.data()), static_cast<const __nv_bfloat16*>(v_cache.data()),
            static_cast<__nv_bfloat16*>(scratch.k.data()), static_cast<__nv_bfloat16*>(scratch.v.data()), block_table,
            batch, total_len, kv_heads, head_dim, max_context_blocks);
    }
    else
    {
        gather_paged_kv_kernel<half><<<blocks, threads, 0, get_default_stream()>>>(
            static_cast<const half*>(k_cache.data()), static_cast<const half*>(v_cache.data()),
            static_cast<half*>(scratch.k.data()), static_cast<half*>(scratch.v.data()), block_table, batch, total_len,
            kv_heads, head_dim, max_context_blocks);
    }

    cudaMemsetAsync(scratch.q.data(), 0, scratch.q.nbytes(), get_default_stream());
    int64_t q_elems = (int64_t)batch * seq_len * heads * head_dim;
    int q_blocks = (int)((q_elems + threads - 1) / threads);
    if (q.dtype() == DType::BF16)
    {
        copy_bshd_to_bshd_offset_kernel<__nv_bfloat16><<<q_blocks, threads, 0, get_default_stream()>>>(
            static_cast<const __nv_bfloat16*>(q.data()), static_cast<__nv_bfloat16*>(scratch.q.data()), batch,
            seq_len, total_len, heads, head_dim, context_len);
    }
    else
    {
        copy_bshd_to_bshd_offset_kernel<half><<<q_blocks, threads, 0, get_default_stream()>>>(
            static_cast<const half*>(q.data()), static_cast<half*>(scratch.q.data()), batch, seq_len, total_len, heads,
            head_dim, context_len);
    }

    Tensor q_view = Tensor::from_external(scratch.q.data(), {batch, total_len, heads, head_dim}, q.dtype(),
                                          Device::CUDA);
    Tensor k_view = Tensor::from_external(scratch.k.data(), {batch, total_len, kv_heads, head_dim}, q.dtype(),
                                          Device::CUDA);
    Tensor v_view = Tensor::from_external(scratch.v.data(), {batch, total_len, kv_heads, head_dim}, q.dtype(),
                                          Device::CUDA);

    try
    {
        int device = 0;
        cudaGetDevice(&device);
        auto firefly_stream = get_default_stream();
        auto torch_stream = c10::cuda::getStreamFromExternal(firefly_stream, device);
        auto previous_stream = c10::cuda::getCurrentCUDAStream(device);
        c10::cuda::setCurrentCUDAStream(torch_stream);

        at::Tensor tq = wrap_cuda_tensor(q_view.data(), q_view.dtype(), {batch, heads, total_len, head_dim},
                                         {total_len * heads * head_dim, head_dim, heads * head_dim, 1});
        at::Tensor tk = wrap_cuda_tensor(k_view.data(), k_view.dtype(), {batch, kv_heads, total_len, head_dim},
                                         {total_len * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1});
        at::Tensor tv = wrap_cuda_tensor(v_view.data(), v_view.dtype(), {batch, kv_heads, total_len, head_dim},
                                         {total_len * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1});

        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        auto result = at::_scaled_dot_product_flash_attention(tq, tk, tv, 0.0, true, false, scale);
        at::Tensor out = std::get<0>(result);

        int64_t out_elems = (int64_t)batch * seq_len * heads * head_dim;
        int out_blocks = (int)((out_elems + threads - 1) / threads);
        if (q.dtype() == DType::BF16)
        {
            copy_bhsd_to_bshd_slice_kernel<__nv_bfloat16><<<out_blocks, threads, 0, firefly_stream>>>(
                static_cast<const __nv_bfloat16*>(out.data_ptr()), static_cast<__nv_bfloat16*>(output.data()), batch,
                seq_len, total_len, heads, head_dim, context_len);
        }
        else
        {
            copy_bhsd_to_bshd_slice_kernel<half><<<out_blocks, threads, 0, firefly_stream>>>(
                static_cast<const half*>(out.data_ptr()), static_cast<half*>(output.data()), batch, seq_len, total_len,
                heads, head_dim, context_len);
        }
        c10::cuda::setCurrentCUDAStream(previous_stream);
        return true;
    }
    catch (const std::exception& e)
    {
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "Torch paged prefill flash attention failed once, falling back: " << e.what() << std::endl;
            warned = true;
        }
        return false;
    }
}

}  // namespace firefly::kernels

#else

namespace firefly::kernels
{
bool torch_flash_attention_available() { return false; }
bool torch_flash_attention(Tensor&, Tensor&, Tensor&, Tensor&, int, int) { return false; }
bool torch_flash_paged_prefill(Tensor&, Tensor&, Tensor&, Tensor&, const int*, int, int, int, int) { return false; }
}  // namespace firefly::kernels

#endif
