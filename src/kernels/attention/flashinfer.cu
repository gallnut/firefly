#include "firefly/kernels/attention/detail/flashinfer.h"
#include "firefly/kernels/attention/attention.h"

#ifdef FIREFLY_USE_FLASHINFER

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

namespace firefly::kernels
{
namespace
{
constexpr int page_size = 16;
constexpr size_t decode_float_workspace_bytes = 32ULL << 20;
constexpr size_t prefill_float_workspace_bytes = 128ULL << 20;
constexpr size_t int_workspace_bytes = 8ULL << 20;

struct FlashInferDecodeState
{
    void* float_workspace = nullptr;
    void* int_workspace = nullptr;
    void* host_int_workspace = nullptr;
    int* indices = nullptr;
    int* indptr = nullptr;
    int* last_page_len = nullptr;
    size_t indices_capacity = 0;
    int batch_size = 0;
    int num_qo_heads = 0;
    std::vector<int> host_indptr;
    std::vector<int> host_indices;
    std::vector<int> host_last_page_len;
    flashinfer::DecodePlanInfo plan;

    ~FlashInferDecodeState()
    {
        if (float_workspace) cudaFree(float_workspace);
        if (int_workspace) cudaFree(int_workspace);
        if (host_int_workspace) cudaFreeHost(host_int_workspace);
        if (indices) cudaFree(indices);
        if (indptr) cudaFree(indptr);
        if (last_page_len) cudaFree(last_page_len);
    }
};

FlashInferDecodeState& decode_state()
{
    static FlashInferDecodeState state;
    return state;
}


struct FlashInferPrefillState
{
    void* float_workspace = nullptr;
    void* int_workspace = nullptr;
    void* host_int_workspace = nullptr;
    int* q_indptr = nullptr;
    int* kv_indptr = nullptr;
    int* last_page_len = nullptr;
    int* indices = nullptr;
    int metadata_capacity = 0;
    int indices_capacity = 0;
    flashinfer::PrefillPlanInfo plan;
    int batch_size = -1;
    int seq_len = -1;
    int context_len = -1;

    ~FlashInferPrefillState()
    {
        if (float_workspace) cudaFree(float_workspace);
        if (int_workspace) cudaFree(int_workspace);
        if (host_int_workspace) cudaFreeHost(host_int_workspace);
        if (q_indptr) cudaFree(q_indptr);
        if (kv_indptr) cudaFree(kv_indptr);
        if (last_page_len) cudaFree(last_page_len);
        if (indices) cudaFree(indices);
    }
};

__global__ void flatten_block_table_kernel(const int* block_table, int* indices, int batch_size,
                                           int max_context_blocks, int pages)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * pages) return;
    int batch = index / pages;
    int page = index % pages;
    indices[index] = block_table[batch * max_context_blocks + page];
}

FlashInferPrefillState& prefill_state()
{
    static FlashInferPrefillState state;
    return state;
}

template <typename scalar_t>
bool launch_flashinfer(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output, int kv_head_num,
                       int head_dim, float scale, cudaStream_t stream)
{
    if (head_dim != 128) return false;
    auto& state = decode_state();
    if (state.batch_size != q.shape()[0] || state.num_qo_heads != q.shape()[2]) return false;

    using Params = flashinfer::BatchDecodeParams<scalar_t, scalar_t, scalar_t, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    flashinfer::paged_kv_t<scalar_t, int32_t> paged_kv(
        kv_head_num, page_size, head_dim, state.batch_size, flashinfer::QKVLayout::kNHD,
        static_cast<scalar_t*>(k_cache.data()), static_cast<scalar_t*>(v_cache.data()), state.indices,
        state.indptr, state.last_page_len);

    Params params;
    params.q = static_cast<scalar_t*>(q.data());
    params.paged_kv = paged_kv;
    params.o = static_cast<scalar_t*>(output.data());
    params.lse = nullptr;
    params.padded_batch_size = state.plan.padded_batch_size;
    params.num_qo_heads = state.num_qo_heads;
    params.q_stride_n = state.num_qo_heads * head_dim;
    params.q_stride_h = head_dim;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;
    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.request_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(state.int_workspace, state.plan.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.kv_chunk_size_ptr_offset);
    params.block_valid_mask = nullptr;
    params.partition_kv = state.plan.split_kv;

    scalar_t* tmp_v = state.plan.split_kv
                          ? flashinfer::GetPtrFromBaseOffset<scalar_t>(state.float_workspace, state.plan.v_offset)
                          : nullptr;
    float* tmp_s = state.plan.split_kv
                       ? flashinfer::GetPtrFromBaseOffset<float>(state.float_workspace, state.plan.s_offset)
                       : nullptr;
    return flashinfer::BatchDecodeWithPagedKVCacheDispatched<128, flashinfer::PosEncodingMode::kNone, Variant>(
               params, tmp_v, tmp_s, false, stream) == cudaSuccess;
}
}  // namespace

void prepare_attention_decode(const int* context_lens, const int* block_table, int batch_size,
                              int max_context_blocks, int num_qo_heads, const device::Context& context)
{
    auto& state = decode_state();
    std::vector<int> indptr(batch_size + 1, 0);
    state.host_last_page_len.resize(batch_size);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        int pages = (context_lens[batch] + 1 + page_size - 1) / page_size;
        indptr[batch + 1] = indptr[batch] + pages;
        state.host_last_page_len[batch] = context_lens[batch] % page_size + 1;
    }

    std::vector<int> indices(indptr.back());
    for (int batch = 0; batch < batch_size; ++batch)
    {
        std::copy_n(block_table + batch * max_context_blocks, indptr[batch + 1] - indptr[batch],
                    indices.data() + indptr[batch]);
    }

    if (!state.float_workspace)
    {
        cudaMalloc(&state.float_workspace, decode_float_workspace_bytes);
        cudaMalloc(&state.int_workspace, int_workspace_bytes);
        cudaMallocHost(&state.host_int_workspace, int_workspace_bytes);
    }
    const size_t required_indices = indices.size();
    if (state.indices_capacity < required_indices)
    {
        if (state.indices) cudaFree(state.indices);
        cudaMalloc(&state.indices, required_indices * sizeof(int));
        state.indices_capacity = required_indices;
    }
    if (state.batch_size != batch_size)
    {
        if (state.indptr) cudaFree(state.indptr);
        if (state.last_page_len) cudaFree(state.last_page_len);
        cudaMalloc(&state.indptr, (batch_size + 1) * sizeof(int));
        cudaMalloc(&state.last_page_len, batch_size * sizeof(int));
    }

    cudaStream_t stream = context.stream();
    if (state.host_indices != indices)
    {
        cudaMemcpyAsync(state.indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        state.host_indices = std::move(indices);
    }
    const bool topology_changed = state.host_indptr != indptr;
    if (topology_changed)
    {
        cudaMemcpyAsync(state.indptr, indptr.data(), indptr.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(state.last_page_len, state.host_last_page_len.data(), state.host_last_page_len.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    state.batch_size = batch_size;
    state.num_qo_heads = num_qo_heads;
    if (topology_changed)
    {
        using Params = flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
        using Variant = flashinfer::DefaultAttention<false, false, false, false>;
        auto estimator = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            2, 128, flashinfer::PosEncodingMode::kNone, Variant, Params>;
        cudaError_t status = flashinfer::DecodePlan<128, flashinfer::PosEncodingMode::kNone, Variant, Params>(
            state.float_workspace, decode_float_workspace_bytes, state.int_workspace, state.host_int_workspace,
            int_workspace_bytes, state.plan, indptr.data(), batch_size, num_qo_heads, page_size, false, stream,
            estimator);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("FlashInfer decode planning failed: ") + cudaGetErrorString(status));
        }
        state.host_indptr = std::move(indptr);
    }
}


bool attention_backend::launch_flashinfer_decode(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                                 int kv_head_num, int head_dim, float scale,
                                                 const device::Context& context)
{
    if (q.dtype() == DType::BF16)
        return launch_flashinfer<__nv_bfloat16>(q, k_cache, v_cache, output, kv_head_num, head_dim, scale,
                                                context.stream());
    if (q.dtype() == DType::F16)
        return launch_flashinfer<half>(q, k_cache, v_cache, output, kv_head_num, head_dim, scale, context.stream());
    return false;
}

template <typename scalar_t, uint32_t cta_tile_q>
bool dispatch_flashinfer_prefill(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                 const int* block_table, int kv_head_num, int seq_len, int context_len, float scale,
                                 cudaStream_t stream)
{
    constexpr int head_dim = 128;
    using Params = flashinfer::BatchPrefillPagedParams<scalar_t, scalar_t, scalar_t, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    auto& state = prefill_state();
    flashinfer::paged_kv_t<scalar_t, int32_t> paged_kv(
        kv_head_num, page_size, head_dim, q.shape()[0], flashinfer::QKVLayout::kNHD,
        static_cast<scalar_t*>(k_cache.data()), static_cast<scalar_t*>(v_cache.data()),
        state.indices, state.kv_indptr, state.last_page_len);

    Params params;
    params.q = static_cast<scalar_t*>(q.data());
    params.paged_kv = paged_kv;
    params.q_indptr = state.q_indptr;
    params.o = static_cast<scalar_t*>(output.data());
    params.lse = nullptr;
    params.num_qo_heads = q.shape()[2];
    params.group_size = flashinfer::uint_fastdiv(params.num_qo_heads / kv_head_num);
    params.q_stride_n = params.num_qo_heads * head_dim;
    params.q_stride_h = head_dim;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;
    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.request_indices_offset);
    params.qo_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.qo_tile_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(state.int_workspace, state.plan.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        state.int_workspace, state.plan.kv_chunk_size_ptr_offset);
    params.merge_indptr = state.plan.split_kv
                              ? flashinfer::GetPtrFromBaseOffset<int32_t>(state.int_workspace,
                                                                         state.plan.merge_indptr_offset)
                              : nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.max_total_num_rows = q.shape()[0] * seq_len;
    params.padded_batch_size = state.plan.padded_batch_size;
    params.partition_kv = state.plan.split_kv;

    scalar_t* tmp_v = state.plan.split_kv
                          ? flashinfer::GetPtrFromBaseOffset<scalar_t>(state.float_workspace, state.plan.v_offset)
                          : nullptr;
    float* tmp_s = state.plan.split_kv
                       ? flashinfer::GetPtrFromBaseOffset<float>(state.float_workspace, state.plan.s_offset)
                       : nullptr;
    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
               cta_tile_q, head_dim, head_dim, flashinfer::PosEncodingMode::kNone, false,
               flashinfer::MaskMode::kCausal, Variant>(params, tmp_v, tmp_s, false, stream) ==
               cudaSuccess;
}

bool attention_backend::launch_flashinfer_prefill(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                                  const int* block_table, int kv_head_num, int seq_len,
                                                  int max_context_blocks, int context_len, float scale,
                                                  const device::Context& context)
{
    if (q.shape()[3] != 128 || seq_len <= 1 || context_len < 0) return false;
    int batch_size = q.shape()[0];
    int total_len = context_len + seq_len;
    int pages = (total_len + page_size - 1) / page_size;
    if (pages > max_context_blocks) return false;

    auto& state = prefill_state();
    if (!state.float_workspace)
    {
        cudaMalloc(&state.float_workspace, prefill_float_workspace_bytes);
        cudaMalloc(&state.int_workspace, int_workspace_bytes);
        cudaMallocHost(&state.host_int_workspace, int_workspace_bytes);
    }
    if (state.metadata_capacity < batch_size)
    {
        if (state.q_indptr) cudaFree(state.q_indptr);
        if (state.kv_indptr) cudaFree(state.kv_indptr);
        if (state.last_page_len) cudaFree(state.last_page_len);
        cudaMalloc(&state.q_indptr, (batch_size + 1) * sizeof(int));
        cudaMalloc(&state.kv_indptr, (batch_size + 1) * sizeof(int));
        cudaMalloc(&state.last_page_len, batch_size * sizeof(int));
        state.metadata_capacity = batch_size;
    }
    if (state.indices_capacity < batch_size * pages)
    {
        if (state.indices) cudaFree(state.indices);
        cudaMalloc(&state.indices, batch_size * pages * sizeof(int));
        state.indices_capacity = batch_size * pages;
    }

    cudaStream_t stream = context.stream();
    int block_count = (batch_size * pages + 255) / 256;
    flatten_block_table_kernel<<<block_count, 256, 0, stream>>>(block_table, state.indices, batch_size,
                                                                max_context_blocks, pages);

    if (state.batch_size != batch_size || state.seq_len != seq_len || state.context_len != context_len)
    {
        std::vector<int> q_indptr_h(batch_size + 1);
        std::vector<int> kv_indptr_h(batch_size + 1);
        std::vector<int> last_page_len_h(batch_size, (total_len - 1) % page_size + 1);
        for (int batch = 0; batch <= batch_size; ++batch)
        {
            q_indptr_h[batch] = batch * seq_len;
            kv_indptr_h[batch] = batch * pages;
        }
        cudaMemcpyAsync(state.q_indptr, q_indptr_h.data(), q_indptr_h.size() * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(state.kv_indptr, kv_indptr_h.data(), kv_indptr_h.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(state.last_page_len, last_page_len_h.data(), last_page_len_h.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaError_t status = flashinfer::PrefillPlan<int32_t>(
            state.float_workspace, prefill_float_workspace_bytes, state.int_workspace, state.host_int_workspace,
            int_workspace_bytes, state.plan, q_indptr_h.data(), kv_indptr_h.data(), batch_size * seq_len, batch_size,
            q.shape()[2], kv_head_num,
            128, 128, page_size, false, 2, -1, -1, false, 0, stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("FlashInfer prefill planning failed: ") + cudaGetErrorString(status));
        state.batch_size = batch_size;
        state.seq_len = seq_len;
        state.context_len = context_len;
    }

    if (q.dtype() == DType::BF16)
    {
        switch (state.plan.cta_tile_q)
        {
            case 128: return dispatch_flashinfer_prefill<__nv_bfloat16, 128>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
            case 64: return dispatch_flashinfer_prefill<__nv_bfloat16, 64>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
            case 32: return dispatch_flashinfer_prefill<__nv_bfloat16, 32>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
            case 16: return dispatch_flashinfer_prefill<__nv_bfloat16, 16>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
        }
    }
    return false;
}
}  // namespace firefly::kernels

#else

namespace firefly::kernels
{
void prepare_attention_decode(const int*, const int*, int, int, int, const device::Context&) {}
bool attention_backend::launch_flashinfer_decode(Tensor&, Tensor&, Tensor&, Tensor&, int, int, float,
                                                 const device::Context&)
{
    return false;
}
bool attention_backend::launch_flashinfer_prefill(Tensor&, Tensor&, Tensor&, Tensor&, const int*, int, int, int, int,
                                                  float, const device::Context&)
{
    return false;
}
}  // namespace firefly::kernels

#endif
