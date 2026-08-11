#include "firefly/kernels/attention/detail/flashinfer.h"
#include "firefly/kernels/attention/attention.h"

#ifdef FIREFLY_USE_FLASHINFER

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
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
    int num_kv_heads = 0;
    int head_dim = 0;
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
    int head_dim = -1;
    int num_qo_heads = -1;
    int num_kv_heads = -1;
    int total_rows = -1;
    int max_context_blocks = -1;
    void* dequantized_k = nullptr;
    void* dequantized_v = nullptr;
    size_t dequantized_capacity = 0;

    ~FlashInferPrefillState()
    {
        if (float_workspace) cudaFree(float_workspace);
        if (int_workspace) cudaFree(int_workspace);
        if (host_int_workspace) cudaFreeHost(host_int_workspace);
        if (q_indptr) cudaFree(q_indptr);
        if (kv_indptr) cudaFree(kv_indptr);
        if (last_page_len) cudaFree(last_page_len);
        if (indices) cudaFree(indices);
        if (dequantized_k) cudaFree(dequantized_k);
        if (dequantized_v) cudaFree(dequantized_v);
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

__global__ void flatten_ragged_block_table_kernel(const int* __restrict__ block_table,
                                                  int* __restrict__ indices,
                                                  const int* __restrict__ kv_indptr, int batch_size,
                                                  int max_context_blocks)
{
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int start = kv_indptr[row];
    int pages = kv_indptr[row + 1] - kv_indptr[row];
    for (int page = threadIdx.x; page < pages; page += blockDim.x)
        indices[start + page] = block_table[(int64_t)row * max_context_blocks + page];
}

FlashInferPrefillState& prefill_state()
{
    static FlashInferPrefillState state;
    return state;
}

template <typename scalar_t>
__global__ void dequantize_paged_kv_kernel(const int8_t* __restrict__ k_cache,
                                           const int8_t* __restrict__ v_cache,
                                           const float* __restrict__ scales,
                                           const int* __restrict__ block_table,
                                           scalar_t* __restrict__ k,
                                           scalar_t* __restrict__ v,
                                           int total_len, int num_kv_heads, int head_dim,
                                           int max_context_blocks)
{
    int64_t batch_offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t batch_elements = static_cast<int64_t>(total_len) * num_kv_heads * head_dim;
    if (batch_offset >= batch_elements) return;

    int batch = blockIdx.y;
    int64_t index = static_cast<int64_t>(batch) * batch_elements + batch_offset;
    int token = batch_offset / (num_kv_heads * head_dim);
    int token_offset = batch_offset - static_cast<int64_t>(token) * num_kv_heads * head_dim;
    int kv_head = token_offset / head_dim;
    int physical_block = block_table[batch * max_context_blocks + (token >> 4)];
    int block_offset = token & 15;
    int64_t cache_offset = static_cast<int64_t>(physical_block) * page_size * num_kv_heads * head_dim +
                           static_cast<int64_t>(block_offset) * num_kv_heads * head_dim + token_offset;
    int64_t scale_offset =
        (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head) * page_size * 2 + block_offset * 2;
    k[index] = static_cast<scalar_t>(static_cast<float>(k_cache[cache_offset]) * scales[scale_offset]);
    v[index] = static_cast<scalar_t>(static_cast<float>(v_cache[cache_offset]) * scales[scale_offset + 1]);
}

template <typename scalar_t, int head_dim>
cudaError_t launch_sm89_contiguous_prefill(Tensor& q, scalar_t* k, scalar_t* v, Tensor& output,
                                           int kv_head_num, int kv_len, float scale,
                                           std::string_view config, cudaStream_t stream)
{
    using Params = flashinfer::SinglePrefillParams<scalar_t, scalar_t, scalar_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;

    auto launch = [&]<uint32_t cta_tile_q, uint32_t num_mma_kv, uint32_t forced_num_warps_q = 0,
                     uint32_t forced_num_warps_kv = 0>() -> cudaError_t
    {
        constexpr uint32_t default_num_warps_q = cta_tile_q > 16 ? 4 : 1;
        constexpr uint32_t num_warps_q = forced_num_warps_q == 0 ? default_num_warps_q : forced_num_warps_q;
        constexpr uint32_t default_num_warps_kv = 4 / num_warps_q;
        constexpr uint32_t num_warps_kv =
            forced_num_warps_kv == 0 ? default_num_warps_kv : forced_num_warps_kv;
        constexpr uint32_t num_mma_q = cta_tile_q / (num_warps_q * 16);
        using Traits = flashinfer::KernelTraits<
            flashinfer::MaskMode::kCausal, cta_tile_q, num_mma_q, num_mma_kv,
            head_dim / 16, head_dim / 16, num_warps_q, num_warps_kv,
            flashinfer::PosEncodingMode::kNone, scalar_t, scalar_t, scalar_t, float, int32_t, Variant>;
        static_assert(!Traits::IsInvalid());

        auto kernel = flashinfer::SinglePrefillWithKVCacheKernel<Traits, Params>;
        size_t smem_size = sizeof(typename Traits::SharedStorage);
        int device = 0;
        int max_smem = 0;
        cudaError_t status = cudaGetDevice(&device);
        if (status != cudaSuccess) return status;
        status = cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (status != cudaSuccess) return status;
        if (smem_size > static_cast<size_t>(max_smem)) return cudaErrorInvalidConfiguration;
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem_size));
        if (status != cudaSuccess) return status;

        int batch_size = q.shape()[0];
        int qo_len = q.shape()[1];
        int num_qo_heads = q.shape()[2];
        int64_t q_batch_stride = static_cast<int64_t>(qo_len) * num_qo_heads * head_dim;
        int64_t kv_batch_stride = static_cast<int64_t>(kv_len) * kv_head_num * head_dim;
        for (int batch = 0; batch < batch_size; ++batch)
        {
            Params params;
            params.q = static_cast<scalar_t*>(q.data()) + batch * q_batch_stride;
            params.k = k + batch * kv_batch_stride;
            params.v = v + batch * kv_batch_stride;
            params.o = static_cast<scalar_t*>(output.data()) + batch * q_batch_stride;
            params.maybe_custom_mask = nullptr;
            params.lse = nullptr;
            params.maybe_alibi_slopes = nullptr;
            params.num_qo_heads = num_qo_heads;
            params.num_kv_heads = kv_head_num;
            params.group_size = flashinfer::uint_fastdiv(num_qo_heads / kv_head_num);
            params.qo_len = qo_len;
            params.kv_len = kv_len;
            params.q_stride_n = num_qo_heads * head_dim;
            params.q_stride_h = head_dim;
            params.k_stride_n = kv_head_num * head_dim;
            params.k_stride_h = head_dim;
            params.v_stride_n = kv_head_num * head_dim;
            params.v_stride_h = head_dim;
            params.head_dim = head_dim;
            params.window_left = -1;
            params.logits_soft_cap = 0.0f;
            params.sm_scale = scale;
            params.rope_rcp_scale = 1.0f;
            params.rope_rcp_theta = 1.0f;
            params.partition_kv = false;

            void* args[] = {&params};
            dim3 grid(flashinfer::ceil_div(qo_len * (num_qo_heads / kv_head_num), cta_tile_q), 1,
                      kv_head_num);
            dim3 block(32, num_warps_q, num_warps_kv);
            status = cudaLaunchKernel(reinterpret_cast<void*>(kernel), grid, block, args, smem_size, stream);
            if (status != cudaSuccess) return status;
        }
        return cudaSuccess;
    };

    if (config == "16x1") return launch.template operator()<16, 1>();
    if (config == "16x2") return launch.template operator()<16, 2>();
    if (config == "64x1") return launch.template operator()<64, 1>();
    if (config == "64x2") return launch.template operator()<64, 2>();
    if (config == "64x3") return launch.template operator()<64, 3>();
    if (config == "64x4") return launch.template operator()<64, 4>();
    if (config == "64w2x1") return launch.template operator()<64, 1, 2>();
    if (config == "64w2x2") return launch.template operator()<64, 2, 2>();
    if (config == "64w2x3") return launch.template operator()<64, 3, 2>();
    if (config == "128x2") return launch.template operator()<128, 2>();
    if (config == "128x3") return launch.template operator()<128, 3>();
    if (config == "128x4") return launch.template operator()<128, 4>();
    return cudaErrorInvalidValue;
}

template <typename scalar_t, int head_dim>
bool launch_contiguous_prefill(Tensor& q, scalar_t* k, scalar_t* v, Tensor& output, int kv_head_num,
                               int kv_len, float scale, cudaStream_t stream)
{
    using Params = flashinfer::SinglePrefillParams<scalar_t, scalar_t, scalar_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    int batch_size = q.shape()[0];
    int qo_len = q.shape()[1];
    int num_qo_heads = q.shape()[2];
    int64_t q_batch_stride = static_cast<int64_t>(qo_len) * num_qo_heads * head_dim;
    int64_t kv_batch_stride = static_cast<int64_t>(kv_len) * kv_head_num * head_dim;

    if constexpr (head_dim == 128)
    {
        const char* custom_config = std::getenv("FIREFLY_FLASHINFER_PREFILL_CONFIG");
        if (custom_config != nullptr && std::string_view(custom_config) != "auto")
        {
            return launch_sm89_contiguous_prefill<scalar_t, head_dim>(q, k, v, output, kv_head_num, kv_len, scale,
                                                                      custom_config, stream) == cudaSuccess;
        }
    }

    for (int batch = 0; batch < batch_size; ++batch)
    {
        Params params;
        params.q = static_cast<scalar_t*>(q.data()) + batch * q_batch_stride;
        params.k = k + batch * kv_batch_stride;
        params.v = v + batch * kv_batch_stride;
        params.o = static_cast<scalar_t*>(output.data()) + batch * q_batch_stride;
        params.maybe_custom_mask = nullptr;
        params.lse = nullptr;
        params.maybe_alibi_slopes = nullptr;
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = kv_head_num;
        params.group_size = flashinfer::uint_fastdiv(num_qo_heads / kv_head_num);
        params.qo_len = qo_len;
        params.kv_len = kv_len;
        params.q_stride_n = num_qo_heads * head_dim;
        params.q_stride_h = head_dim;
        params.k_stride_n = kv_head_num * head_dim;
        params.k_stride_h = head_dim;
        params.v_stride_n = kv_head_num * head_dim;
        params.v_stride_h = head_dim;
        params.head_dim = head_dim;
        params.window_left = -1;
        params.logits_soft_cap = 0.0f;
        params.sm_scale = scale;
        params.rope_rcp_scale = 1.0f;
        params.rope_rcp_theta = 1.0f;
        params.partition_kv = false;

        cudaError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<
            head_dim, head_dim, flashinfer::PosEncodingMode::kNone, false,
            flashinfer::MaskMode::kCausal, Variant>(params, nullptr, stream);
        if (status != cudaSuccess) return false;
    }
    return true;
}

template <typename scalar_t, uint32_t head_dim>
bool launch_flashinfer(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output, int kv_head_num,
                       float scale, cudaStream_t stream)
{
    auto& state = decode_state();
    if (state.batch_size != q.shape()[0] || state.num_qo_heads != q.shape()[2] ||
        state.num_kv_heads != kv_head_num || state.head_dim != head_dim)
        return false;

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
    params.block_valid_mask = state.plan.split_kv
                                  ? flashinfer::GetPtrFromBaseOffset<bool>(
                                        state.int_workspace, state.plan.block_valid_mask_offset)
                                  : nullptr;
    params.partition_kv = state.plan.split_kv;

    scalar_t* tmp_v = state.plan.split_kv
                          ? flashinfer::GetPtrFromBaseOffset<scalar_t>(state.float_workspace, state.plan.v_offset)
                          : nullptr;
    float* tmp_s = state.plan.split_kv
                       ? flashinfer::GetPtrFromBaseOffset<float>(state.float_workspace, state.plan.s_offset)
                       : nullptr;
    return flashinfer::BatchDecodeWithPagedKVCacheDispatched<head_dim, flashinfer::PosEncodingMode::kNone, Variant>(
               params, tmp_v, tmp_s, false, stream) == cudaSuccess;
}

template <uint32_t group_size, uint32_t head_dim>
cudaError_t plan_flashinfer_decode(FlashInferDecodeState& state, const int* indptr, int batch_size,
                                   int num_qo_heads, cudaStream_t stream)
{
    using Params = flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    auto estimator = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
        group_size, head_dim, flashinfer::PosEncodingMode::kNone, Variant, Params>;
    return flashinfer::DecodePlan<head_dim, flashinfer::PosEncodingMode::kNone, Variant, Params>(
        state.float_workspace, decode_float_workspace_bytes, state.int_workspace, state.host_int_workspace,
        int_workspace_bytes, state.plan, const_cast<int*>(indptr), batch_size, num_qo_heads, page_size, false, stream,
        estimator);
}
}  // namespace

void prepare_attention_decode(const int* context_lens, const int* block_table, int batch_size,
                              int max_context_blocks, int num_qo_heads, int num_kv_heads, int head_dim,
                              const device::Context& context)
{
    if ((head_dim != 128 && head_dim != 256) || num_kv_heads <= 0 || num_qo_heads % num_kv_heads != 0) return;
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
    const bool plan_changed = state.host_indptr != indptr || state.num_qo_heads != num_qo_heads ||
                              state.num_kv_heads != num_kv_heads || state.head_dim != head_dim;
    if (state.host_indptr != indptr)
    {
        cudaMemcpyAsync(state.indptr, indptr.data(), indptr.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(state.last_page_len, state.host_last_page_len.data(), state.host_last_page_len.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    state.batch_size = batch_size;
    state.num_qo_heads = num_qo_heads;
    state.num_kv_heads = num_kv_heads;
    state.head_dim = head_dim;
    if (plan_changed)
    {
        const int group_size = num_qo_heads / num_kv_heads;
        auto dispatch = [&]<uint32_t dimension>()
        {
            switch (group_size)
            {
                case 1: return plan_flashinfer_decode<1, dimension>(state, indptr.data(), batch_size, num_qo_heads, stream);
                case 2: return plan_flashinfer_decode<2, dimension>(state, indptr.data(), batch_size, num_qo_heads, stream);
                case 3: return plan_flashinfer_decode<3, dimension>(state, indptr.data(), batch_size, num_qo_heads, stream);
                case 4: return plan_flashinfer_decode<4, dimension>(state, indptr.data(), batch_size, num_qo_heads, stream);
                case 8: return plan_flashinfer_decode<8, dimension>(state, indptr.data(), batch_size, num_qo_heads, stream);
                default: return cudaErrorInvalidValue;
            }
        };
        cudaError_t status = head_dim == 128 ? dispatch.template operator()<128>()
                                             : dispatch.template operator()<256>();
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
        return head_dim == 128
                   ? launch_flashinfer<__nv_bfloat16, 128>(q, k_cache, v_cache, output, kv_head_num, scale,
                                                          context.stream())
                   : head_dim == 256
                         ? launch_flashinfer<__nv_bfloat16, 256>(q, k_cache, v_cache, output, kv_head_num, scale,
                                                                context.stream())
                         : false;
    if (q.dtype() == DType::F16)
        return head_dim == 128
                   ? launch_flashinfer<half, 128>(q, k_cache, v_cache, output, kv_head_num, scale, context.stream())
                   : head_dim == 256
                         ? launch_flashinfer<half, 256>(q, k_cache, v_cache, output, kv_head_num, scale,
                                                       context.stream())
                         : false;
    return false;
}

bool attention_backend::launch_flashinfer_contiguous_prefill(Tensor& q, Tensor& k, Tensor& v, Tensor& output,
                                                             int kv_head_num, float scale,
                                                             const device::Context& context)
{
    const int head_dim = q.shape()[3];
    if ((head_dim != 128 && head_dim != 256) || q.shape()[1] <= 1 || k.shape()[1] != q.shape()[1]) return false;
    if (q.dtype() == DType::BF16)
        return head_dim == 128
                   ? launch_contiguous_prefill<__nv_bfloat16, 128>(q, static_cast<__nv_bfloat16*>(k.data()),
                                                                  static_cast<__nv_bfloat16*>(v.data()), output,
                                                                  kv_head_num, q.shape()[1], scale, context.stream())
                   : launch_contiguous_prefill<__nv_bfloat16, 256>(q, static_cast<__nv_bfloat16*>(k.data()),
                                                                  static_cast<__nv_bfloat16*>(v.data()), output,
                                                                  kv_head_num, q.shape()[1], scale, context.stream());
    if (q.dtype() == DType::F16)
        return head_dim == 128
                   ? launch_contiguous_prefill<half, 128>(q, static_cast<half*>(k.data()),
                                                          static_cast<half*>(v.data()), output, kv_head_num,
                                                          q.shape()[1], scale, context.stream())
                   : launch_contiguous_prefill<half, 256>(q, static_cast<half*>(k.data()),
                                                          static_cast<half*>(v.data()), output, kv_head_num,
                                                          q.shape()[1], scale, context.stream());
    return false;
}

template <typename scalar_t>
bool launch_quantized_prefill(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& scales,
                              const int* block_table, Tensor& output, int kv_head_num,
                              int max_context_blocks, int context_len, float scale, cudaStream_t stream)
{
    constexpr int head_dim = 128;
    int batch_size = q.shape()[0];
    int total_len = context_len + q.shape()[1];
    int64_t element_count = static_cast<int64_t>(batch_size) * total_len * kv_head_num * head_dim;
    size_t required_bytes = element_count * sizeof(scalar_t);
    auto& state = prefill_state();
    if (state.dequantized_capacity < required_bytes)
    {
        if (state.dequantized_k) cudaFree(state.dequantized_k);
        if (state.dequantized_v) cudaFree(state.dequantized_v);
        cudaMalloc(&state.dequantized_k, required_bytes);
        cudaMalloc(&state.dequantized_v, required_bytes);
        state.dequantized_capacity = required_bytes;
    }

    int threads = 256;
    int64_t batch_elements = static_cast<int64_t>(total_len) * kv_head_num * head_dim;
    int blocks = static_cast<int>((batch_elements + threads - 1) / threads);
    dequantize_paged_kv_kernel<scalar_t><<<dim3(blocks, batch_size), threads, 0, stream>>>(
        static_cast<const int8_t*>(k_cache.data()), static_cast<const int8_t*>(v_cache.data()),
        static_cast<const float*>(scales.data()), block_table,
        static_cast<scalar_t*>(state.dequantized_k), static_cast<scalar_t*>(state.dequantized_v),
        total_len, kv_head_num, head_dim, max_context_blocks);
    return launch_contiguous_prefill<scalar_t, 128>(q, static_cast<scalar_t*>(state.dequantized_k),
                                                    static_cast<scalar_t*>(state.dequantized_v), output, kv_head_num,
                                                    total_len, scale, stream);
}

bool attention_backend::launch_flashinfer_quantized_prefill(
    Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& scales, const int* block_table, Tensor& output,
    int kv_head_num, int max_context_blocks, int context_length, float scale, const device::Context& context)
{
    if (q.shape()[3] != 128 || q.shape()[1] <= 1 || context_length < 0) return false;
    if (q.dtype() == DType::BF16)
        return launch_quantized_prefill<__nv_bfloat16>(q, k_cache, v_cache, scales, block_table, output,
                                                       kv_head_num, max_context_blocks, context_length, scale,
                                                       context.stream());
    if (q.dtype() == DType::F16)
        return launch_quantized_prefill<half>(q, k_cache, v_cache, scales, block_table, output,
                                              kv_head_num, max_context_blocks, context_length, scale,
                                              context.stream());
    return false;
}

template <typename scalar_t, uint32_t head_dim, uint32_t cta_tile_q>
bool dispatch_flashinfer_prefill(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                 const int* block_table, int kv_head_num, int seq_len, int context_len, float scale,
                                 cudaStream_t stream)
{
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
    params.block_valid_mask = state.plan.split_kv
                                  ? flashinfer::GetPtrFromBaseOffset<bool>(
                                        state.int_workspace, state.plan.block_valid_mask_offset)
                                  : nullptr;
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

void attention_backend::prepare_flashinfer_prefill(const int* block_table, int batch_size, int sequence_length,
                                                   int context_length, int max_context_blocks, int num_qo_heads,
                                                   int num_kv_heads, int head_dim, cudaStream_t stream)
{
    if ((head_dim != 128 && head_dim != 256) || sequence_length <= 1 || context_length < 0) return;
    const int total_len = context_length + sequence_length;
    const int pages = (total_len + page_size - 1) / page_size;
    if (pages > max_context_blocks) return;

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
    const int indices_capacity = batch_size * max_context_blocks;
    if (state.indices_capacity < indices_capacity)
    {
        if (state.indices) cudaFree(state.indices);
        cudaMalloc(&state.indices, indices_capacity * sizeof(int));
        state.indices_capacity = indices_capacity;
    }

    if (state.batch_size != batch_size || state.seq_len != sequence_length ||
        state.context_len != context_length || state.head_dim != head_dim)
    {
        std::vector<int> q_indptr_h(batch_size + 1);
        std::vector<int> kv_indptr_h(batch_size + 1);
        std::vector<int> last_page_len_h(batch_size, (total_len - 1) % page_size + 1);
        for (int batch = 0; batch <= batch_size; ++batch)
        {
            q_indptr_h[batch] = batch * sequence_length;
            kv_indptr_h[batch] = batch * pages;
        }
        cudaMemcpyAsync(state.q_indptr, q_indptr_h.data(), q_indptr_h.size() * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(state.kv_indptr, kv_indptr_h.data(), kv_indptr_h.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(state.last_page_len, last_page_len_h.data(), last_page_len_h.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        const cudaError_t status = flashinfer::PrefillPlan<int32_t>(
            state.float_workspace, prefill_float_workspace_bytes, state.int_workspace, state.host_int_workspace,
            int_workspace_bytes, state.plan, q_indptr_h.data(), kv_indptr_h.data(), batch_size * sequence_length,
            batch_size, num_qo_heads, num_kv_heads, head_dim, head_dim, page_size, false, 2, -1, -1, false, 0, stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("FlashInfer prefill planning failed: ") + cudaGetErrorString(status));
        state.batch_size = batch_size;
        state.seq_len = sequence_length;
        state.context_len = context_length;
        state.head_dim = head_dim;
    }
}

bool attention_backend::launch_flashinfer_prefill(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                                  const int* block_table, int kv_head_num, int seq_len,
                                                  int max_context_blocks, int context_len, float scale,
                                                  const device::Context& context)
{
    const int head_dim = q.shape()[3];
    if ((head_dim != 128 && head_dim != 256) || seq_len <= 1 || context_len < 0) return false;
    int batch_size = q.shape()[0];
    int total_len = context_len + seq_len;
    int pages = (total_len + page_size - 1) / page_size;
    if (pages > max_context_blocks) return false;

    auto& state = prefill_state();
    if (state.batch_size != batch_size || state.seq_len != seq_len || state.context_len != context_len ||
        state.head_dim != head_dim)
        return false;

    cudaStream_t stream = context.stream();
    int block_count = (batch_size * pages + 255) / 256;
    flatten_block_table_kernel<<<block_count, 256, 0, stream>>>(block_table, state.indices, batch_size,
                                                                max_context_blocks, pages);

    if (q.dtype() == DType::BF16)
    {
        auto dispatch = [&]<uint32_t dimension>() -> bool
        {
            switch (state.plan.cta_tile_q)
            {
                case 128: return dispatch_flashinfer_prefill<__nv_bfloat16, dimension, 128>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
                case 64: return dispatch_flashinfer_prefill<__nv_bfloat16, dimension, 64>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
                case 32: return dispatch_flashinfer_prefill<__nv_bfloat16, dimension, 32>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
                case 16: return dispatch_flashinfer_prefill<__nv_bfloat16, dimension, 16>(q, k_cache, v_cache, output, block_table, kv_head_num, seq_len, context_len, scale, stream);
            }
            return false;
        };
        return head_dim == 128 ? dispatch.template operator()<128>() : dispatch.template operator()<256>();
    }
    return false;
}

template <typename scalar_t, uint32_t head_dim, uint32_t cta_tile_q>
bool dispatch_flashinfer_prefill_ragged(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                        int kv_head_num, float scale, cudaStream_t stream)
{
    using Params = flashinfer::BatchPrefillPagedParams<scalar_t, scalar_t, scalar_t, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    auto& state = prefill_state();
    flashinfer::paged_kv_t<scalar_t, int32_t> paged_kv(
        kv_head_num, page_size, head_dim, state.batch_size, flashinfer::QKVLayout::kNHD,
        static_cast<scalar_t*>(k_cache.data()), static_cast<scalar_t*>(v_cache.data()), state.indices,
        state.kv_indptr, state.last_page_len);

    Params params;
    params.q = static_cast<scalar_t*>(q.data());
    params.paged_kv = paged_kv;
    params.q_indptr = state.q_indptr;
    params.o = static_cast<scalar_t*>(output.data());
    params.lse = nullptr;
    params.num_qo_heads = state.num_qo_heads;
    params.group_size = flashinfer::uint_fastdiv(state.num_qo_heads / kv_head_num);
    params.q_stride_n = state.num_qo_heads * head_dim;
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
    params.block_valid_mask = state.plan.split_kv
                                  ? flashinfer::GetPtrFromBaseOffset<bool>(
                                        state.int_workspace, state.plan.block_valid_mask_offset)
                                  : nullptr;
    params.total_num_rows = nullptr;
    params.max_total_num_rows = state.total_rows;
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
               flashinfer::MaskMode::kCausal,
               Variant>(params, tmp_v, tmp_s, false, stream) == cudaSuccess;
}

void attention_backend::prepare_flashinfer_prefill_ragged(const std::vector<int>& q_indptr,
                                                          const std::vector<int>& kv_indptr,
                                                          const std::vector<int>& last_page_len,
                                                          int max_context_blocks, int num_qo_heads, int num_kv_heads,
                                                          int head_dim, cudaStream_t stream)
{
    if ((head_dim != 128 && head_dim != 256) || q_indptr.size() < 2) return;
    const int batch_size = static_cast<int>(q_indptr.size()) - 1;
    const int total_rows = q_indptr.back();
    const int total_pages = kv_indptr.back();
    if (batch_size <= 0 || total_rows <= 0 || total_pages <= 0) return;

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
    if (state.indices_capacity < total_pages)
    {
        if (state.indices) cudaFree(state.indices);
        cudaMalloc(&state.indices, total_pages * sizeof(int));
        state.indices_capacity = total_pages;
    }

    std::vector<int> q_indptr_h = q_indptr;
    std::vector<int> kv_indptr_h = kv_indptr;
    std::vector<int> last_page_len_h = last_page_len;
    cudaMemcpyAsync(state.q_indptr, q_indptr_h.data(), q_indptr_h.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(state.kv_indptr, kv_indptr_h.data(), kv_indptr_h.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(state.last_page_len, last_page_len_h.data(), last_page_len_h.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    const cudaError_t status = flashinfer::PrefillPlan<int32_t>(
        state.float_workspace, prefill_float_workspace_bytes, state.int_workspace, state.host_int_workspace,
        int_workspace_bytes, state.plan, q_indptr_h.data(), kv_indptr_h.data(), total_rows, batch_size, num_qo_heads,
        num_kv_heads, head_dim, head_dim, page_size, false, 2, -1, -1, false, 0, stream);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("FlashInfer ragged prefill planning failed: ") +
                                 cudaGetErrorString(status));
    state.batch_size = batch_size;
    state.seq_len = -1;
    state.context_len = -1;
    state.head_dim = head_dim;
    state.num_qo_heads = num_qo_heads;
    state.num_kv_heads = num_kv_heads;
    state.total_rows = total_rows;
    state.max_context_blocks = max_context_blocks;
}

bool attention_backend::launch_flashinfer_prefill_ragged(Tensor& q, Tensor& k_cache, Tensor& v_cache, Tensor& output,
                                                         const int* block_table, int kv_head_num,
                                                         int max_context_blocks, float scale,
                                                         const device::Context& context)
{
    const int head_dim = q.shape()[3];
    const int num_qo_heads = q.shape()[2];
    auto& state = prefill_state();
    if (state.batch_size <= 0 || state.total_rows <= 0 || state.seq_len != -1 || state.head_dim != head_dim ||
        state.num_qo_heads != num_qo_heads || state.num_kv_heads != kv_head_num ||
        state.max_context_blocks != max_context_blocks)
        return false;

    cudaStream_t stream = context.stream();
    flatten_ragged_block_table_kernel<<<state.batch_size, 256, 0, stream>>>(
        block_table, state.indices, state.kv_indptr, state.batch_size, max_context_blocks);

    auto dispatch = [&]<typename scalar_t, uint32_t dimension>() -> bool
    {
        switch (state.plan.cta_tile_q)
        {
            case 128:
                return dispatch_flashinfer_prefill_ragged<scalar_t, dimension, 128>(
                    q, k_cache, v_cache, output, kv_head_num, scale, stream);
            case 64:
                return dispatch_flashinfer_prefill_ragged<scalar_t, dimension, 64>(
                    q, k_cache, v_cache, output, kv_head_num, scale, stream);
            case 32:
                return dispatch_flashinfer_prefill_ragged<scalar_t, dimension, 32>(
                    q, k_cache, v_cache, output, kv_head_num, scale, stream);
            case 16:
                return dispatch_flashinfer_prefill_ragged<scalar_t, dimension, 16>(
                    q, k_cache, v_cache, output, kv_head_num, scale, stream);
        }
        return false;
    };
    if (q.dtype() == DType::BF16)
        return head_dim == 128 ? dispatch.template operator()<__nv_bfloat16, 128>()
                               : head_dim == 256 ? dispatch.template operator()<__nv_bfloat16, 256>() : false;
    if (q.dtype() == DType::F16)
        return head_dim == 128 ? dispatch.template operator()<half, 128>()
                               : head_dim == 256 ? dispatch.template operator()<half, 256>() : false;
    return false;
}
}  // namespace firefly::kernels

#else

namespace firefly::kernels
{
void prepare_attention_decode(const int*, const int*, int, int, int, int, int, const device::Context&) {}
void attention_backend::prepare_flashinfer_prefill(const int*, int, int, int, int, int, int, int, cudaStream_t) {}
void attention_backend::prepare_flashinfer_prefill_ragged(const std::vector<int>&, const std::vector<int>&,
                                                          const std::vector<int>&, int, int, int, int, cudaStream_t)
{
}
bool attention_backend::launch_flashinfer_decode(Tensor&, Tensor&, Tensor&, Tensor&, int, int, float,
                                                 const device::Context&)
{
    return false;
}
bool attention_backend::launch_flashinfer_contiguous_prefill(Tensor&, Tensor&, Tensor&, Tensor&, int, float,
                                                             const device::Context&)
{
    return false;
}
bool attention_backend::launch_flashinfer_quantized_prefill(Tensor&, Tensor&, Tensor&, Tensor&, const int*, Tensor&,
                                                            int, int, int, float, const device::Context&)
{
    return false;
}
bool attention_backend::launch_flashinfer_prefill(Tensor&, Tensor&, Tensor&, Tensor&, const int*, int, int, int, int,
                                                  float, const device::Context&)
{
    return false;
}
bool attention_backend::launch_flashinfer_prefill_ragged(Tensor&, Tensor&, Tensor&, Tensor&, const int*, int, int,
                                                         float, const device::Context&)
{
    return false;
}
}  // namespace firefly::kernels

#endif
