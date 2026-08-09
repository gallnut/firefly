#include "firefly/kernels/attention/detail/workspace.h"

#include <algorithm>

#include "firefly/kernels/attention/detail/launch.h"

namespace firefly::kernels::attention_detail
{
DecodeWorkspace& decode_workspace()
{
    static thread_local DecodeWorkspace* workspace = new DecodeWorkspace();
    return *workspace;
}

void ensure_decode_workspace(int batch_size, int num_heads, int num_splits, int head_dim)
{
    auto&   workspace = decode_workspace();
    int64_t partial_count = static_cast<int64_t>(batch_size) * num_heads * num_splits;
    int64_t accumulator_count = partial_count * head_dim;

    if (workspace.partial_m_capacity < partial_count)
    {
        workspace.partial_m = Tensor({partial_count}, DType::F32, Device::CUDA);
        workspace.partial_m_capacity = partial_count;
    }
    if (workspace.partial_l_capacity < partial_count)
    {
        workspace.partial_l = Tensor({partial_count}, DType::F32, Device::CUDA);
        workspace.partial_l_capacity = partial_count;
    }
    if (workspace.partial_acc_capacity < accumulator_count)
    {
        workspace.partial_acc = Tensor({accumulator_count}, DType::F32, Device::CUDA);
        workspace.partial_acc_capacity = accumulator_count;
    }

    int64_t query_count = static_cast<int64_t>(batch_size) * num_heads * head_dim;
    int64_t query_scale_count = static_cast<int64_t>(batch_size) * num_heads;
    if (workspace.quantized_query_capacity < query_count)
    {
        workspace.quantized_query = Tensor({query_count}, DType::I8, Device::CUDA);
        workspace.quantized_query_capacity = query_count;
    }
    if (workspace.quantized_query_scales_capacity < query_scale_count)
    {
        workspace.quantized_query_scales = Tensor({query_scale_count}, DType::F32, Device::CUDA);
        workspace.quantized_query_scales_capacity = query_scale_count;
    }
}

void reserve_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim, int split_size)
{
    int token_count = max_context_blocks * 16;
    int num_splits = std::max(1, (token_count + split_size - 1) / split_size);
    ensure_decode_workspace(batch_size, num_heads, num_splits, head_dim);
}
}  // namespace firefly::kernels::attention_detail
