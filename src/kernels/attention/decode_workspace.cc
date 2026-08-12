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

Status ensure_decode_workspace(int batch_size, int num_heads, int num_splits, int head_dim)
{
    if (batch_size <= 0 || num_heads <= 0 || num_splits <= 0 || head_dim <= 0)
        return unexpected(Error{ErrorCode::InvalidArgument, "decode workspace dimensions must be positive"});
    auto&   workspace = decode_workspace();
    int64_t partial_count = static_cast<int64_t>(batch_size) * num_heads * num_splits;
    int64_t accumulator_count = partial_count * head_dim;

    if (workspace.partial_m_capacity < partial_count)
    {
        workspace.partial_m = FIREFLY_TRY(Tensor::create({partial_count}, DType::F32, Device::CUDA));
        workspace.partial_m_capacity = partial_count;
    }
    if (workspace.partial_l_capacity < partial_count)
    {
        workspace.partial_l = FIREFLY_TRY(Tensor::create({partial_count}, DType::F32, Device::CUDA));
        workspace.partial_l_capacity = partial_count;
    }
    if (workspace.partial_acc_capacity < accumulator_count)
    {
        workspace.partial_acc = FIREFLY_TRY(Tensor::create({accumulator_count}, DType::F32, Device::CUDA));
        workspace.partial_acc_capacity = accumulator_count;
    }

    int64_t query_count = static_cast<int64_t>(batch_size) * num_heads * head_dim;
    int64_t query_scale_count = static_cast<int64_t>(batch_size) * num_heads;
    if (workspace.quantized_query_capacity < query_count)
    {
        workspace.quantized_query = FIREFLY_TRY(Tensor::create({query_count}, DType::I8, Device::CUDA));
        workspace.quantized_query_capacity = query_count;
    }
    if (workspace.quantized_query_scales_capacity < query_scale_count)
    {
        workspace.quantized_query_scales =
            FIREFLY_TRY(Tensor::create({query_scale_count}, DType::F32, Device::CUDA));
        workspace.quantized_query_scales_capacity = query_scale_count;
    }
    return {};
}

Status reserve_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim, int split_size)
{
    int token_count = max_context_blocks * 16;
    int num_splits = std::max(1, (token_count + split_size - 1) / split_size);
    return ensure_decode_workspace(batch_size, num_heads, num_splits, head_dim);
}
}  // namespace firefly::kernels::attention_detail
