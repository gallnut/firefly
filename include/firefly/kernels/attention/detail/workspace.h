#pragma once

#include <cstdint>

#include "firefly/core/tensor.h"

namespace firefly::kernels::attention_detail
{
struct DecodeWorkspace
{
    Tensor  partial_m;
    Tensor  partial_l;
    Tensor  partial_acc;
    Tensor  quantized_query;
    Tensor  quantized_query_scales;
    int64_t partial_m_capacity = 0;
    int64_t partial_l_capacity = 0;
    int64_t partial_acc_capacity = 0;
    int64_t quantized_query_capacity = 0;
    int64_t quantized_query_scales_capacity = 0;
};

DecodeWorkspace& decode_workspace();
void             ensure_decode_workspace(int batch_size, int num_heads, int num_splits, int head_dim);
}  // namespace firefly::kernels::attention_detail
