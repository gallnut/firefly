#pragma once

#include <span>

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::model
{

struct KVCacheView
{
    std::span<Tensor> key_layers;
    std::span<Tensor> value_layers;
    int*              block_table = nullptr;
    int               max_blocks_per_sequence = 0;
};

struct ModelInput
{
    const Tensor& input_ids;
    const Tensor& context_lens;
    KVCacheView   kv_cache;
};

struct ForwardOptions
{
    device::Context context;
    bool compute_logits = true;
    bool prefer_split_decode = false;
    int  max_decode_context_len = 0;
    int  min_context_len = -1;
};

}  // namespace firefly::model
