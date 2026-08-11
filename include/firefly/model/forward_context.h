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
    std::span<Tensor> scale_layers;
    int*              block_table = nullptr;
    int               max_blocks_per_sequence = 0;

    [[nodiscard]] bool quantized() const { return !scale_layers.empty(); }
};

struct ModelInput
{
    const Tensor& input_ids;
    const Tensor& context_lens;
    KVCacheView   kv_cache;
    const int*    state_slots = nullptr;

    // Ragged batch metadata: input_ids flattened to (total_tokens, 1),
    // seq_offsets/seq_lengths describe each sequence's token range.
    const int* seq_offsets = nullptr;
    const int* seq_lengths = nullptr;
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
