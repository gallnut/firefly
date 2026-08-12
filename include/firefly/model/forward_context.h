#pragma once

#include <span>

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::model
{

/** @brief Non-owning view of the paged key/value cache supplied to one model forward pass. */
struct KVCacheView
{
    std::span<Tensor> key_layers;    ///< Per-attention-layer key-cache tensors.
    std::span<Tensor> value_layers;  ///< Per-attention-layer value-cache tensors.
    std::span<Tensor> scale_layers;  ///< Optional per-layer quantization scale tensors.
    int*              block_table = nullptr; ///< Device pointer mapping logical pages to physical cache blocks.
    int               max_blocks_per_sequence = 0; ///< Row stride of the flattened block table.

    /** @brief Returns true when scale tensors identify an integer-quantized cache. */
    [[nodiscard]] bool quantized() const { return !scale_layers.empty(); }
};

/** @brief Tensor inputs and cache metadata consumed by a model forward pass. */
struct ModelInput
{
    const Tensor& input_ids;    ///< Token IDs, normally shaped `[batch, sequence]` or ragged `[tokens, 1]`.
    const Tensor& context_lens; ///< Device tensor containing the materialized context length per sequence.
    KVCacheView   kv_cache;     ///< Borrowed paged-cache views updated by the forward pass.
    const int*    state_slots = nullptr; ///< Device slot indices for models with recurrent sequence state.

    // Ragged batch metadata: input_ids flattened to (total_tokens, 1),
    // seq_offsets/seq_lengths describe each sequence's token range.
    const int* seq_offsets = nullptr; ///< Device offsets delimiting rows in a flattened ragged token tensor.
    const int* seq_lengths = nullptr; ///< Device sequence lengths for each ragged batch row.
};

/** @brief Optional execution controls and side-channel outputs for `Model::forward`. */
struct ForwardOptions
{
    device::Context context; ///< CUDA stream and allocation context for all asynchronous work.
    bool compute_logits = true; ///< Whether the model must execute its final normalization and language head.
    bool return_all_logits = false; ///< Whether logits are returned for every input position instead of only the last.
    Tensor* hidden_state_output = nullptr; ///< Optional owner receiving the final transformer hidden states.
    std::span<const int> hidden_state_layers; ///< Sorted zero-based layers requested through the layered side channel.
    Tensor* layered_hidden_state_output = nullptr; ///< Optional tensor receiving requested layer outputs.
    Tensor* auxiliary_output = nullptr; ///< Architecture-defined auxiliary result, such as confidence scores.
    const Tensor* input_hidden_bias = nullptr; ///< Optional bias added immediately after token embedding lookup.
    const Tensor* attention_prefix = nullptr; ///< Optional learned prefix injected into self-attention.
    const Tensor* injected_context = nullptr; ///< Optional architecture-defined context supplied by a caller.
    bool causal_attention = true; ///< Whether attention masks future positions inside the current input block.
    bool prefer_split_decode = false; ///< Hints that long-context decode should use a split reduction kernel.
    int  max_decode_context_len = 0; ///< Largest context length represented by the current decode batch.
    int  min_context_len = -1; ///< Smallest existing context length for paged prefill planning, or `-1` if unknown.
};

}  // namespace firefly::model
