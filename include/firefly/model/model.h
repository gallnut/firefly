#pragma once

#include "firefly/core/tensor.h"
#include "firefly/model/forward_context.h"

#include <string>
#include <string_view>
#include <unordered_map>

namespace firefly::model
{

/** @brief Architecture-independent model dimensions and numerical hyperparameters. */
struct ModelConfig
{
    DType  dtype = DType::F16; ///< Weight and activation scalar type.
    int    hidden_size; ///< Transformer residual-stream width.
    int    intermediate_size; ///< Feed-forward expansion width.
    int    num_hidden_layers; ///< Number of transformer or hybrid mixer layers.
    int    num_attention_heads; ///< Number of query attention heads.
    int    num_key_value_heads; ///< Number of key/value heads after grouped-query sharing.
    int    head_dim; ///< Scalar width of each attention head.
    int    vocab_size; ///< Number of token logits produced by the language head.
    int    max_position_embeddings; ///< Maximum configured position index.
    double rms_norm_eps; ///< Epsilon used by RMS normalization kernels.
    float  rope_theta; ///< Base period used by rotary position embeddings.

};

/** @brief Runtime capabilities and memory dimensions advertised by a model implementation. */
struct ModelRuntimeRequirements
{
    int  kv_cache_layer_count = 0; ///< Number of layers that allocate paged key/value cache storage.
    int  kv_cache_head_count = 0; ///< Key/value heads stored per cached token.
    int  kv_cache_head_dim = 0; ///< Scalars stored by each cached key/value head.
    int  prefill_chunk_limit = 0; ///< Model-specific maximum prefill chunk, or zero for no additional limit.
    bool sequence_state = false; ///< Whether the model owns mutable recurrent state indexed by sequence slot.
    bool prefix_cache = true; ///< Whether paged KV blocks may safely participate in prefix sharing.
    bool cuda_graph = true; ///< Whether ordinary decode is safe to capture and replay through CUDA Graph.
};

/**
 * @brief Polymorphic inference-model interface used by the execution engine.
 *
 * The interface deliberately contains only ordinary model execution. Optional features,
 * including speculative proposal and transactional recurrent state, are expressed by
 * separate capability interfaces so disabled features impose no model-side state.
 */
class Model
{
public:
    /** @brief Enables destruction through the base interface. */
    virtual ~Model() = default;

    /**
     * @brief Moves recognized tensors from a loaded weight map into model-owned fields.
     * @param weights Mutable map of device tensors keyed by checkpoint name.
     * @return Success or a structured missing-weight, compatibility, allocation, or kernel error.
     */
    virtual Status load_weights(std::unordered_map<std::string, Tensor>& weights) = 0;

    /**
     * @brief Reports whether a named checkpoint tensor is consumed by this model.
     * @return `true` by default so generic models load every checkpoint tensor.
     */
    [[nodiscard]] virtual bool accepts_weight(std::string_view) const { return true; }
    /**
     * @brief Reports whether an accepted tensor must remain in persistent pooled storage.
     * @return `true` by default; implementations may return false for temporary fusion inputs.
     */
    [[nodiscard]] virtual bool retains_source_weight(std::string_view) const { return true; }
    /**
     * @brief Describes cache dimensions and optional runtime capabilities.
     * @return Architecture requirements used by the engine before allocating runtime resources.
     */
    [[nodiscard]] virtual ModelRuntimeRequirements runtime_requirements() const = 0;
    /**
     * @brief Allocates model-specific runtime state for the requested sequence-slot capacity.
     * @note The default implementation is stateless and performs no work.
     */
    virtual Status initialize_runtime(int, const device::Context&) { return {}; }
    /**
     * @brief Resets all model-specific runtime state to its initial value.
     * @note The default implementation is stateless and performs no work.
     */
    virtual Status reset_runtime(const device::Context&) { return {}; }

    /**
     * @brief Executes one prefill, decode, verification, or ragged mixed-batch forward pass.
     * @param input Token IDs, context lengths, cache views, and optional ragged metadata.
     * @param options Execution controls and optional side-channel outputs.
     * @return Logits requested by `options`, or an empty tensor when logits are disabled.
     */
    virtual Result<Tensor> forward(const ModelInput& input, const ForwardOptions& options = {}) = 0;
};

}  // namespace firefly::model
