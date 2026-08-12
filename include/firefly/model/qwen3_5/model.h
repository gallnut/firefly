#pragma once

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "firefly/kernels/linear_attention/gated_delta_net.h"
#include "firefly/model/config_loader.h"
#include "firefly/model/qwen3_5/config.h"
#include "firefly/model/speculative.h"

namespace firefly::model::qwen3_5
{

/** @brief Fused projection weights for one Qwen3.5 gated feed-forward network. */
struct MLP
{
    Tensor fused_projection; ///< Concatenated gate and up-projection weights.
    Tensor down_projection; ///< Projection from intermediate activations to the residual width.
};

/** @brief Weight tensors for a Qwen3.5 full-attention mixer. */
struct FullAttention
{
    Tensor query_projection; ///< Query and optional gate projection weights.
    Tensor key_projection; ///< Key projection weights.
    Tensor value_projection; ///< Value projection weights.
    Tensor output_projection; ///< Attention output projection weights.
    Tensor query_norm; ///< Per-head query normalization weights.
    Tensor key_norm; ///< Per-head key normalization weights.
};

/** @brief Weight tensors for a Qwen3.5 Gated Delta Net mixer. */
struct LinearAttention
{
    Tensor fused_projection; ///< Joint projection producing Q/K/V, gates, decay, and update coefficients.
    Tensor convolution_weight; ///< Depthwise causal-convolution weights.
    Tensor decay_log; ///< Learned log-decay parameter.
    Tensor decay_bias; ///< Bias applied to the decay projection.
    Tensor norm_weight; ///< Output normalization weights inside the recurrent mixer.
    Tensor output_projection; ///< Projection from linear-attention output to residual width.
};

/** @brief One Qwen3.5 hybrid layer containing either full or linear attention plus an MLP. */
struct Layer
{
    bool            full_attention = false; ///< Selects the active mixer representation.
    FullAttention   attention; ///< Full-attention weights when `full_attention` is true.
    LinearAttention linear_attention; ///< Gated Delta Net weights when `full_attention` is false.
    MLP             mlp; ///< Feed-forward network weights.
    Tensor          input_norm; ///< Pre-mixer zero-centered RMS normalization weights.
    Tensor          post_attention_norm; ///< Pre-MLP zero-centered RMS normalization weights.
};

/**
 * @brief Qwen3.5 hybrid recurrent inference model and transactional speculative target.
 *
 * The model owns convolution and Gated Delta Net state for each scheduler slot. Its
 * speculative capability snapshots a single slot so verification can be rolled back
 * and replayed without perturbing exact greedy generation.
 */
class Model final : public firefly::model::Model, public firefly::model::SpeculativeTargetRuntime
{
public:
    /**
     * @brief Parses Qwen3.5-specific configuration and constructs unloaded layer objects.
     * @param descriptor Generic model metadata and raw architecture-specific JSON.
     * @return Unloaded model or a structured parse or validation error.
     */
    static Result<std::unique_ptr<Model>> create(const ModelDescriptor& descriptor);
    /** @brief Releases weights, recurrent state, and Gated Delta Net workspace. */
    ~Model() override;

    /** @copydoc firefly::model::Model::accepts_weight */
    [[nodiscard]] bool accepts_weight(std::string_view name) const override;
    /** @copydoc firefly::model::Model::retains_source_weight */
    [[nodiscard]] bool retains_source_weight(std::string_view name) const override;
    /** @copydoc firefly::model::Model::runtime_requirements */
    [[nodiscard]] ModelRuntimeRequirements runtime_requirements() const override;
    /** @copydoc firefly::model::Model::initialize_runtime */
    Status initialize_runtime(int max_sequence_slots, const device::Context& context) override;
    /** @copydoc firefly::model::Model::reset_runtime */
    Status reset_runtime(const device::Context& context) override;
    /** @copydoc firefly::model::SpeculativeTargetRuntime::snapshot_speculative_runtime */
    [[nodiscard]] Result<std::unique_ptr<SpeculativeRuntimeState>> snapshot_speculative_runtime(
        int state_slot, const device::Context& context) override;
    /** @copydoc firefly::model::SpeculativeTargetRuntime::restore_speculative_runtime */
    Status restore_speculative_runtime(std::unique_ptr<SpeculativeRuntimeState> state,
                                       const device::Context& context) override;
    /** @copydoc firefly::model::Model::load_weights */
    Status load_weights(std::unordered_map<std::string, Tensor>& weights) override;
    /** @copydoc firefly::model::Model::forward */
    Result<Tensor> forward(const ModelInput& input, const ForwardOptions& options = {}) override;

private:
    explicit Model(Config config);
    Config                  config_; ///< Parsed hybrid architecture and numerical configuration.
    Tensor                  token_embeddings_; ///< Input embedding table and optionally tied LM head.
    std::vector<Layer>      layers_; ///< Ordered full-attention and linear-attention layer weights.
    Tensor                  final_norm_; ///< Final zero-centered RMS normalization weights.
    std::vector<Tensor>     convolution_states_; ///< Per-linear-layer causal convolution state by sequence slot.
    std::vector<Tensor>     recurrent_states_; ///< Per-linear-layer Gated Delta Net state by sequence slot.
    std::unique_ptr<firefly::kernels::linear_attention::GatedDeltaNetWorkspace> gated_delta_net_workspace_; ///< Reusable kernel planning and scratch state.
    int                     max_sequence_slots_ = 0; ///< Allocated leading dimension of recurrent state tensors.
};

}  // namespace firefly::model::qwen3_5
