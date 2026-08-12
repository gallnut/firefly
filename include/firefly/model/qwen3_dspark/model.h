#pragma once

#include "firefly/model/config_loader.h"
#include "firefly/model/model.h"
#include "firefly/model/speculative.h"

namespace firefly::model::qwen3_dspark
{

/**
 * @brief Qwen3-based DSpark proposer with target-context injection and Markov refinement.
 *
 * The model runs one non-causal draft backbone over an anchor-plus-mask block, then
 * sequentially refines each position through the learned Markov head. When trained,
 * the confidence head supplies prefix-pruning scores to the generic speculative runner.
 */
class Model final : public firefly::model::Model, public firefly::model::SpeculativeProposer
{
public:
    /**
     * @brief Parses DSpark metadata and constructs the internal Qwen3 draft backbone.
     * @param descriptor Architecture name, model directory, and parsed configuration JSON.
     * @return An unloaded proposer or a structured parse, model, or allocation error.
     */
    [[nodiscard]] static Result<std::unique_ptr<Model>> create(const ModelDescriptor& descriptor);

    /** @copydoc firefly::model::Model::accepts_weight */
    [[nodiscard]] bool accepts_weight(std::string_view name) const override;
    /** @copydoc firefly::model::Model::retains_source_weight */
    [[nodiscard]] bool retains_source_weight(std::string_view name) const override;
    /** @copydoc firefly::model::Model::runtime_requirements */
    [[nodiscard]] ModelRuntimeRequirements runtime_requirements() const override;
    /** @copydoc firefly::model::SpeculativeProposer::validate_target */
    Status validate_target(const SpeculativeTargetMetadata& target) const override;
    /** @copydoc firefly::model::SpeculativeProposer::initialize_speculative_runtime */
    Status initialize_speculative_runtime(int max_sequence_slots, const device::Context& context) override;
    /** @copydoc firefly::model::SpeculativeProposer::reset_speculative_runtime */
    Status reset_speculative_runtime(const device::Context& context) override;
    /** @copydoc firefly::model::SpeculativeProposer::target_hidden_layers */
    [[nodiscard]] std::span<const int> target_hidden_layers() const override { return target_layer_indices_; }
    /** @copydoc firefly::model::SpeculativeProposer::block_size */
    [[nodiscard]] int block_size() const override { return block_size_; }
    /** @copydoc firefly::model::Model::initialize_runtime */
    Status initialize_runtime(int max_sequence_slots, const device::Context& context) override;
    /** @copydoc firefly::model::Model::reset_runtime */
    Status reset_runtime(const device::Context& context) override;
    /** @copydoc firefly::model::Model::load_weights */
    Status load_weights(std::unordered_map<std::string, Tensor>& weights) override;
    /** @copydoc firefly::model::SpeculativeProposer::propose */
    Result<SpeculativeProposal> propose(const SpeculativeProposalInput& input) override;
    /** @copydoc firefly::model::Model::forward */
    Result<Tensor> forward(const ModelInput& input, const ForwardOptions& options = {}) override;

private:
    /** @brief Constructs a no-fail DSpark object from validated metadata and an initialized backbone. */
    Model(ModelConfig config, std::unique_ptr<firefly::model::Model> backbone);

    std::unique_ptr<firefly::model::Model> backbone_; ///< Owned Qwen3 draft backbone used for block feature extraction.
    ModelConfig config_{}; ///< Draft backbone dimensions and vocabulary metadata.
    Tensor context_projection_; ///< Projection from concatenated target layers into draft hidden width.
    Tensor hidden_norm_; ///< Normalization weight applied before the DSpark prediction heads.
    Tensor markov_w1_; ///< First low-rank Markov refinement projection.
    Tensor markov_w2_; ///< Second Markov projection mapping refined features to hidden width.
    Tensor confidence_weight_; ///< Optional confidence-head matrix used for prefix pruning.
    Tensor confidence_bias_; ///< Optional confidence-head bias used for prefix pruning.
    int   markov_rank_ = 0; ///< Intermediate width of the learned Markov refinement head.
    bool  confidence_enabled_ = false; ///< Whether both confidence-head tensors were loaded.
    std::vector<int> target_layer_ids_; ///< Layer identifiers recorded in DSpark model metadata.
    std::vector<int> target_layer_indices_; ///< Normalized zero-based target layers requested from the engine.
    int mask_token_id_ = 0; ///< Vocabulary ID inserted into unfilled draft positions.
    int block_size_ = 0; ///< Maximum number of tokens produced by one DSpark proposal.
};

}  // namespace firefly::model::qwen3_dspark
