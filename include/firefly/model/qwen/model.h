#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/model/model.h"

namespace firefly::model::qwen
{

/** @brief Weight tensors for one Qwen grouped-query self-attention module. */
struct QwenAttention
{
    Tensor q_proj; ///< Query projection weights.
    Tensor k_proj; ///< Key projection weights.
    Tensor v_proj; ///< Value projection weights.
    Tensor o_proj; ///< Attention output projection weights.

    Tensor q_norm; ///< Optional per-head query RMS normalization weights.
    Tensor k_norm; ///< Optional per-head key RMS normalization weights.
};

/** @brief Weight tensors for one Qwen gated feed-forward network. */
struct QwenMLP
{
    Tensor gate_proj; ///< SiLU gate projection weights.
    Tensor up_proj; ///< Parallel value projection weights.
    Tensor down_proj; ///< Projection from intermediate activations to the residual width.
};

/** @brief All attention, MLP, and normalization weights for one Qwen transformer layer. */
struct QwenLayer
{
    QwenAttention attention; ///< Self-attention weights.
    QwenMLP       mlp; ///< Feed-forward weights.
    Tensor        input_layernorm; ///< Pre-attention RMS normalization weights.
    Tensor        post_attention_layernorm; ///< Pre-MLP RMS normalization weights.
};

/** @brief CUDA inference implementation shared by Qwen2 and dense Qwen3 checkpoints. */
class QwenModel : public Model
{
public:
    ModelConfig config; ///< Parsed model dimensions and numerical hyperparameters.

    Tensor                 token_embeddings; ///< Input embedding table.
    std::vector<QwenLayer> layers; ///< Ordered transformer layers.
    Tensor                 norm; ///< Final RMS normalization weights.
    Tensor                 hidden_norm; ///< Optional architecture-specific hidden-state normalization.
    Tensor                 lm_head; ///< Vocabulary projection weights.

    /**
     * @brief Constructs an unloaded Qwen model with storage for configured layers.
     * @param cfg Parsed architecture dimensions and numerical hyperparameters.
     */
    QwenModel(const ModelConfig& cfg);
    /** @brief Releases all owned weight tensors. */
    ~QwenModel() override = default;

    /** @copydoc firefly::model::Model::load_weights */
    Status load_weights(std::unordered_map<std::string, Tensor>& weights) override;
    /** @copydoc firefly::model::Model::runtime_requirements */
    [[nodiscard]] ModelRuntimeRequirements runtime_requirements() const override;

    /** @copydoc firefly::model::Model::forward */
    Result<Tensor> forward(const ModelInput& input, const ForwardOptions& options = {}) override;
};

}  // namespace firefly::model::qwen
