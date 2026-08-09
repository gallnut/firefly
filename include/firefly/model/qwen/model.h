#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/model/model.h"

namespace firefly::model::qwen
{

struct QwenAttention
{
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;

    Tensor q_norm;
    Tensor k_norm;
};

struct QwenMLP
{
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

struct QwenLayer
{
    QwenAttention attention;
    QwenMLP       mlp;
    Tensor        input_layernorm;
    Tensor        post_attention_layernorm;
};

class QwenModel : public Model
{
public:
    ModelConfig config;

    Tensor                 token_embeddings;
    std::vector<QwenLayer> layers;
    Tensor                 norm;
    Tensor                 lm_head;

    QwenModel(const ModelConfig& cfg);
    ~QwenModel() override = default;

    void load_weights(std::unordered_map<std::string, Tensor>& weights) override;
    [[nodiscard]] ModelRuntimeRequirements runtime_requirements() const override;

    Tensor forward(const ModelInput& input, const ForwardOptions& options = {}) override;
};

}  // namespace firefly::model::qwen
