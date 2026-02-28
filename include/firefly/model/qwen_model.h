#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/model/model.h"
#include "firefly/tensor.h"

namespace firefly::model
{

// Basic Layers
// In a real implementation, we would define classes for Attention, MLP, RMSNorm, Embedding
// For now, we store tensors directly in the model or in simplified layer structs.

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

    // Model Weights
    Tensor                 token_embeddings;
    std::vector<QwenLayer> layers;
    Tensor                 norm;     // Final RMSNorm
    Tensor                 lm_head;  // Output projection

    QwenModel(const ModelConfig& cfg);
    ~QwenModel() override = default;

    void load_weights(std::unordered_map<std::string, Tensor>& weights) override;

    Tensor forward(const Tensor& input_ids, const Tensor& context_lens, std::vector<Tensor>& k_caches,
                   std::vector<Tensor>& v_caches, int* block_table, int max_blocks) override;
};

}  // namespace firefly::model
