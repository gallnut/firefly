#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/tensor.h"

namespace firefly::model
{

struct ModelConfig
{
    int    hidden_size;
    int    intermediate_size;
    int    num_hidden_layers;
    int    num_attention_heads;
    int    num_key_value_heads;
    int    head_dim;
    int    vocab_size;
    int    max_position_embeddings;
    double rms_norm_eps;
    float  rope_theta;

    virtual ~ModelConfig() = default;
};

class Model
{
public:
    virtual ~Model() = default;

    virtual void load_weights(std::unordered_map<std::string, Tensor>& weights) = 0;

    /**
     * @brief Forward pass for the model.
     * @param input_ids Token IDs of shape [batch_size, seq_len]
     * @param context_lens Tensor of shape [batch_size] containing the current context length for each sequence
     * @param k_caches K cache blocks of shape [num_layers, max_blocks, 16, num_kv_heads, head_dim]
     * @param v_caches V cache blocks of shape [num_layers, max_blocks, 16, num_kv_heads, head_dim]
     * @param block_table Block table for paged attention
     * @param max_blocks Maximum number of blocks per sequence
     * @return Tensor shape [batch_size, seq_len, vocab_size] representing logits.
     */
    virtual Tensor forward(const Tensor& input_ids, const Tensor& context_lens, std::vector<Tensor>& k_caches,
                           std::vector<Tensor>& v_caches, int* block_table, int max_blocks) = 0;
};

}  // namespace firefly::model
