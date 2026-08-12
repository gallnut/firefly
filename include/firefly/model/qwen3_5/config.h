#pragma once

#include <string>
#include <vector>

#include "firefly/model/model.h"

namespace firefly::model::qwen3_5
{

/** @brief Qwen3.5 hybrid-attention architecture and kernel configuration. */
struct Config
{
    ModelConfig              base; ///< Generic text-model configuration.
    std::vector<std::string> layer_types; ///< Ordered full-attention or linear-attention layer labels.
    int                      full_attention_interval = 4; ///< Expected spacing between full-attention layers.
    int                      linear_conv_kernel_dim = 4; ///< Causal convolution state width.
    int                      linear_key_head_dim = 128; ///< Key dimension of each Gated Delta Net head.
    int                      linear_num_key_heads = 16; ///< Number of linear-attention key heads.
    int                      linear_num_value_heads = 16; ///< Number of linear-attention value heads.
    int                      linear_value_head_dim = 128; ///< Value dimension of each linear-attention head.
    double                   partial_rotary_factor = 0.25; ///< Fraction of full-attention head channels using RoPE.
    bool                     attention_output_gate = true; ///< Whether full-attention outputs use learned gating.
    bool                     tie_word_embeddings = true; ///< Whether input embeddings also serve as the LM head.
};

}  // namespace firefly::model::qwen3_5
