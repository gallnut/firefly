#pragma once

#include <string>
#include <vector>

#include "firefly/model/model.h"

namespace firefly::model::qwen3_5
{

struct Config
{
    ModelConfig              base;
    std::vector<std::string> layer_types;
    int                      full_attention_interval = 4;
    int                      linear_conv_kernel_dim = 4;
    int                      linear_key_head_dim = 128;
    int                      linear_num_key_heads = 16;
    int                      linear_num_value_heads = 16;
    int                      linear_value_head_dim = 128;
    double                   partial_rotary_factor = 0.25;
    bool                     attention_output_gate = true;
    bool                     tie_word_embeddings = true;
};

}  // namespace firefly::model::qwen3_5
