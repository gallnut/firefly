#pragma once

#include <string>
#include <unordered_map>
#include "firefly/model/forward_context.h"
#include "firefly/core/tensor.h"

namespace firefly::model
{

struct ModelConfig
{
    DType  dtype = DType::F16;
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

    virtual Tensor forward(const ModelInput& input, const ForwardOptions& options = {}) = 0;
};

}  // namespace firefly::model
