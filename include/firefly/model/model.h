#pragma once

#include <string>
#include <string_view>
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

};

struct ModelRuntimeRequirements
{
    int  kv_cache_layer_count = 0;
    int  kv_cache_head_count = 0;
    int  kv_cache_head_dim = 0;
    int  prefill_chunk_limit = 0;
    bool sequence_state = false;
    bool prefix_cache = true;
    bool cuda_graph = true;
};

class Model
{
public:
    virtual ~Model() = default;

    virtual void load_weights(std::unordered_map<std::string, Tensor>& weights) = 0;

    [[nodiscard]] virtual bool accepts_weight(std::string_view) const { return true; }
    [[nodiscard]] virtual bool retains_source_weight(std::string_view) const { return true; }
    [[nodiscard]] virtual ModelRuntimeRequirements runtime_requirements() const = 0;
    virtual void initialize_runtime(int, const device::Context&) {}
    virtual void reset_runtime(const device::Context&) {}

    virtual Tensor forward(const ModelInput& input, const ForwardOptions& options = {}) = 0;
};

}  // namespace firefly::model
