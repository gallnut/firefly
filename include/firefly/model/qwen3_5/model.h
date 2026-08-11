#pragma once

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "firefly/kernels/linear_attention/gated_delta_net.h"
#include "firefly/model/config_loader.h"
#include "firefly/model/qwen3_5/config.h"

namespace firefly::model::qwen3_5
{

struct MLP
{
    Tensor fused_projection;
    Tensor down_projection;
};

struct FullAttention
{
    Tensor query_projection;
    Tensor key_projection;
    Tensor value_projection;
    Tensor output_projection;
    Tensor query_norm;
    Tensor key_norm;
};

struct LinearAttention
{
    Tensor fused_projection;
    Tensor convolution_weight;
    Tensor decay_log;
    Tensor decay_bias;
    Tensor norm_weight;
    Tensor output_projection;
};

struct Layer
{
    bool            full_attention = false;
    FullAttention   attention;
    LinearAttention linear_attention;
    MLP             mlp;
    Tensor          input_norm;
    Tensor          post_attention_norm;
};

class Model final : public firefly::model::Model
{
public:
    explicit Model(const ModelDescriptor& descriptor);
    ~Model() override;

    [[nodiscard]] bool accepts_weight(std::string_view name) const override;
    [[nodiscard]] bool retains_source_weight(std::string_view name) const override;
    [[nodiscard]] ModelRuntimeRequirements runtime_requirements() const override;
    void initialize_runtime(int max_sequence_slots, const device::Context& context) override;
    void reset_runtime(const device::Context& context) override;
    void load_weights(std::unordered_map<std::string, Tensor>& weights) override;
    Tensor forward(const ModelInput& input, const ForwardOptions& options = {}) override;

private:
    Config                  config_;
    Tensor                  token_embeddings_;
    std::vector<Layer>      layers_;
    Tensor                  final_norm_;
    std::vector<Tensor>     convolution_states_;
    std::vector<Tensor>     recurrent_states_;
    std::unique_ptr<firefly::kernels::linear_attention::GatedDeltaNetWorkspace> gated_delta_net_workspace_;
    int                     max_sequence_slots_ = 0;
};

}  // namespace firefly::model::qwen3_5
