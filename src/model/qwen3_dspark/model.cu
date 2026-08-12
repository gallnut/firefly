#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "firefly/core/logging.h"
#include "firefly/device/error.h"
#include "firefly/kernels/transformer/embedding.h"
#include "firefly/kernels/transformer/linear.h"
#include "firefly/kernels/transformer/residual.h"
#include "firefly/kernels/transformer/rms_norm.h"
#include "firefly/kernels/sampling/argmax.h"
#include "firefly/model/qwen3_dspark/model.h"
#include "firefly/model/registry.h"

namespace firefly::model::qwen3_dspark
{
namespace
{
template <typename scalar_t>
__global__ void sigmoid_kernel(scalar_t* values, const scalar_t* bias, int count)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x)
    {
        float value = 0.0f;
        if constexpr (std::is_same_v<scalar_t, half>)
            value = __half2float(values[index]) + __half2float(*bias);
        else
            value = __bfloat162float(values[index]) + __bfloat162float(*bias);
        const float sigmoid = 1.0f / (1.0f + expf(-value));
        if constexpr (std::is_same_v<scalar_t, half>)
            values[index] = __float2half(sigmoid);
        else
            values[index] = __float2bfloat16(sigmoid);
    }
}

Result<Tensor> take(std::unordered_map<std::string, Tensor>& weights, std::string_view name)
{
    auto found = weights.find(std::string(name));
    if (found == weights.end())
        return unexpected(Error{ErrorCode::NotFound, "missing DSpark weight: " + std::string(name)});
    return std::move(found->second);
}

Status move_weight(std::unordered_map<std::string, Tensor>& source,
                   std::unordered_map<std::string, Tensor>& target, const std::string& source_name,
                   const std::string& target_name)
{
    target.emplace(target_name, FIREFLY_TRY(take(source, source_name)));
    return {};
}

}  // namespace

Model::Model(ModelConfig config, std::unique_ptr<firefly::model::Model> backbone)
    : backbone_(std::move(backbone)), config_(std::move(config))
{
}

Result<std::unique_ptr<Model>> Model::create(const ModelDescriptor& descriptor)
{
    int              markov_rank = 0;
    bool             confidence_enabled = false;
    std::vector<int> target_layer_ids;
    int              mask_token_id = 0;
    int              block_size = 0;
    try
    {
        const auto root = nlohmann::json::parse(descriptor.raw_config);
        markov_rank = root.value("markov_rank", 256);
        confidence_enabled = root.value("enable_confidence_head", false);
        target_layer_ids = root.value("target_layer_ids", std::vector<int>{});
        mask_token_id = root.value("mask_token_id", 0);
        block_size = root.value("block_size", 0);
    }
    catch (const nlohmann::json::exception& exception)
    {
        return unexpected(Error{ErrorCode::Parse, "invalid Qwen3 DSpark configuration: " +
                                                      std::string(exception.what())});
    }

    if (markov_rank <= 0)
        return unexpected(Error{ErrorCode::InvalidArgument, "Qwen3 DSpark markov_rank must be positive"});
    if (block_size <= 0)
        return unexpected(Error{ErrorCode::InvalidArgument, "Qwen3 DSpark block_size must be positive"});
    if (target_layer_ids.empty())
        return unexpected(Error{ErrorCode::InvalidArgument, "Qwen3 DSpark target_layer_ids must not be empty"});
    if (mask_token_id < 0 || mask_token_id >= descriptor.config.vocab_size)
        return unexpected(Error{ErrorCode::InvalidArgument, "Qwen3 DSpark mask_token_id is outside the vocabulary"});

    ModelDescriptor backbone_descriptor = descriptor;
    backbone_descriptor.architecture = "Qwen3ForCausalLM";
    auto backbone = FIREFLY_TRY_CONTEXT(ModelRegistry::get().create(backbone_descriptor),
                                        "create Qwen3 DSpark draft backbone");
    auto model = std::unique_ptr<Model>(new Model(descriptor.config, std::move(backbone)));
    model->markov_rank_ = markov_rank;
    model->confidence_enabled_ = confidence_enabled;
    model->target_layer_ids_ = std::move(target_layer_ids);
    model->target_layer_indices_ = model->target_layer_ids_;
    for (int& layer : model->target_layer_indices_) --layer;
    model->mask_token_id_ = mask_token_id;
    model->block_size_ = block_size;
    return model;
}

bool Model::accepts_weight(std::string_view) const { return true; }

bool Model::retains_source_weight(std::string_view) const { return true; }

ModelRuntimeRequirements Model::runtime_requirements() const { return backbone_->runtime_requirements(); }

Status Model::validate_target(const SpeculativeTargetMetadata& target) const
{
    if (target.hidden_size != config_.hidden_size || target.vocab_size != config_.vocab_size)
        return unexpected(Error{ErrorCode::Model,
                                "Qwen3 DSpark target hidden size or vocabulary is incompatible"});
    for (const int layer : target_layer_indices_)
        if (layer < 0 || layer >= target.num_hidden_layers)
            return unexpected(Error{ErrorCode::Model, "Qwen3 DSpark target layer id is incompatible"});
    return {};
}

Status Model::initialize_runtime(int max_sequence_slots, const device::Context& context)
{
    return backbone_->initialize_runtime(max_sequence_slots, context);
}

Status Model::reset_runtime(const device::Context& context) { return backbone_->reset_runtime(context); }

Status Model::initialize_speculative_runtime(int max_sequence_slots, const device::Context& context)
{
    return initialize_runtime(max_sequence_slots, context);
}

Status Model::reset_speculative_runtime(const device::Context& context) { return reset_runtime(context); }

Status Model::load_weights(std::unordered_map<std::string, Tensor>& weights)
{
    auto embedding = weights.find("embed_tokens.weight");
    if (embedding == weights.end())
        return unexpected(Error{ErrorCode::NotFound, "missing DSpark weight: embed_tokens.weight"});
    config_.dtype = embedding->second.dtype();
    FIREFLY_TRY(require_float16_or_bfloat16(config_.dtype, "Qwen3 DSpark"));
    std::unordered_map<std::string, Tensor> backbone_weights;
    FIREFLY_TRY(move_weight(weights, backbone_weights, "embed_tokens.weight", "model.embed_tokens.weight"));
    for (int index = 0; index < config_.num_hidden_layers; ++index)
    {
        const std::string prefix = "layers." + std::to_string(index) + ".";
        const std::string target_prefix = "model.layers." + std::to_string(index) + ".";
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "self_attn.q_proj.weight", target_prefix + "self_attn.q_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "self_attn.k_proj.weight", target_prefix + "self_attn.k_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "self_attn.v_proj.weight", target_prefix + "self_attn.v_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "self_attn.o_proj.weight", target_prefix + "self_attn.o_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "self_attn.q_norm.weight", target_prefix + "self_attn.q_norm.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "self_attn.k_norm.weight", target_prefix + "self_attn.k_norm.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "mlp.gate_proj.weight", target_prefix + "mlp.gate_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "mlp.up_proj.weight", target_prefix + "mlp.up_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "mlp.down_proj.weight", target_prefix + "mlp.down_proj.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "input_layernorm.weight", target_prefix + "input_layernorm.weight"));
        FIREFLY_TRY(move_weight(weights, backbone_weights, prefix + "post_attention_layernorm.weight", target_prefix + "post_attention_layernorm.weight"));
    }
    FIREFLY_TRY(move_weight(weights, backbone_weights, "norm.weight", "model.norm.weight"));
    FIREFLY_TRY(move_weight(weights, backbone_weights, "lm_head.weight", "lm_head.weight"));
    FIREFLY_TRY_CONTEXT(backbone_->load_weights(backbone_weights), "load Qwen3 DSpark draft backbone weights");
    hidden_norm_ = FIREFLY_TRY(take(weights, "hidden_norm.weight"));
    context_projection_ = FIREFLY_TRY(take(weights, "fc.weight"));
    markov_w1_ = FIREFLY_TRY(take(weights, "markov_head.markov_w1.weight"));
    markov_w2_ = FIREFLY_TRY(take(weights, "markov_head.markov_w2.weight"));
    if (confidence_enabled_)
    {
        confidence_weight_ = FIREFLY_TRY(take(weights, "confidence_head.proj.weight"));
        confidence_bias_ = FIREFLY_TRY(take(weights, "confidence_head.proj.bias"));
    }
    FIREFLY_LOG_INFO("model", "Qwen3 DSpark weights mapped layers={} block_size={} markov_rank={}",
                     config_.num_hidden_layers, block_size_, markov_rank_);
    return {};
}

Result<Tensor> Model::forward(const ModelInput& input, const ForwardOptions& options)
{
    Tensor hidden_states;
    Tensor context_bias;
    ForwardOptions backbone_options = options;
    backbone_options.hidden_state_output = &hidden_states;
    if (options.injected_context != nullptr && context_projection_.data() != nullptr)
    {
        Tensor projected_input = Tensor::from_external(
            options.injected_context->data(),
            {input.input_ids.shape()[0], static_cast<int64_t>(target_layer_ids_.size()) * config_.hidden_size},
            options.injected_context->dtype(), Device::CUDA);
        context_bias = FIREFLY_TRY(Tensor::create(
            {input.input_ids.shape()[0], input.input_ids.shape()[1], config_.hidden_size}, config_.dtype,
            Device::CUDA, options.context));
        FIREFLY_TRY(kernels::matmul(projected_input, context_projection_, context_bias, options.context));
        FIREFLY_TRY(kernels::rms_norm(context_bias, hidden_norm_, context_bias, config_.rms_norm_eps,
                                      options.context));
        backbone_options.input_hidden_bias = &context_bias;
    }
    Tensor logits = FIREFLY_TRY_CONTEXT(backbone_->forward(input, backbone_options),
                                        "execute Qwen3 DSpark draft backbone");
    if (!options.compute_logits || logits.data() == nullptr) return logits;

    if (markov_w1_.data() != nullptr && markov_w2_.data() != nullptr)
    {
        Tensor previous_embedding = FIREFLY_TRY(Tensor::create(
            {input.input_ids.shape()[0], input.input_ids.shape()[1], markov_rank_}, config_.dtype, Device::CUDA,
            options.context));
        FIREFLY_TRY(kernels::embedding_lookup(input.input_ids, markov_w1_, previous_embedding, options.context));
        Tensor markov_bias = FIREFLY_TRY(Tensor::create(logits.shape(), config_.dtype, Device::CUDA,
                                                        options.context));
        FIREFLY_TRY(kernels::matmul(previous_embedding, markov_w2_, markov_bias, options.context));
        FIREFLY_TRY(kernels::add_inplace(logits, markov_bias, options.context));

        if (options.auxiliary_output != nullptr && confidence_enabled_ && confidence_weight_.data() != nullptr)
        {
            const int batch = input.input_ids.shape()[0];
            const int sequence = input.input_ids.shape()[1];
            Tensor confidence_input = FIREFLY_TRY(Tensor::create(
                {static_cast<int64_t>(batch) * sequence, config_.hidden_size + markov_rank_}, config_.dtype,
                Device::CUDA, options.context));
            const size_t hidden_bytes = static_cast<size_t>(config_.hidden_size) * dtype_size(config_.dtype);
            const size_t rank_bytes = static_cast<size_t>(markov_rank_) * dtype_size(config_.dtype);
            FIREFLY_TRY(device::check_cuda(
                cudaMemcpy2DAsync(confidence_input.data(),
                                  (config_.hidden_size + markov_rank_) * dtype_size(config_.dtype),
                                  hidden_states.data(), hidden_bytes, hidden_bytes, batch * sequence,
                                  cudaMemcpyDeviceToDevice, options.context.stream()),
                "copy DSpark confidence hidden states"));
            FIREFLY_TRY(device::check_cuda(
                cudaMemcpy2DAsync(static_cast<std::byte*>(confidence_input.data()) + hidden_bytes,
                                  (config_.hidden_size + markov_rank_) * dtype_size(config_.dtype),
                                  previous_embedding.data(), rank_bytes, rank_bytes, batch * sequence,
                                  cudaMemcpyDeviceToDevice, options.context.stream()),
                "copy DSpark confidence Markov embeddings"));
            Tensor confidence = FIREFLY_TRY(Tensor::create(
                {static_cast<int64_t>(batch) * sequence, 1}, config_.dtype, Device::CUDA, options.context));
            FIREFLY_TRY(kernels::matmul(confidence_input, confidence_weight_, confidence, options.context));
            if (config_.dtype == DType::BF16)
                sigmoid_kernel<__nv_bfloat16><<<1, 128, 0, options.context.stream()>>>(
                    static_cast<__nv_bfloat16*>(confidence.data()), static_cast<const __nv_bfloat16*>(confidence_bias_.data()),
                    static_cast<int>(confidence.numel()));
            else
                sigmoid_kernel<half><<<1, 128, 0, options.context.stream()>>>(static_cast<half*>(confidence.data()),
                                                                                static_cast<const half*>(confidence_bias_.data()),
                                                                                static_cast<int>(confidence.numel()));
            FIREFLY_TRY(device::check_cuda(cudaGetLastError(), "launch DSpark confidence sigmoid"));
            *options.auxiliary_output = std::move(confidence);
        }
    }
    return logits;
}

Result<SpeculativeProposal> Model::propose(const SpeculativeProposalInput& input)
{
    if (input.token_count <= 0 || input.token_count > block_size_)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "Qwen3 DSpark proposal token count is outside the configured block size"});
    if (input.anchor_token < 0 || input.anchor_token >= config_.vocab_size)
        return unexpected(Error{ErrorCode::InvalidArgument, "Qwen3 DSpark anchor token is outside the vocabulary"});

    std::vector<int> host_ids(input.token_count, mask_token_id_);
    host_ids.front() = input.anchor_token;
    Tensor input_ids = FIREFLY_TRY(Tensor::create({1, input.token_count}, DType::I32, Device::CUDA,
                                                  input.context));
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(input_ids.data(), host_ids.data(), host_ids.size() * sizeof(int), cudaMemcpyHostToDevice,
                        input.context.stream()),
        "copy Qwen3 DSpark proposal input tokens"));
    Tensor empty_context;
    ModelInput model_input{input_ids, empty_context, {}};
    const int block_length = input.token_count;
    Tensor hidden_states;
    Tensor context_bias;
    ForwardOptions backbone_options{.context = input.context};
    backbone_options.return_all_logits = true;
    backbone_options.causal_attention = false;
    backbone_options.hidden_state_output = &hidden_states;
    if (input.target_context != nullptr && context_projection_.data() != nullptr)
    {
        Tensor projected_input = Tensor::from_external(
            input.target_context->data(),
            {1, static_cast<int64_t>(target_layer_ids_.size()) * config_.hidden_size},
            input.target_context->dtype(), Device::CUDA);
        context_bias = FIREFLY_TRY(Tensor::create({1, 1, config_.hidden_size}, config_.dtype, Device::CUDA,
                                                  input.context));
        FIREFLY_TRY(kernels::matmul(projected_input, context_projection_, context_bias, input.context));
        FIREFLY_TRY(kernels::rms_norm(context_bias, hidden_norm_, context_bias, config_.rms_norm_eps,
                                      input.context));
        backbone_options.attention_prefix = &context_bias;
    }
    Tensor base_logits = FIREFLY_TRY_CONTEXT(backbone_->forward(model_input, backbone_options),
                                             "execute Qwen3 DSpark proposal backbone");
    Tensor proposal_tokens = FIREFLY_TRY(Tensor::create({block_length}, DType::I32, Device::CUDA,
                                                        input.context));
    Tensor confidence;
    if (confidence_enabled_ && confidence_weight_.data() != nullptr)
        confidence = FIREFLY_TRY(Tensor::create({block_length}, config_.dtype, Device::CUDA, input.context));
    Tensor previous_token = Tensor::from_external(input_ids.data(), {1, 1}, DType::I32, Device::CUDA);
    for (int position = 0; position < block_length; ++position)
    {
        Tensor previous_embedding = FIREFLY_TRY(Tensor::create({1, 1, markov_rank_}, config_.dtype, Device::CUDA,
                                                               input.context));
        FIREFLY_TRY(kernels::embedding_lookup(previous_token, markov_w1_, previous_embedding, input.context));
        Tensor markov_bias = FIREFLY_TRY(Tensor::create({1, 1, config_.vocab_size}, config_.dtype, Device::CUDA,
                                                        input.context));
        FIREFLY_TRY(kernels::matmul(previous_embedding, markov_w2_, markov_bias, input.context));
        const size_t row_bytes = static_cast<size_t>(config_.vocab_size) * dtype_size(config_.dtype);
        Tensor position_logits = Tensor::from_external(
            static_cast<std::byte*>(base_logits.data()) + static_cast<size_t>(position) * row_bytes,
            {1, config_.vocab_size}, config_.dtype, Device::CUDA);
        FIREFLY_TRY(markov_bias.reshape({1, config_.vocab_size}));
        FIREFLY_TRY(kernels::add_inplace(position_logits, markov_bias, input.context));
        Tensor position_token = Tensor::from_external(
            static_cast<int*>(proposal_tokens.data()) + position, {1}, DType::I32, Device::CUDA);
        FIREFLY_TRY(kernels::argmax(position_logits, position_token, input.context));
        if (confidence.data() != nullptr)
        {
            Tensor confidence_input = FIREFLY_TRY(Tensor::create(
                {1, 1, config_.hidden_size + markov_rank_}, config_.dtype, Device::CUDA, input.context));
            const size_t hidden_bytes = static_cast<size_t>(config_.hidden_size) * dtype_size(config_.dtype);
            const size_t rank_bytes = static_cast<size_t>(markov_rank_) * dtype_size(config_.dtype);
            FIREFLY_TRY(device::check_cuda(
                cudaMemcpyAsync(confidence_input.data(),
                                static_cast<const std::byte*>(hidden_states.data()) + position * hidden_bytes,
                                hidden_bytes, cudaMemcpyDeviceToDevice, input.context.stream()),
                "copy Qwen3 DSpark proposal hidden state"));
            FIREFLY_TRY(device::check_cuda(
                cudaMemcpyAsync(static_cast<std::byte*>(confidence_input.data()) + hidden_bytes,
                                previous_embedding.data(), rank_bytes, cudaMemcpyDeviceToDevice,
                                input.context.stream()),
                "copy Qwen3 DSpark proposal Markov embedding"));
            Tensor position_confidence = Tensor::from_external(
                static_cast<std::byte*>(confidence.data()) + position * dtype_size(config_.dtype),
                {1, 1, 1}, config_.dtype, Device::CUDA);
            FIREFLY_TRY(kernels::matmul(confidence_input, confidence_weight_, position_confidence, input.context));
            if (config_.dtype == DType::BF16)
                sigmoid_kernel<__nv_bfloat16><<<1, 1, 0, input.context.stream()>>>(
                    static_cast<__nv_bfloat16*>(position_confidence.data()),
                    static_cast<const __nv_bfloat16*>(confidence_bias_.data()), 1);
            else
                sigmoid_kernel<half><<<1, 1, 0, input.context.stream()>>>(
                    static_cast<half*>(position_confidence.data()),
                    static_cast<const half*>(confidence_bias_.data()), 1);
            FIREFLY_TRY(device::check_cuda(cudaGetLastError(), "launch Qwen3 DSpark proposal confidence sigmoid"));
        }
        previous_token = Tensor::from_external(position_token.data(), {1, 1}, DType::I32, Device::CUDA);
    }
    return SpeculativeProposal{.token_ids = std::move(proposal_tokens), .confidence = std::move(confidence)};
}

}  // namespace firefly::model::qwen3_dspark

namespace firefly::model
{
void register_qwen3_dspark_models(ModelRegistry& registry)
{
    registry.register_factory("Qwen3DSparkModel", [](const ModelDescriptor& descriptor)
                              { return qwen3_dspark::Model::create(descriptor); });
}
}  // namespace firefly::model
