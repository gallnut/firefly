#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "firefly/core/logging.h"
#include "firefly/device/error.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/attention/hybrid_attention.h"
#include "firefly/kernels/cache/kv_cache.h"
#include "firefly/kernels/linear_attention/gated_delta_net.h"
#include "firefly/kernels/transformer/embedding.h"
#include "firefly/kernels/transformer/linear.h"
#include "firefly/kernels/transformer/residual.h"
#include "firefly/kernels/transformer/rms_norm.h"
#include "firefly/kernels/transformer/swiglu.h"
#include "firefly/model/qwen3_5/model.h"
#include "firefly/model/registry.h"

namespace firefly::model::qwen3_5
{
namespace
{
struct RuntimeSnapshot final : SpeculativeRuntimeState
{
    int slot = -1;
    std::vector<Tensor> convolution_states;
    std::vector<Tensor> recurrent_states;
};

Result<Config> parse_config(const ModelDescriptor& descriptor)
{
    Config config;
    config.base = descriptor.config;
    try
    {
        const auto root = nlohmann::json::parse(descriptor.raw_config);
        const auto& text = root.contains("text_config") ? root.at("text_config") : root;
        config.layer_types = text.at("layer_types").get<std::vector<std::string>>();
        config.full_attention_interval = text.value("full_attention_interval", 4);
        config.linear_conv_kernel_dim = text.value("linear_conv_kernel_dim", 4);
        config.linear_key_head_dim = text.value("linear_key_head_dim", 128);
        config.linear_num_key_heads = text.value("linear_num_key_heads", 16);
        config.linear_num_value_heads = text.value("linear_num_value_heads", 16);
        config.linear_value_head_dim = text.value("linear_value_head_dim", 128);
        const auto& rope = text.value("rope_parameters", nlohmann::json::object());
        config.partial_rotary_factor =
            rope.value("partial_rotary_factor", text.value("partial_rotary_factor", 0.25));
        config.attention_output_gate = text.value("attention_output_gate", true);
        config.tie_word_embeddings = text.value("tie_word_embeddings", true);
    }
    catch (const nlohmann::json::exception& error)
    {
        return unexpected(Error{ErrorCode::Parse, "failed to parse Qwen3.5 configuration: " +
                                                      std::string(error.what())});
    }

    if (static_cast<int>(config.layer_types.size()) != config.base.num_hidden_layers)
        return unexpected(Error{ErrorCode::Model, "Qwen3.5 layer_types does not match num_hidden_layers"});
    if (config.linear_num_key_heads != config.linear_num_value_heads ||
        config.linear_key_head_dim != config.linear_value_head_dim)
        return unexpected(Error{ErrorCode::Model,
                                "Firefly currently requires matching Qwen3.5 linear-attention head shapes"});
    if (config.linear_key_head_dim != 128)
        return unexpected(Error{ErrorCode::Model,
                                "Firefly currently supports Qwen3.5 linear head dimension 128"});
    return config;
}

Result<Tensor> take(std::unordered_map<std::string, Tensor>& weights, const std::string& name)
{
    auto found = weights.find(name);
    if (found == weights.end())
        return unexpected(Error{ErrorCode::NotFound, "missing Qwen3.5 weight: " + name});
    return std::move(found->second);
}

template <typename... Tensors>
Result<Tensor> fuse_projection_weights(Tensors&&... source_weights)
{
    std::array<Tensor, sizeof...(Tensors)> weights{std::forward<Tensors>(source_weights)...};
    static_assert(sizeof...(Tensors) > 0);
    const DType dtype = weights.front().dtype();
    const int64_t input_width = weights.front().shape()[1];
    int64_t output_width = 0;
    for (const Tensor& weight : weights)
    {
        if (weight.shape().size() != 2 || weight.shape()[1] != input_width || weight.dtype() != dtype)
            return unexpected(Error{ErrorCode::Model,
                                    "Qwen3.5 linear-attention projection weights are incompatible"});
        output_width += weight.shape()[0];
    }
    Tensor fused = FIREFLY_TRY(Tensor::create({output_width, input_width}, dtype, Device::CUDA));
    auto* destination = static_cast<std::byte*>(fused.data());
    size_t offset = 0;
    for (const Tensor& weight : weights)
    {
        const cudaError_t status = cudaMemcpy(destination + offset, weight.data(), weight.nbytes(),
                                               cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
            return unexpected(device::cuda_error(status, "fuse Qwen3.5 projection weights"));
        offset += weight.nbytes();
    }
    return fused;
}

}  // namespace

Model::Model(Config config)
    : config_(std::move(config)),
      gated_delta_net_workspace_(
          std::make_unique<firefly::kernels::linear_attention::GatedDeltaNetWorkspace>())
{
    layers_.resize(config_.base.num_hidden_layers);
    for (int index = 0; index < config_.base.num_hidden_layers; ++index)
        layers_[index].full_attention = config_.layer_types[index] == "full_attention";
}

Result<std::unique_ptr<Model>> Model::create(const ModelDescriptor& descriptor)
{
    Config config = FIREFLY_TRY(parse_config(descriptor));
    return std::unique_ptr<Model>(new Model(std::move(config)));
}

Model::~Model() = default;

bool Model::accepts_weight(std::string_view name) const
{
    return name.starts_with("model.language_model.");
}

bool Model::retains_source_weight(std::string_view name) const
{
    return !name.ends_with("linear_attn.in_proj_qkv.weight") &&
           !name.ends_with("linear_attn.in_proj_z.weight") &&
           !name.ends_with("linear_attn.in_proj_a.weight") &&
           !name.ends_with("linear_attn.in_proj_b.weight") && !name.ends_with("mlp.gate_proj.weight") &&
           !name.ends_with("mlp.up_proj.weight");
}

ModelRuntimeRequirements Model::runtime_requirements() const
{
    const int full_attention_layers = std::ranges::count(config_.layer_types, "full_attention");
    int       prefill_chunk_limit = 768;
    if (const char* environment = std::getenv("FIREFLY_QWEN35_PREFILL_CHUNK_LIMIT"))
        prefill_chunk_limit = std::atoi(environment);
    return {
        .kv_cache_layer_count = full_attention_layers,
        .kv_cache_head_count = config_.base.num_key_value_heads,
        .kv_cache_head_dim = config_.base.head_dim,
        .prefill_chunk_limit = prefill_chunk_limit,
        .sequence_state = true,
        .prefix_cache = false,
        .cuda_graph = false,
    };
}

Status Model::initialize_runtime(int max_sequence_slots, const device::Context& context)
{
    if (max_sequence_slots <= 0)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "Qwen3.5 requires at least one sequence-state slot"});
    max_sequence_slots_ = max_sequence_slots;
    convolution_states_.clear();
    recurrent_states_.clear();
    const int convolution_channels = config_.linear_num_key_heads * config_.linear_key_head_dim * 2 +
                                     config_.linear_num_value_heads * config_.linear_value_head_dim;
    for (const auto& layer : layers_)
    {
        if (layer.full_attention) continue;
        convolution_states_.emplace_back(FIREFLY_TRY(Tensor::create(
            {max_sequence_slots, convolution_channels, config_.linear_conv_kernel_dim}, config_.base.dtype,
            Device::CUDA, context)));
        recurrent_states_.emplace_back(FIREFLY_TRY(Tensor::create(
            {max_sequence_slots, config_.linear_num_value_heads, config_.linear_key_head_dim,
             config_.linear_value_head_dim},
            config_.base.dtype, Device::CUDA, context)));
    }
    FIREFLY_TRY(reset_runtime(context));
    FIREFLY_LOG_INFO("model", "Qwen3.5 sequence state allocated slots={} linear_layers={}", max_sequence_slots,
                     recurrent_states_.size());
    return {};
}

Status Model::reset_runtime(const device::Context& context)
{
    for (auto& state : convolution_states_)
        FIREFLY_TRY(device::check_cuda(cudaMemsetAsync(state.data(), 0, state.nbytes(), context.stream()),
                                       "reset Qwen3.5 convolution state"));
    for (auto& state : recurrent_states_)
        FIREFLY_TRY(device::check_cuda(cudaMemsetAsync(state.data(), 0, state.nbytes(), context.stream()),
                                       "reset Qwen3.5 recurrent state"));
    return {};
}

Result<std::unique_ptr<SpeculativeRuntimeState>> Model::snapshot_speculative_runtime(
    int state_slot, const device::Context& context)
{
    if (state_slot < 0 || state_slot >= max_sequence_slots_)
        return unexpected(Error{ErrorCode::InvalidArgument, "invalid Qwen3.5 runtime snapshot slot"});
    auto snapshot = std::make_unique<RuntimeSnapshot>();
    snapshot->slot = state_slot;
    snapshot->convolution_states.reserve(convolution_states_.size());
    snapshot->recurrent_states.reserve(recurrent_states_.size());
    auto copy_slot = [&](const Tensor& state) -> Result<Tensor>
    {
        std::vector<int64_t> shape = state.shape();
        shape.front() = 1;
        Tensor copy = FIREFLY_TRY(Tensor::create(shape, state.dtype(), Device::CUDA, context));
        const size_t slot_bytes = state.nbytes() / max_sequence_slots_;
        cudaMemcpyAsync(copy.data(), static_cast<const std::byte*>(state.data()) + state_slot * slot_bytes,
                        slot_bytes, cudaMemcpyDeviceToDevice, context.stream());
        return copy;
    };
    for (const auto& state : convolution_states_)
        snapshot->convolution_states.push_back(FIREFLY_TRY(copy_slot(state)));
    for (const auto& state : recurrent_states_)
        snapshot->recurrent_states.push_back(FIREFLY_TRY(copy_slot(state)));
    FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(context.stream()),
                                   "synchronize Qwen3.5 runtime snapshot"));
    return std::unique_ptr<SpeculativeRuntimeState>(std::move(snapshot));
}

Status Model::restore_speculative_runtime(std::unique_ptr<SpeculativeRuntimeState> state,
                                          const device::Context& context)
{
    auto* snapshot = dynamic_cast<RuntimeSnapshot*>(state.get());
    if (snapshot == nullptr || snapshot->slot < 0 || snapshot->slot >= max_sequence_slots_ ||
        snapshot->convolution_states.size() != convolution_states_.size() ||
        snapshot->recurrent_states.size() != recurrent_states_.size())
        return unexpected(Error{ErrorCode::InvalidArgument, "invalid Qwen3.5 runtime snapshot"});
    for (size_t index = 0; index < convolution_states_.size(); ++index)
    {
        const size_t slot_bytes = snapshot->convolution_states[index].nbytes();
        cudaMemcpyAsync(static_cast<std::byte*>(convolution_states_[index].data()) +
                            snapshot->slot * slot_bytes,
                        snapshot->convolution_states[index].data(), slot_bytes, cudaMemcpyDeviceToDevice,
                        context.stream());
    }
    for (size_t index = 0; index < recurrent_states_.size(); ++index)
    {
        const size_t slot_bytes = snapshot->recurrent_states[index].nbytes();
        cudaMemcpyAsync(static_cast<std::byte*>(recurrent_states_[index].data()) +
                            snapshot->slot * slot_bytes,
                        snapshot->recurrent_states[index].data(), slot_bytes, cudaMemcpyDeviceToDevice,
                        context.stream());
    }
    return {};
}

Status Model::load_weights(std::unordered_map<std::string, Tensor>& weights)
{
    const std::string model_prefix = "model.language_model.";
    token_embeddings_ = FIREFLY_TRY(take(weights, model_prefix + "embed_tokens.weight"));
    config_.base.dtype = token_embeddings_.dtype();
    FIREFLY_TRY(require_float16_or_bfloat16(config_.base.dtype, "Qwen3.5"));

    for (int index = 0; index < config_.base.num_hidden_layers; ++index)
    {
        auto& layer = layers_[index];
        const std::string prefix = model_prefix + "layers." + std::to_string(index) + ".";
        if (layer.full_attention)
        {
            layer.attention.query_projection = FIREFLY_TRY(take(weights, prefix + "self_attn.q_proj.weight"));
            layer.attention.key_projection = FIREFLY_TRY(take(weights, prefix + "self_attn.k_proj.weight"));
            layer.attention.value_projection = FIREFLY_TRY(take(weights, prefix + "self_attn.v_proj.weight"));
            layer.attention.output_projection = FIREFLY_TRY(take(weights, prefix + "self_attn.o_proj.weight"));
            layer.attention.query_norm = FIREFLY_TRY(take(weights, prefix + "self_attn.q_norm.weight"));
            layer.attention.key_norm = FIREFLY_TRY(take(weights, prefix + "self_attn.k_norm.weight"));
        }
        else
        {
            Tensor projection_qkv = FIREFLY_TRY(take(weights, prefix + "linear_attn.in_proj_qkv.weight"));
            Tensor projection_z = FIREFLY_TRY(take(weights, prefix + "linear_attn.in_proj_z.weight"));
            Tensor projection_a = FIREFLY_TRY(take(weights, prefix + "linear_attn.in_proj_a.weight"));
            Tensor projection_b = FIREFLY_TRY(take(weights, prefix + "linear_attn.in_proj_b.weight"));
            layer.linear_attention.fused_projection = FIREFLY_TRY(fuse_projection_weights(
                std::move(projection_qkv), std::move(projection_z), std::move(projection_a),
                std::move(projection_b)));
            layer.linear_attention.convolution_weight =
                FIREFLY_TRY(take(weights, prefix + "linear_attn.conv1d.weight"));
            layer.linear_attention.decay_log = FIREFLY_TRY(take(weights, prefix + "linear_attn.A_log"));
            layer.linear_attention.decay_bias = FIREFLY_TRY(take(weights, prefix + "linear_attn.dt_bias"));
            layer.linear_attention.norm_weight = FIREFLY_TRY(take(weights, prefix + "linear_attn.norm.weight"));
            layer.linear_attention.output_projection =
                FIREFLY_TRY(take(weights, prefix + "linear_attn.out_proj.weight"));
        }
        Tensor gate_projection = FIREFLY_TRY(take(weights, prefix + "mlp.gate_proj.weight"));
        Tensor up_projection = FIREFLY_TRY(take(weights, prefix + "mlp.up_proj.weight"));
        layer.mlp.fused_projection = FIREFLY_TRY(
            fuse_projection_weights(std::move(gate_projection), std::move(up_projection)));
        layer.mlp.down_projection = FIREFLY_TRY(take(weights, prefix + "mlp.down_proj.weight"));
        layer.input_norm = FIREFLY_TRY(take(weights, prefix + "input_layernorm.weight"));
        layer.post_attention_norm = FIREFLY_TRY(take(weights, prefix + "post_attention_layernorm.weight"));
    }
    final_norm_ = FIREFLY_TRY(take(weights, model_prefix + "norm.weight"));
    FIREFLY_LOG_INFO("model", "Qwen3.5 weights mapped layers={} full_attention_layers={}", layers_.size(),
                     runtime_requirements().kv_cache_layer_count);
    return {};
}

Result<Tensor> Model::forward(const ModelInput& input, const ForwardOptions& options)
{
    if (!input.state_slots || max_sequence_slots_ == 0)
        return unexpected(Error{ErrorCode::InvalidState, "Qwen3.5 runtime state has not been initialized"});
    const int batch_size = input.input_ids.shape()[0];
    const int sequence_length = input.input_ids.shape()[1];
    const int hidden_size = config_.base.hidden_size;
    const int query_heads = config_.base.num_attention_heads;
    const int kv_heads = config_.base.num_key_value_heads;
    const int head_dim = config_.base.head_dim;
    const int linear_heads = config_.linear_num_value_heads;
    const int linear_width = linear_heads * config_.linear_value_head_dim;
    const int mixed_width = linear_heads * config_.linear_key_head_dim * 2 + linear_width;
    const int* context_lengths = input.context_lens.data_as<const int>();
    auto& kv_cache = input.kv_cache;

    Tensor hidden_states = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, hidden_size},
                                                      config_.base.dtype, Device::CUDA, options.context));
    Tensor norm_output =
        FIREFLY_TRY(Tensor::create(hidden_states.shape(), config_.base.dtype, Device::CUDA, options.context));
    FIREFLY_TRY(firefly::kernels::embedding_lookup(input.input_ids, token_embeddings_, hidden_states,
                                                   options.context));
    FIREFLY_TRY(firefly::kernels::rms_norm_zero_centered(
        hidden_states, layers_.front().input_norm, norm_output, config_.base.rms_norm_eps, options.context));

    Tensor mixer_projection = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, hidden_size},
                                                         config_.base.dtype, Device::CUDA, options.context));
    Tensor mixer_output = FIREFLY_TRY(
        Tensor::create(mixer_projection.shape(), config_.base.dtype, Device::CUDA, options.context));
    Tensor mlp_projection = FIREFLY_TRY(Tensor::create(
        {batch_size, sequence_length, config_.base.intermediate_size * 2}, config_.base.dtype, Device::CUDA,
        options.context));
    Tensor mlp_intermediate = FIREFLY_TRY(Tensor::create(
        {batch_size, sequence_length, config_.base.intermediate_size}, config_.base.dtype, Device::CUDA,
        options.context));
    Tensor mlp_output = FIREFLY_TRY(
        Tensor::create(mixer_projection.shape(), config_.base.dtype, Device::CUDA, options.context));

    Tensor projected_query = FIREFLY_TRY(Tensor::create(
        {batch_size, sequence_length, query_heads * head_dim * 2}, config_.base.dtype, Device::CUDA,
        options.context));
    Tensor query = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, query_heads, head_dim},
                                              config_.base.dtype, Device::CUDA, options.context));
    Tensor attention_gate =
        FIREFLY_TRY(Tensor::create(query.shape(), config_.base.dtype, Device::CUDA, options.context));
    Tensor key = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, kv_heads, head_dim},
                                            config_.base.dtype, Device::CUDA, options.context));
    Tensor value = FIREFLY_TRY(Tensor::create(key.shape(), config_.base.dtype, Device::CUDA, options.context));
    Tensor attention_output = FIREFLY_TRY(Tensor::create(
        {batch_size, sequence_length, query_heads * head_dim}, config_.base.dtype, Device::CUDA, options.context));

    const int fused_linear_width = mixed_width + linear_width + linear_heads * 2;
    Tensor linear_projection = FIREFLY_TRY(Tensor::create(
        {batch_size, sequence_length, fused_linear_width}, config_.base.dtype, Device::CUDA, options.context));
    const std::vector<int64_t> linear_strides{static_cast<int64_t>(sequence_length) * fused_linear_width,
                                               fused_linear_width, 1};
    auto* linear_base = static_cast<std::byte*>(linear_projection.data());
    const size_t scalar_bytes = dtype_size(config_.base.dtype);
    Tensor projected_qkv = FIREFLY_TRY(Tensor::from_external(
        linear_base, {batch_size, sequence_length, mixed_width}, linear_strides, config_.base.dtype, Device::CUDA));
    Tensor linear_gate = FIREFLY_TRY(Tensor::from_external(
        linear_base + static_cast<size_t>(mixed_width) * scalar_bytes,
        {batch_size, sequence_length, linear_width}, linear_strides, config_.base.dtype, Device::CUDA));
    Tensor linear_decay = FIREFLY_TRY(Tensor::from_external(
        linear_base + static_cast<size_t>(mixed_width + linear_width) * scalar_bytes,
        {batch_size, sequence_length, linear_heads}, linear_strides, config_.base.dtype, Device::CUDA));
    Tensor linear_beta = FIREFLY_TRY(Tensor::from_external(
        linear_base + static_cast<size_t>(mixed_width + linear_width + linear_heads) * scalar_bytes,
        {batch_size, sequence_length, linear_heads}, linear_strides, config_.base.dtype, Device::CUDA));
    Tensor mixed_qkv = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, mixed_width}, config_.base.dtype,
                                                  Device::CUDA, options.context));
    Tensor linear_output =
        FIREFLY_TRY(Tensor::create(linear_gate.shape(), config_.base.dtype, Device::CUDA, options.context));

    int full_attention_index = 0;
    int linear_attention_index = 0;
    size_t hidden_layer_output_index = 0;
    const auto full_attention_backend = firefly::kernels::get_attention_backend();
    for (int layer_index = 0; layer_index < config_.base.num_hidden_layers; ++layer_index)
    {
        const auto& layer = layers_[layer_index];
        if (layer.full_attention)
        {
            FIREFLY_TRY(firefly::kernels::matmul(norm_output, layer.attention.query_projection, projected_query,
                                                 options.context));
            FIREFLY_TRY(firefly::kernels::matmul(norm_output, layer.attention.key_projection, key,
                                                 options.context));
            FIREFLY_TRY(firefly::kernels::matmul(norm_output, layer.attention.value_projection, value,
                                                 options.context));
            FIREFLY_TRY(firefly::kernels::prepare_full_attention(
                projected_query, query, attention_gate, key, layer.attention.query_norm, layer.attention.key_norm,
                context_lengths, sequence_length,
                static_cast<int>(head_dim * config_.partial_rotary_factor), config_.base.rope_theta,
                config_.base.rms_norm_eps, options.context));

            FIREFLY_TRY(firefly::kernels::append_paged_kv(key, value, kv_cache.key_layers[full_attention_index],
                                              kv_cache.value_layers[full_attention_index], kv_cache.block_table,
                                              context_lengths, kv_cache.max_blocks_per_sequence, nullptr,
                                              options.context));
            if (!context_lengths || !kv_cache.block_table)
            {
                FIREFLY_TRY(firefly::kernels::attention(query, key, value, attention_output,
                                                        {.kv_head_count = kv_heads}, options.context));
            }
            else if (options.min_context_len == 0 && sequence_length > 1)
            {
                FIREFLY_TRY(firefly::kernels::attention(query, key, value, attention_output,
                                             {.backend = full_attention_backend,
                                              .kv_head_count = kv_heads,
                                              .max_context_blocks = kv_cache.max_blocks_per_sequence},
                                             options.context));
            }
            else
            {
                int prefill_context_length = -1;
                if (sequence_length > 1 && options.min_context_len >= 0)
                    prefill_context_length = options.min_context_len;
                else if (sequence_length == 1 && options.max_decode_context_len > 0)
                    prefill_context_length = options.max_decode_context_len;
                FIREFLY_TRY(firefly::kernels::attention(
                    query, kv_cache.key_layers[full_attention_index], kv_cache.value_layers[full_attention_index],
                    attention_output,
                    {.backend = full_attention_backend,
                     .block_table = kv_cache.block_table,
                     .context_lengths = context_lengths,
                     .kv_head_count = kv_heads,
                     .max_context_blocks = kv_cache.max_blocks_per_sequence,
                     .max_decode_context_length = options.max_decode_context_len,
                     .prefill_context_length = prefill_context_length,
                     .prefer_split_decode = options.prefer_split_decode},
                    options.context));
            }
            FIREFLY_TRY(firefly::kernels::apply_attention_gate(attention_output, attention_gate,
                                                               options.context));
            FIREFLY_TRY(firefly::kernels::matmul(attention_output, layer.attention.output_projection,
                                                 mixer_output, options.context));
            ++full_attention_index;
        }
        else
        {
            FIREFLY_TRY(firefly::kernels::matmul(norm_output, layer.linear_attention.fused_projection,
                                                 linear_projection, options.context));
            FIREFLY_TRY(firefly::kernels::linear_attention::causal_convolution(
                projected_qkv, layer.linear_attention.convolution_weight,
                convolution_states_[linear_attention_index], input.state_slots, context_lengths, mixed_qkv,
                options.context));
            FIREFLY_TRY(firefly::kernels::linear_attention::gated_delta_net(
                mixed_qkv, linear_gate, linear_decay, linear_beta, layer.linear_attention.decay_log,
                layer.linear_attention.decay_bias, layer.linear_attention.norm_weight,
                recurrent_states_[linear_attention_index], input.state_slots, context_lengths, linear_output,
                gated_delta_net_workspace_.get(), config_.base.rms_norm_eps, options.context));
            FIREFLY_TRY(firefly::kernels::matmul(linear_output, layer.linear_attention.output_projection,
                                                 mixer_output, options.context));
            ++linear_attention_index;
        }

        FIREFLY_TRY(firefly::kernels::add_rms_norm_zero_centered(
            hidden_states, mixer_output, layer.post_attention_norm, norm_output, config_.base.rms_norm_eps,
            options.context));
        FIREFLY_TRY(firefly::kernels::matmul(norm_output, layer.mlp.fused_projection, mlp_projection,
                                             options.context));
        FIREFLY_TRY(firefly::kernels::swiglu_fused(mlp_projection, mlp_intermediate, options.context));
        FIREFLY_TRY(firefly::kernels::matmul(mlp_intermediate, layer.mlp.down_projection, mlp_output,
                                             options.context));
        if (layer_index + 1 < config_.base.num_hidden_layers)
            FIREFLY_TRY(firefly::kernels::add_rms_norm_zero_centered(
                hidden_states, mlp_output, layers_[layer_index + 1].input_norm, norm_output,
                config_.base.rms_norm_eps, options.context));
        else
            FIREFLY_TRY(firefly::kernels::add_inplace(hidden_states, mlp_output, options.context));

        if (options.layered_hidden_state_output != nullptr &&
            hidden_layer_output_index < options.hidden_state_layers.size() &&
            options.hidden_state_layers[hidden_layer_output_index] == layer_index)
        {
            if (options.layered_hidden_state_output->data() == nullptr)
            {
                *options.layered_hidden_state_output = FIREFLY_TRY(Tensor::create(
                    {static_cast<int64_t>(options.hidden_state_layers.size()), batch_size, sequence_length,
                     hidden_size},
                    config_.base.dtype, Device::CUDA, options.context));
            }
            const size_t layer_bytes = hidden_states.nbytes();
            cudaMemcpyAsync(static_cast<std::byte*>(options.layered_hidden_state_output->data()) +
                                hidden_layer_output_index * layer_bytes,
                            hidden_states.data(), layer_bytes, cudaMemcpyDeviceToDevice,
                            options.context.stream());
            ++hidden_layer_output_index;
        }
    }

    if (hidden_layer_output_index != options.hidden_state_layers.size())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "Qwen3.5 hidden-state layer selection is invalid"});

    if (options.hidden_state_output != nullptr)
        *options.hidden_state_output = FIREFLY_TRY(hidden_states.clone());
    if (!options.compute_logits) return Tensor();
    if (options.return_all_logits)
    {
        Tensor final_hidden = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, hidden_size},
                                                         config_.base.dtype, Device::CUDA, options.context));
        FIREFLY_TRY(firefly::kernels::rms_norm_zero_centered(
            hidden_states, final_norm_, final_hidden, config_.base.rms_norm_eps, options.context));
        Tensor logits = FIREFLY_TRY(Tensor::create({batch_size, sequence_length, config_.base.vocab_size},
                                                   config_.base.dtype, Device::CUDA, options.context));
        FIREFLY_TRY(firefly::kernels::matmul(final_hidden, token_embeddings_, logits, options.context));
        return logits;
    }
    Tensor last_hidden;
    Tensor compact_last_hidden;
    if (sequence_length == 1)
    {
        last_hidden = Tensor::from_external(hidden_states.data(), {batch_size, 1, hidden_size}, config_.base.dtype,
                                            Device::CUDA);
    }
    else
    {
        compact_last_hidden = FIREFLY_TRY(Tensor::create({batch_size, 1, hidden_size}, config_.base.dtype,
                                                         Device::CUDA, options.context));
        cudaMemcpy2DAsync(compact_last_hidden.data(), hidden_size * dtype_size(config_.base.dtype),
                          static_cast<std::byte*>(hidden_states.data()) +
                              (sequence_length - 1) * hidden_size * dtype_size(config_.base.dtype),
                          sequence_length * hidden_size * dtype_size(config_.base.dtype),
                          hidden_size * dtype_size(config_.base.dtype), batch_size, cudaMemcpyDeviceToDevice,
                          options.context.stream());
        last_hidden = Tensor::from_external(compact_last_hidden.data(), {batch_size, 1, hidden_size},
                                            config_.base.dtype, Device::CUDA);
    }
    Tensor final_hidden = FIREFLY_TRY(Tensor::create({batch_size, 1, hidden_size}, config_.base.dtype,
                                                     Device::CUDA, options.context));
    FIREFLY_TRY(firefly::kernels::rms_norm_zero_centered(last_hidden, final_norm_, final_hidden,
                                                         config_.base.rms_norm_eps, options.context));
    Tensor logits = FIREFLY_TRY(Tensor::create({batch_size, 1, config_.base.vocab_size}, config_.base.dtype,
                                               Device::CUDA, options.context));
    FIREFLY_TRY(firefly::kernels::matmul(final_hidden, token_embeddings_, logits, options.context));
    return logits;
}

}  // namespace firefly::model::qwen3_5

namespace firefly::model
{
void register_qwen3_5_models(ModelRegistry& registry)
{
    registry.register_factory("Qwen3_5ForConditionalGeneration", qwen3_5::Model::create);
}
}  // namespace firefly::model
