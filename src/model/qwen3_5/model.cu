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
Config parse_config(const ModelDescriptor& descriptor)
{
    Config config;
    config.base = descriptor.config;
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
    config.partial_rotary_factor = rope.value("partial_rotary_factor", text.value("partial_rotary_factor", 0.25));
    config.attention_output_gate = text.value("attention_output_gate", true);
    config.tie_word_embeddings = text.value("tie_word_embeddings", true);

    if (static_cast<int>(config.layer_types.size()) != config.base.num_hidden_layers)
        throw std::runtime_error("Qwen3.5 layer_types does not match num_hidden_layers");
    if (config.linear_num_key_heads != config.linear_num_value_heads ||
        config.linear_key_head_dim != config.linear_value_head_dim)
        throw std::runtime_error("Firefly currently requires matching Qwen3.5 linear-attention head shapes");
    if (config.linear_key_head_dim != 128)
        throw std::runtime_error("Firefly currently supports Qwen3.5 linear head dimension 128");
    return config;
}

Tensor take(std::unordered_map<std::string, Tensor>& weights, const std::string& name)
{
    auto found = weights.find(name);
    if (found == weights.end()) throw std::runtime_error("missing Qwen3.5 weight: " + name);
    return std::move(found->second);
}

template <typename... Tensors>
Tensor fuse_projection_weights(Tensors&&... source_weights)
{
    std::array<Tensor, sizeof...(Tensors)> weights{std::forward<Tensors>(source_weights)...};
    static_assert(sizeof...(Tensors) > 0);
    const DType dtype = weights.front().dtype();
    const int64_t input_width = weights.front().shape()[1];
    int64_t output_width = 0;
    for (const Tensor& weight : weights)
    {
        if (weight.shape().size() != 2 || weight.shape()[1] != input_width || weight.dtype() != dtype)
            throw std::runtime_error("Qwen3.5 linear-attention projection weights are incompatible");
        output_width += weight.shape()[0];
    }
    Tensor fused({output_width, input_width}, dtype, Device::CUDA);
    auto* destination = static_cast<std::byte*>(fused.data());
    size_t offset = 0;
    for (const Tensor& weight : weights)
    {
        const cudaError_t status = cudaMemcpy(destination + offset, weight.data(), weight.nbytes(),
                                               cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Qwen3.5 projection fusion failed: ") + cudaGetErrorString(status));
        offset += weight.nbytes();
    }
    return fused;
}

}  // namespace

Model::Model(const ModelDescriptor& descriptor)
    : config_(parse_config(descriptor)),
      gated_delta_net_workspace_(
          std::make_unique<firefly::kernels::linear_attention::GatedDeltaNetWorkspace>())
{
    layers_.resize(config_.base.num_hidden_layers);
    for (int index = 0; index < config_.base.num_hidden_layers; ++index)
        layers_[index].full_attention = config_.layer_types[index] == "full_attention";
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

void Model::initialize_runtime(int max_sequence_slots, const device::Context& context)
{
    if (max_sequence_slots <= 0) throw std::runtime_error("Qwen3.5 requires at least one sequence-state slot");
    max_sequence_slots_ = max_sequence_slots;
    convolution_states_.clear();
    recurrent_states_.clear();
    const int convolution_channels = config_.linear_num_key_heads * config_.linear_key_head_dim * 2 +
                                     config_.linear_num_value_heads * config_.linear_value_head_dim;
    for (const auto& layer : layers_)
    {
        if (layer.full_attention) continue;
        convolution_states_.emplace_back(
            Tensor({max_sequence_slots, convolution_channels, config_.linear_conv_kernel_dim}, config_.base.dtype,
                   Device::CUDA, context));
        recurrent_states_.emplace_back(
            Tensor({max_sequence_slots, config_.linear_num_value_heads, config_.linear_key_head_dim,
                    config_.linear_value_head_dim}, config_.base.dtype, Device::CUDA, context));
    }
    reset_runtime(context);
    FIREFLY_LOG_INFO("model", "Qwen3.5 sequence state allocated slots={} linear_layers={}", max_sequence_slots,
                     recurrent_states_.size());
}

void Model::reset_runtime(const device::Context& context)
{
    for (auto& state : convolution_states_) cudaMemsetAsync(state.data(), 0, state.nbytes(), context.stream());
    for (auto& state : recurrent_states_) cudaMemsetAsync(state.data(), 0, state.nbytes(), context.stream());
}

void Model::load_weights(std::unordered_map<std::string, Tensor>& weights)
{
    const std::string model_prefix = "model.language_model.";
    token_embeddings_ = take(weights, model_prefix + "embed_tokens.weight");
    config_.base.dtype = token_embeddings_.dtype();
    require_float16_or_bfloat16(config_.base.dtype, "Qwen3.5");

    for (int index = 0; index < config_.base.num_hidden_layers; ++index)
    {
        auto& layer = layers_[index];
        const std::string prefix = model_prefix + "layers." + std::to_string(index) + ".";
        if (layer.full_attention)
        {
            layer.attention.query_projection = take(weights, prefix + "self_attn.q_proj.weight");
            layer.attention.key_projection = take(weights, prefix + "self_attn.k_proj.weight");
            layer.attention.value_projection = take(weights, prefix + "self_attn.v_proj.weight");
            layer.attention.output_projection = take(weights, prefix + "self_attn.o_proj.weight");
            layer.attention.query_norm = take(weights, prefix + "self_attn.q_norm.weight");
            layer.attention.key_norm = take(weights, prefix + "self_attn.k_norm.weight");
        }
        else
        {
            layer.linear_attention.fused_projection = fuse_projection_weights(
                take(weights, prefix + "linear_attn.in_proj_qkv.weight"),
                take(weights, prefix + "linear_attn.in_proj_z.weight"),
                take(weights, prefix + "linear_attn.in_proj_a.weight"),
                take(weights, prefix + "linear_attn.in_proj_b.weight"));
            layer.linear_attention.convolution_weight = take(weights, prefix + "linear_attn.conv1d.weight");
            layer.linear_attention.decay_log = take(weights, prefix + "linear_attn.A_log");
            layer.linear_attention.decay_bias = take(weights, prefix + "linear_attn.dt_bias");
            layer.linear_attention.norm_weight = take(weights, prefix + "linear_attn.norm.weight");
            layer.linear_attention.output_projection = take(weights, prefix + "linear_attn.out_proj.weight");
        }
        layer.mlp.fused_projection = fuse_projection_weights(take(weights, prefix + "mlp.gate_proj.weight"),
                                                             take(weights, prefix + "mlp.up_proj.weight"));
        layer.mlp.down_projection = take(weights, prefix + "mlp.down_proj.weight");
        layer.input_norm = take(weights, prefix + "input_layernorm.weight");
        layer.post_attention_norm = take(weights, prefix + "post_attention_layernorm.weight");
    }
    final_norm_ = take(weights, model_prefix + "norm.weight");
    FIREFLY_LOG_INFO("model", "Qwen3.5 weights mapped layers={} full_attention_layers={}", layers_.size(),
                     runtime_requirements().kv_cache_layer_count);
}

Tensor Model::forward(const ModelInput& input, const ForwardOptions& options)
{
    if (!input.state_slots || max_sequence_slots_ == 0)
        throw std::runtime_error("Qwen3.5 runtime state has not been initialized");
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

    Tensor hidden_states({batch_size, sequence_length, hidden_size}, config_.base.dtype, Device::CUDA,
                         options.context);
    Tensor norm_output(hidden_states.shape(), config_.base.dtype, Device::CUDA, options.context);
    firefly::kernels::embedding_lookup(input.input_ids, token_embeddings_, hidden_states, options.context);
    firefly::kernels::rms_norm_zero_centered(hidden_states, layers_.front().input_norm, norm_output,
                                             config_.base.rms_norm_eps, options.context);

    Tensor mixer_projection({batch_size, sequence_length, hidden_size}, config_.base.dtype, Device::CUDA,
                            options.context);
    Tensor mixer_output(mixer_projection.shape(), config_.base.dtype, Device::CUDA, options.context);
    Tensor mlp_projection({batch_size, sequence_length, config_.base.intermediate_size * 2}, config_.base.dtype,
                          Device::CUDA, options.context);
    Tensor mlp_intermediate({batch_size, sequence_length, config_.base.intermediate_size}, config_.base.dtype,
                            Device::CUDA, options.context);
    Tensor mlp_output(mixer_projection.shape(), config_.base.dtype, Device::CUDA, options.context);

    Tensor projected_query({batch_size, sequence_length, query_heads * head_dim * 2}, config_.base.dtype,
                           Device::CUDA, options.context);
    Tensor query({batch_size, sequence_length, query_heads, head_dim}, config_.base.dtype, Device::CUDA,
                 options.context);
    Tensor attention_gate(query.shape(), config_.base.dtype, Device::CUDA, options.context);
    Tensor key({batch_size, sequence_length, kv_heads, head_dim}, config_.base.dtype, Device::CUDA, options.context);
    Tensor value(key.shape(), config_.base.dtype, Device::CUDA, options.context);
    Tensor attention_output({batch_size, sequence_length, query_heads * head_dim}, config_.base.dtype, Device::CUDA,
                            options.context);

    const int fused_linear_width = mixed_width + linear_width + linear_heads * 2;
    Tensor linear_projection({batch_size, sequence_length, fused_linear_width}, config_.base.dtype, Device::CUDA,
                             options.context);
    const std::vector<int64_t> linear_strides{static_cast<int64_t>(sequence_length) * fused_linear_width,
                                               fused_linear_width, 1};
    auto* linear_base = static_cast<std::byte*>(linear_projection.data());
    const size_t scalar_bytes = dtype_size(config_.base.dtype);
    Tensor projected_qkv = Tensor::from_external(linear_base, {batch_size, sequence_length, mixed_width},
                                                 linear_strides, config_.base.dtype, Device::CUDA);
    Tensor linear_gate = Tensor::from_external(linear_base + static_cast<size_t>(mixed_width) * scalar_bytes,
                                               {batch_size, sequence_length, linear_width}, linear_strides,
                                               config_.base.dtype, Device::CUDA);
    Tensor linear_decay = Tensor::from_external(
        linear_base + static_cast<size_t>(mixed_width + linear_width) * scalar_bytes,
        {batch_size, sequence_length, linear_heads}, linear_strides, config_.base.dtype, Device::CUDA);
    Tensor linear_beta = Tensor::from_external(
        linear_base + static_cast<size_t>(mixed_width + linear_width + linear_heads) * scalar_bytes,
        {batch_size, sequence_length, linear_heads}, linear_strides, config_.base.dtype, Device::CUDA);
    Tensor mixed_qkv({batch_size, sequence_length, mixed_width}, config_.base.dtype, Device::CUDA, options.context);
    Tensor linear_output(linear_gate.shape(), config_.base.dtype, Device::CUDA, options.context);

    int full_attention_index = 0;
    int linear_attention_index = 0;
    const auto full_attention_backend = firefly::kernels::get_attention_backend();
    for (int layer_index = 0; layer_index < config_.base.num_hidden_layers; ++layer_index)
    {
        const auto& layer = layers_[layer_index];
        if (layer.full_attention)
        {
            firefly::kernels::matmul(norm_output, layer.attention.query_projection, projected_query, options.context);
            firefly::kernels::matmul(norm_output, layer.attention.key_projection, key, options.context);
            firefly::kernels::matmul(norm_output, layer.attention.value_projection, value, options.context);
            firefly::kernels::prepare_full_attention(
                projected_query, query, attention_gate, key, layer.attention.query_norm, layer.attention.key_norm,
                context_lengths, sequence_length,
                static_cast<int>(head_dim * config_.partial_rotary_factor), config_.base.rope_theta,
                config_.base.rms_norm_eps, options.context);

            firefly::kernels::append_paged_kv(key, value, kv_cache.key_layers[full_attention_index],
                                              kv_cache.value_layers[full_attention_index], kv_cache.block_table,
                                              context_lengths, kv_cache.max_blocks_per_sequence, nullptr,
                                              options.context);
            if (!context_lengths || !kv_cache.block_table)
            {
                firefly::kernels::attention(query, key, value, attention_output,
                                             {.kv_head_count = kv_heads}, options.context);
            }
            else if (options.min_context_len == 0 && sequence_length > 1)
            {
                firefly::kernels::attention(query, key, value, attention_output,
                                             {.backend = full_attention_backend,
                                              .kv_head_count = kv_heads,
                                              .max_context_blocks = kv_cache.max_blocks_per_sequence},
                                             options.context);
            }
            else
            {
                int prefill_context_length = -1;
                if (sequence_length > 1 && options.min_context_len >= 0)
                    prefill_context_length = options.min_context_len;
                else if (sequence_length == 1 && options.max_decode_context_len > 0)
                    prefill_context_length = options.max_decode_context_len;
                firefly::kernels::attention(
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
                    options.context);
            }
            firefly::kernels::apply_attention_gate(attention_output, attention_gate, options.context);
            firefly::kernels::matmul(attention_output, layer.attention.output_projection, mixer_output,
                                     options.context);
            ++full_attention_index;
        }
        else
        {
            firefly::kernels::matmul(norm_output, layer.linear_attention.fused_projection, linear_projection,
                                     options.context);
            firefly::kernels::linear_attention::causal_convolution(
                projected_qkv, layer.linear_attention.convolution_weight,
                convolution_states_[linear_attention_index], input.state_slots, context_lengths, mixed_qkv,
                options.context);
            firefly::kernels::linear_attention::gated_delta_net(
                mixed_qkv, linear_gate, linear_decay, linear_beta, layer.linear_attention.decay_log,
                layer.linear_attention.decay_bias, layer.linear_attention.norm_weight,
                recurrent_states_[linear_attention_index], input.state_slots, context_lengths, linear_output,
                gated_delta_net_workspace_.get(), config_.base.rms_norm_eps, options.context);
            firefly::kernels::matmul(linear_output, layer.linear_attention.output_projection, mixer_output,
                                     options.context);
            ++linear_attention_index;
        }

        firefly::kernels::add_rms_norm_zero_centered(hidden_states, mixer_output, layer.post_attention_norm,
                                                     norm_output, config_.base.rms_norm_eps, options.context);
        firefly::kernels::matmul(norm_output, layer.mlp.fused_projection, mlp_projection, options.context);
        firefly::kernels::swiglu_fused(mlp_projection, mlp_intermediate, options.context);
        firefly::kernels::matmul(mlp_intermediate, layer.mlp.down_projection, mlp_output, options.context);
        if (layer_index + 1 < config_.base.num_hidden_layers)
            firefly::kernels::add_rms_norm_zero_centered(
                hidden_states, mlp_output, layers_[layer_index + 1].input_norm, norm_output,
                config_.base.rms_norm_eps, options.context);
        else
            firefly::kernels::add_inplace(hidden_states, mlp_output, options.context);
    }

    if (!options.compute_logits) return Tensor();
    Tensor last_hidden;
    Tensor compact_last_hidden;
    if (sequence_length == 1)
    {
        last_hidden = Tensor::from_external(hidden_states.data(), {batch_size, 1, hidden_size}, config_.base.dtype,
                                            Device::CUDA);
    }
    else
    {
        compact_last_hidden = Tensor({batch_size, 1, hidden_size}, config_.base.dtype, Device::CUDA, options.context);
        cudaMemcpy2DAsync(compact_last_hidden.data(), hidden_size * dtype_size(config_.base.dtype),
                          static_cast<std::byte*>(hidden_states.data()) +
                              (sequence_length - 1) * hidden_size * dtype_size(config_.base.dtype),
                          sequence_length * hidden_size * dtype_size(config_.base.dtype),
                          hidden_size * dtype_size(config_.base.dtype), batch_size, cudaMemcpyDeviceToDevice,
                          options.context.stream());
        last_hidden = Tensor::from_external(compact_last_hidden.data(), {batch_size, 1, hidden_size},
                                            config_.base.dtype, Device::CUDA);
    }
    Tensor final_hidden({batch_size, 1, hidden_size}, config_.base.dtype, Device::CUDA, options.context);
    firefly::kernels::rms_norm_zero_centered(last_hidden, final_norm_, final_hidden, config_.base.rms_norm_eps,
                                             options.context);
    Tensor logits({batch_size, 1, config_.base.vocab_size}, config_.base.dtype, Device::CUDA, options.context);
    firefly::kernels::matmul(final_hidden, token_embeddings_, logits, options.context);
    return logits;
}

}  // namespace firefly::model::qwen3_5

namespace firefly::model
{
void register_qwen3_5_models(ModelRegistry& registry)
{
    registry.register_factory("Qwen3_5ForConditionalGeneration",
                              [](const ModelDescriptor& descriptor)
                              { return std::make_unique<qwen3_5::Model>(descriptor); });
}
}  // namespace firefly::model
