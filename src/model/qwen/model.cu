#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "firefly/core/logging.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/cache/kv_cache.h"
#include "firefly/kernels/transformer/embedding.h"
#include "firefly/kernels/transformer/linear.h"
#include "firefly/kernels/transformer/qk_rms_norm_rope.h"
#include "firefly/kernels/transformer/residual.h"
#include "firefly/kernels/transformer/rms_norm.h"
#include "firefly/kernels/transformer/rope.h"
#include "firefly/kernels/transformer/swiglu.h"
#include "firefly/model/qwen/model.h"
#include "firefly/model/qwen/ragged_forward.h"
#include "firefly/model/registry.h"

namespace firefly::model::qwen
{
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
namespace
{
class ForwardProfile
{
public:
    ForwardProfile(cudaStream_t stream, int sequence_length)
        : stream_(stream), sequence_length_(sequence_length), enabled_(profile_enabled() && sequence_length > 1)
    {
        if (enabled_) mark("setup");
    }

    ~ForwardProfile()
    {
        for (cudaEvent_t event : events_) cudaEventDestroy(event);
    }

    void mark(std::string_view label)
    {
        if (!enabled_) return;
        cudaEvent_t event = nullptr;
        cudaEventCreate(&event);
        cudaEventRecord(event, stream_);
        events_.push_back(event);
        labels_.emplace_back(label);
    }

    void print()
    {
        if (!enabled_ || events_.size() < 2) return;
        cudaEventSynchronize(events_.back());
        std::vector<float>       totals(labels_.size(), 0.0f);
        std::vector<std::string> categories;
        for (size_t index = 1; index < events_.size(); ++index)
        {
            float elapsed_ms = 0.0f;
            cudaEventElapsedTime(&elapsed_ms, events_[index - 1], events_[index]);
            auto   found = std::find(categories.begin(), categories.end(), labels_[index]);
            size_t category = found == categories.end() ? categories.size() : found - categories.begin();
            if (found == categories.end()) categories.push_back(labels_[index]);
            totals[category] += elapsed_ms;
        }
        float       total_ms = 0.0f;
        std::string message = std::format("model GPU profile sequence_length={}", sequence_length_);
        for (size_t index = 0; index < categories.size(); ++index)
        {
            total_ms += totals[index];
            message += std::format(" {}_ms={:.3f}", categories[index], totals[index]);
        }
        message += std::format(" total_ms={:.3f}", total_ms);
        FIREFLY_LOG_INFO("performance", "{}", message);
    }

private:
    static bool profile_enabled()
    {
        const char* value = std::getenv("FIREFLY_PROFILE_MODEL");
        return value != nullptr && std::string_view(value) != "0";
    }

    cudaStream_t             stream_ = nullptr;
    int                      sequence_length_ = 0;
    bool                     enabled_ = false;
    std::vector<cudaEvent_t> events_;
    std::vector<std::string> labels_;
};
}  // namespace
#endif

QwenModel::QwenModel(const ModelConfig& cfg) : config(cfg) { layers.resize(config.num_hidden_layers); }

namespace
{
Tensor forward_ragged(QwenModel& model, const ModelInput& input, const ForwardOptions& options)
{
    auto state = prepare_ragged_forward(model, input, options);
    const int batch = state.batch;
    const int total_tokens = state.total_tokens;

    Tensor hidden_states({(long)total_tokens, (long)model.config.hidden_size}, model.config.dtype, Device::CUDA,
                         options.context);
    kernels::embedding_lookup(input.input_ids, model.token_embeddings, hidden_states, options.context);
    Tensor norm_out({(long)total_tokens, (long)model.config.hidden_size}, model.config.dtype, Device::CUDA,
                    options.context);
    Tensor q({(long)total_tokens, 1, (long)model.config.num_attention_heads, (long)model.config.head_dim},
             model.config.dtype, Device::CUDA, options.context);
    Tensor k({(long)total_tokens, 1, (long)model.config.num_key_value_heads, (long)model.config.head_dim},
             model.config.dtype, Device::CUDA, options.context);
    Tensor v({(long)total_tokens, 1, (long)model.config.num_key_value_heads, (long)model.config.head_dim},
             model.config.dtype, Device::CUDA, options.context);
    Tensor attn_out({(long)total_tokens, 1, (long)(model.config.num_attention_heads * model.config.head_dim)},
                    model.config.dtype, Device::CUDA, options.context);
    Tensor o_out({(long)total_tokens, (long)model.config.hidden_size}, model.config.dtype, Device::CUDA,
                 options.context);
    Tensor gate({(long)total_tokens, (long)model.config.intermediate_size}, model.config.dtype, Device::CUDA,
                options.context);
    Tensor up({(long)total_tokens, (long)model.config.intermediate_size}, model.config.dtype, Device::CUDA,
              options.context);
    Tensor mlp_intermediate({(long)total_tokens, (long)model.config.intermediate_size}, model.config.dtype,
                            Device::CUDA, options.context);
    Tensor mlp_out({(long)total_tokens, (long)model.config.hidden_size}, model.config.dtype, Device::CUDA,
                   options.context);

    kernels::rms_norm(hidden_states, model.layers.front().input_layernorm, norm_out, model.config.rms_norm_eps,
                      options.context);
    const float scale = 1.0f / std::sqrt(static_cast<float>(model.config.head_dim));
    for (int layer_index = 0; layer_index < model.config.num_hidden_layers; ++layer_index)
    {
        const auto& layer = model.layers[layer_index];
        kernels::matmul(norm_out, layer.attention.q_proj, q, options.context);
        kernels::matmul(norm_out, layer.attention.k_proj, k, options.context);
        kernels::matmul(norm_out, layer.attention.v_proj, v, options.context);
        if (layer.attention.q_norm.data() != nullptr && layer.attention.k_norm.data() != nullptr)
        {
            kernels::rms_norm(q, layer.attention.q_norm, q, model.config.rms_norm_eps, options.context);
            kernels::rms_norm(k, layer.attention.k_norm, k, model.config.rms_norm_eps, options.context);
        }
        kernels::apply_rope_positions(q, k, state.positions, model.config.rope_theta, options.context);

        Tensor k_flat = Tensor::from_external(k.data(),
                                              {total_tokens, model.config.num_key_value_heads,
                                               model.config.head_dim},
                                              model.config.dtype, Device::CUDA);
        Tensor v_flat = Tensor::from_external(v.data(),
                                              {total_tokens, model.config.num_key_value_heads,
                                               model.config.head_dim},
                                              model.config.dtype, Device::CUDA);
        kernels::append_paged_kv_ragged(k_flat, v_flat, input.kv_cache.key_layers[layer_index],
                                        input.kv_cache.value_layers[layer_index], input.kv_cache.block_table,
                                        input.seq_offsets, input.seq_lengths,
                                        static_cast<const int*>(input.context_lens.data()), batch, state.max_seq_len,
                                        state.max_blocks, options.context);
        run_ragged_attention(model, input, state, layer_index, q, attn_out, scale, options);

        kernels::matmul(attn_out, layer.attention.o_proj, o_out, options.context);
        kernels::add_rms_norm(hidden_states, o_out, layer.post_attention_layernorm, norm_out,
                              model.config.rms_norm_eps, options.context);
        kernels::matmul(norm_out, layer.mlp.gate_proj, gate, options.context);
        kernels::matmul(norm_out, layer.mlp.up_proj, up, options.context);
        kernels::swiglu(gate, up, mlp_intermediate, options.context);
        kernels::matmul(mlp_intermediate, layer.mlp.down_proj, mlp_out, options.context);
        if (layer_index + 1 < model.config.num_hidden_layers)
            kernels::add_rms_norm(hidden_states, mlp_out, model.layers[layer_index + 1].input_layernorm, norm_out,
                                  model.config.rms_norm_eps, options.context);
        else
            kernels::add_inplace(hidden_states, mlp_out, options.context);
    }

    if (!options.compute_logits) return Tensor();
    return finish_ragged_forward(model, state, hidden_states, options);
}
}  // namespace

ModelRuntimeRequirements QwenModel::runtime_requirements() const
{
    return {
        .kv_cache_layer_count = config.num_hidden_layers,
        .kv_cache_head_count = config.num_key_value_heads,
        .kv_cache_head_dim = config.head_dim,
    };
}

void QwenModel::load_weights(std::unordered_map<std::string, Tensor>& weights)
{
    // 1. Embeddings
    if (weights.count("model.embed_tokens.weight"))
    {
        token_embeddings = std::move(weights.at("model.embed_tokens.weight"));
        config.dtype = token_embeddings.dtype();
        require_float16_or_bfloat16(config.dtype, "QwenModel");
    }

    // 2. Layers
    for (int i = 0; i < config.num_hidden_layers; ++i)
    {
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        // Attention
        layers[i].attention.q_proj = std::move(weights.at(prefix + "self_attn.q_proj.weight"));
        layers[i].attention.k_proj = std::move(weights.at(prefix + "self_attn.k_proj.weight"));
        layers[i].attention.v_proj = std::move(weights.at(prefix + "self_attn.v_proj.weight"));
        layers[i].attention.o_proj = std::move(weights.at(prefix + "self_attn.o_proj.weight"));

        if (weights.count(prefix + "self_attn.q_norm.weight"))
        {
            layers[i].attention.q_norm = std::move(weights.at(prefix + "self_attn.q_norm.weight"));
            layers[i].attention.k_norm = std::move(weights.at(prefix + "self_attn.k_norm.weight"));
        }

        // MLP
        layers[i].mlp.gate_proj = std::move(weights.at(prefix + "mlp.gate_proj.weight"));
        layers[i].mlp.up_proj = std::move(weights.at(prefix + "mlp.up_proj.weight"));
        layers[i].mlp.down_proj = std::move(weights.at(prefix + "mlp.down_proj.weight"));

        // Norms
        layers[i].input_layernorm = std::move(weights.at(prefix + "input_layernorm.weight"));
        layers[i].post_attention_layernorm = std::move(weights.at(prefix + "post_attention_layernorm.weight"));
    }

    // 3. Final Norm & Head
    norm = std::move(weights.at("model.norm.weight"));
    lm_head = std::move(weights.at("lm_head.weight"));

    FIREFLY_LOG_DEBUG("model", "Qwen weights mapped layers={}", config.num_hidden_layers);
}

Tensor QwenModel::forward(const ModelInput& input, const ForwardOptions& options)
{
    if (input.seq_offsets != nullptr && input.seq_lengths != nullptr)
        return forward_ragged(*this, input, options);

    const Tensor& input_ids = input.input_ids;
    const Tensor& context_lens = input.context_lens;
    auto&         kv_cache = input.kv_cache;
    int*          block_table = kv_cache.block_table;
    int           max_blocks = kv_cache.max_blocks_per_sequence;
    int           batch_size = input_ids.shape()[0];
    int           seq_len = input_ids.shape()[1];
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    ForwardProfile profile(options.context.stream(), seq_len);
#endif

    Tensor hidden_states({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA,
                         options.context);
    kernels::embedding_lookup(input_ids, token_embeddings, hidden_states, options.context);

    Tensor norm_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA, options.context);
    Tensor q({(long)batch_size, seq_len, (long)config.num_attention_heads, (long)config.head_dim}, config.dtype,
             Device::CUDA, options.context);
    Tensor k({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, config.dtype,
             Device::CUDA, options.context);
    Tensor v({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, config.dtype,
             Device::CUDA, options.context);
    Tensor attn_out({(long)batch_size, seq_len, (long)(config.num_attention_heads * config.head_dim)}, config.dtype,
                    Device::CUDA, options.context);
    Tensor o_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA, options.context);
    Tensor gate({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA,
                options.context);
    Tensor up({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA, options.context);
    Tensor mlp_intermediate({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA,
                            options.context);
    Tensor mlp_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA, options.context);

    const bool use_fused_qk = !layers.empty() && layers.front().attention.q_norm.data() != nullptr &&
                              layers.front().attention.k_norm.data() != nullptr && config.head_dim == 128 &&
                              config.num_attention_heads == 2 * config.num_key_value_heads;
    Tensor     rope_factors;
    Tensor     quantized_q;
    Tensor     quantized_q_scales;
    const bool fuse_query_quantization = use_fused_qk && kv_cache.quantized() && seq_len == 1;
    if (use_fused_qk)
    {
        rope_factors = Tensor({static_cast<int64_t>(batch_size) * seq_len, config.head_dim / 2, 2}, DType::F32,
                              Device::CUDA, options.context);
        kernels::prepare_rope_factors(rope_factors, seq_len, config.head_dim, config.rope_theta,
                                      context_lens.data() ? static_cast<const int*>(context_lens.data()) : nullptr,
                                      options.context);
    }
    if (fuse_query_quantization)
    {
        quantized_q = Tensor(q.shape(), DType::I8, Device::CUDA, options.context);
        quantized_q_scales = Tensor({static_cast<int64_t>(batch_size) * config.num_attention_heads}, DType::F32,
                                    Device::CUDA, options.context);
    }

    kernels::rms_norm(hidden_states, layers.front().input_layernorm, norm_out, config.rms_norm_eps, options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    profile.mark("setup");
#endif
    for (int i = 0; i < config.num_hidden_layers; ++i)
    {
        const auto& layer = layers[i];

        kernels::matmul(norm_out, layer.attention.q_proj, q, options.context);
        kernels::matmul(norm_out, layer.attention.k_proj, k, options.context);
        kernels::matmul(norm_out, layer.attention.v_proj, v, options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("qkv_gemm");
#endif

        if (use_fused_qk)
        {
            if (fuse_query_quantization)
            {
                kernels::qk_rms_norm_rope_quantized(q, k, layer.attention.q_norm, layer.attention.k_norm, rope_factors,
                                                    quantized_q, quantized_q_scales, config.rms_norm_eps,
                                                    options.context);
            }
            else
            {
                kernels::qk_rms_norm_rope(q, k, layer.attention.q_norm, layer.attention.k_norm, rope_factors,
                                          config.rms_norm_eps, options.context);
            }
        }
        else
        {
            if (layer.attention.q_norm.data() != nullptr && layer.attention.k_norm.data() != nullptr)
            {
                kernels::rms_norm(q, layer.attention.q_norm, q, config.rms_norm_eps, options.context);
                kernels::rms_norm(k, layer.attention.k_norm, k, config.rms_norm_eps, options.context);
            }

            kernels::apply_rope(q, k, config.head_dim, seq_len, config.rope_theta,
                                context_lens.data() ? static_cast<const int*>(context_lens.data()) : nullptr,
                                options.context);
        }

        const int* context_lens_ptr = nullptr;
        if (context_lens.data() != nullptr)
        {
            context_lens_ptr = static_cast<const int*>(context_lens.data());
        }

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("qk_norm_rope");
#endif

        kernels::append_paged_kv(k, v, kv_cache.key_layers[i], kv_cache.value_layers[i], block_table, context_lens_ptr,
                                 max_blocks, kv_cache.quantized() ? &kv_cache.scale_layers[i] : nullptr,
                                 options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("kv_append");
#endif

        if (context_lens_ptr == nullptr || block_table == nullptr)
        {
            kernels::AttentionOptions attention_options{
                .kv_head_count = config.num_key_value_heads,
                .max_context_blocks = max_blocks,
            };
            kernels::attention(q, k, v, attn_out, attention_options, options.context);
        }
        else
        {
            auto backend = kernels::get_attention_backend();
            if ((backend == kernels::AttentionBackend::Contiguous || kv_cache.quantized()) &&
                options.min_context_len == 0 && seq_len > 1)
            {
                auto                      local_prefill_backend = backend == kernels::AttentionBackend::FlashInfer
                                                                      ? kernels::AttentionBackend::FlashInfer
                                                                      : kernels::AttentionBackend::Contiguous;
                kernels::AttentionOptions attention_options{
                    .backend = local_prefill_backend,
                    .kv_head_count = config.num_key_value_heads,
                    .max_context_blocks = max_blocks,
                    .max_decode_context_length = options.max_decode_context_len,
                    .prefill_context_length = 0,
                    .prefer_split_decode = options.prefer_split_decode,
                };
                kernels::attention(q, k, v, attn_out, attention_options, options.context);
            }
            else
            {
                int prefill_context_len = -1;
                if (seq_len > 1 && options.min_context_len >= 0)
                {
                    prefill_context_len = options.min_context_len;
                }
                else if (seq_len == 1 && options.max_decode_context_len > 0)
                {
                    prefill_context_len = options.max_decode_context_len;
                }
                kernels::AttentionOptions attention_options{
                    .backend = backend,
                    .block_table = block_table,
                    .context_lengths = context_lens_ptr,
                    .kv_scales = kv_cache.quantized() ? &kv_cache.scale_layers[i] : nullptr,
                    .quantized_query = fuse_query_quantization
                                           ? kernels::QuantizedQuery{&quantized_q, &quantized_q_scales}
                                           : kernels::QuantizedQuery{},
                    .kv_head_count = config.num_key_value_heads,
                    .max_context_blocks = max_blocks,
                    .max_decode_context_length = options.max_decode_context_len,
                    .prefill_context_length = prefill_context_len,
                    .prefer_split_decode = options.prefer_split_decode,
                };
                kernels::attention(q, kv_cache.key_layers[i], kv_cache.value_layers[i], attn_out, attention_options,
                                   options.context);
            }
        }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("attention");
#endif

        kernels::matmul(attn_out, layer.attention.o_proj, o_out, options.context);
        kernels::add_rms_norm(hidden_states, o_out, layer.post_attention_layernorm, norm_out, config.rms_norm_eps,
                              options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("o_gemm_residual");
#endif

        kernels::matmul(norm_out, layer.mlp.gate_proj, gate, options.context);
        kernels::matmul(norm_out, layer.mlp.up_proj, up, options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("gate_up_gemm");
#endif
        kernels::swiglu(gate, up, mlp_intermediate, options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("swiglu");
#endif

        kernels::matmul(mlp_intermediate, layer.mlp.down_proj, mlp_out, options.context);
        if (i + 1 < config.num_hidden_layers)
        {
            kernels::add_rms_norm(hidden_states, mlp_out, layers[i + 1].input_layernorm, norm_out, config.rms_norm_eps,
                                  options.context);
        }
        else
        {
            kernels::add_inplace(hidden_states, mlp_out, options.context);
        }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.mark("down_gemm_residual");
#endif
    }

    if (!options.compute_logits)
    {
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        profile.print();
#endif
        return Tensor();
    }

    Tensor final_norm_out({(long)batch_size, 1, (long)config.hidden_size}, config.dtype, Device::CUDA, options.context);

    // If context_len == 0, hidden_states is [batch, seq_len, hidden_size]
    // We only need the last token to calculate the logits for the *next* token
    Tensor last_hidden;
    Tensor compact_last_hidden;
    if (seq_len == 1)
    {
        last_hidden = Tensor::from_external(hidden_states.data(), {batch_size, 1, (long)config.hidden_size},
                                            config.dtype, Device::CUDA);
    }
    else
    {
        compact_last_hidden =
            Tensor({(long)batch_size, 1, (long)config.hidden_size}, config.dtype, Device::CUDA, options.context);
        cudaMemcpy2DAsync(compact_last_hidden.data(), config.hidden_size * dtype_size(config.dtype),
                          static_cast<std::byte*>(hidden_states.data()) +
                              (seq_len - 1) * config.hidden_size * dtype_size(config.dtype),
                          seq_len * config.hidden_size * dtype_size(config.dtype),
                          config.hidden_size * dtype_size(config.dtype), batch_size, cudaMemcpyDeviceToDevice,
                          options.context.stream());
        last_hidden = Tensor::from_external(compact_last_hidden.data(), {batch_size, 1, (long)config.hidden_size},
                                            config.dtype, Device::CUDA);
    }

    kernels::rms_norm(last_hidden, norm, final_norm_out, config.rms_norm_eps, options.context);

    Tensor logits({(long)batch_size, 1, (long)config.vocab_size}, config.dtype, Device::CUDA, options.context);
    kernels::matmul(final_norm_out, lm_head, logits, options.context);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    profile.mark("logits");
    profile.print();
#endif

    return logits;
}

}  // namespace firefly::model::qwen

namespace firefly::model
{

void register_qwen_models(ModelRegistry& registry)
{
    registry.register_factory("Qwen2ForCausalLM", [](const ModelDescriptor& descriptor)
                              { return std::make_unique<qwen::QwenModel>(descriptor.config); });
    registry.register_factory("Qwen3ForCausalLM", [](const ModelDescriptor& descriptor)
                              { return std::make_unique<qwen::QwenModel>(descriptor.config); });
}

}  // namespace firefly::model
