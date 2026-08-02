#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>

#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/cache/kv_cache.h"
#include "firefly/kernels/transformer/embedding.h"
#include "firefly/kernels/transformer/linear.h"
#include "firefly/kernels/transformer/residual.h"
#include "firefly/kernels/transformer/rms_norm.h"
#include "firefly/kernels/transformer/rope.h"
#include "firefly/kernels/transformer/swiglu.h"
#include "firefly/model/qwen/model.h"
#include "firefly/model/registry.h"

namespace firefly::model::qwen
{
QwenModel::QwenModel(const ModelConfig& cfg) : config(cfg) { layers.resize(config.num_hidden_layers); }

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

    std::cout << "QwenModel weights mapped successfully!" << std::endl;
}

Tensor QwenModel::forward(const ModelInput& input, const ForwardOptions& options)
{
    const Tensor& input_ids = input.input_ids;
    const Tensor& context_lens = input.context_lens;
    auto&         kv_cache = input.kv_cache;
    int*          block_table = kv_cache.block_table;
    int           max_blocks = kv_cache.max_blocks_per_sequence;
    int batch_size = input_ids.shape()[0];
    int seq_len = input_ids.shape()[1];

    Tensor hidden_states({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA,
                         options.context);
    kernels::embedding_lookup(input_ids, token_embeddings, hidden_states, options.context);

    Tensor next_input_norm;
    for (int i = 0; i < config.num_hidden_layers; ++i)
    {
        const auto& layer = layers[i];

        Tensor norm_out;
        if (i == 0)
        {
            norm_out = Tensor({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA,
                              options.context);
            kernels::rms_norm(hidden_states, layer.input_layernorm, norm_out, config.rms_norm_eps, options.context);
        }
        else
        {
            norm_out = std::move(next_input_norm);
        }

        Tensor q({(long)batch_size, seq_len, (long)config.num_attention_heads, (long)config.head_dim}, config.dtype,
                 Device::CUDA, options.context);
        Tensor k({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, config.dtype,
                 Device::CUDA, options.context);
        Tensor v({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, config.dtype,
                 Device::CUDA, options.context);
        kernels::matmul(norm_out, layer.attention.q_proj, q, options.context);
        kernels::matmul(norm_out, layer.attention.k_proj, k, options.context);
        kernels::matmul(norm_out, layer.attention.v_proj, v, options.context);

        if (layer.attention.q_norm.data() != nullptr && layer.attention.k_norm.data() != nullptr)
        {
            kernels::rms_norm(q, layer.attention.q_norm, q, config.rms_norm_eps, options.context);
            kernels::rms_norm(k, layer.attention.k_norm, k, config.rms_norm_eps, options.context);
        }

        const int* context_lens_ptr = nullptr;
        if (context_lens.data() != nullptr)
        {
            context_lens_ptr = static_cast<const int*>(context_lens.data());
        }

        kernels::apply_rope(q, k, config.head_dim, seq_len, config.rope_theta, context_lens_ptr, options.context);

        kernels::append_paged_kv(k, v, kv_cache.key_layers[i], kv_cache.value_layers[i], block_table,
                                 context_lens_ptr, max_blocks, options.context);

        Tensor attn_out({(long)batch_size, seq_len, (long)(config.num_attention_heads * config.head_dim)}, config.dtype,
                        Device::CUDA, options.context);
        if (context_lens_ptr == nullptr || block_table == nullptr)
        {
            kernels::attention(q, k, v, attn_out, nullptr, config.num_key_value_heads, seq_len, max_blocks, nullptr,
                               false, 0, options.context);
        }
        else
        {
            auto backend = kernels::get_attention_backend();
            if (backend == kernels::AttentionBackend::Contiguous && options.min_context_len == 0 && seq_len > 1)
            {
                kernels::attention_ex(q, k, v, attn_out, nullptr, config.num_key_value_heads, seq_len, max_blocks,
                                      nullptr, backend, options.prefer_split_decode,
                                      options.max_decode_context_len, 0, options.context);
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
                kernels::attention_ex(q, kv_cache.key_layers[i], kv_cache.value_layers[i], attn_out, block_table,
                                      config.num_key_value_heads,
                                      seq_len, max_blocks, context_lens_ptr, backend, options.prefer_split_decode,
                                      options.max_decode_context_len, prefill_context_len, options.context);
            }
        }

        Tensor o_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA,
                     options.context);
        kernels::matmul(attn_out, layer.attention.o_proj, o_out, options.context);
        kernels::add_rms_norm(hidden_states, o_out, layer.post_attention_layernorm, norm_out, config.rms_norm_eps,
                              options.context);

        Tensor gate({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA,
                    options.context);
        Tensor up({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA,
                  options.context);
        Tensor mlp_intermediate({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype,
                                Device::CUDA, options.context);

        kernels::matmul(norm_out, layer.mlp.gate_proj, gate, options.context);
        kernels::matmul(norm_out, layer.mlp.up_proj, up, options.context);
        kernels::swiglu(gate, up, mlp_intermediate, options.context);

        Tensor mlp_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA,
                       options.context);
        kernels::matmul(mlp_intermediate, layer.mlp.down_proj, mlp_out, options.context);
        if (i + 1 < config.num_hidden_layers)
        {
            next_input_norm = Tensor({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype,
                                     Device::CUDA, options.context);
            kernels::add_rms_norm(hidden_states, mlp_out, layers[i + 1].input_layernorm, next_input_norm,
                                  config.rms_norm_eps, options.context);
        }
        else
        {
            kernels::add_inplace(hidden_states, mlp_out, options.context);
        }
    }

    if (!options.compute_logits)
    {
        return Tensor();
    }

    Tensor final_norm_out({(long)batch_size, 1, (long)config.hidden_size}, config.dtype, Device::CUDA,
                          options.context);

    // If context_len == 0, hidden_states is [batch, seq_len, hidden_size]
    // We only need the last token to calculate the logits for the *next* token
    Tensor last_hidden;
    Tensor compact_last_hidden;
    if (seq_len == 1)
    {
        last_hidden =
            Tensor::from_external(hidden_states.data(), {batch_size, 1, (long)config.hidden_size}, config.dtype,
                                  Device::CUDA);
    }
    else
    {
        compact_last_hidden = Tensor({(long)batch_size, 1, (long)config.hidden_size}, config.dtype, Device::CUDA,
                                     options.context);
        cudaMemcpy2DAsync(compact_last_hidden.data(), config.hidden_size * dtype_size(config.dtype),
                          static_cast<std::byte*>(hidden_states.data()) +
                              (seq_len - 1) * config.hidden_size * dtype_size(config.dtype),
                          seq_len * config.hidden_size * dtype_size(config.dtype),
                          config.hidden_size * dtype_size(config.dtype), batch_size, cudaMemcpyDeviceToDevice,
                          options.context.stream());
        last_hidden =
            Tensor::from_external(compact_last_hidden.data(), {batch_size, 1, (long)config.hidden_size}, config.dtype,
                                  Device::CUDA);
    }

    kernels::rms_norm(last_hidden, norm, final_norm_out, config.rms_norm_eps, options.context);

    Tensor logits({(long)batch_size, 1, (long)config.vocab_size}, config.dtype, Device::CUDA, options.context);
    kernels::matmul(final_norm_out, lm_head, logits, options.context);

    return logits;
}

}  // namespace firefly::model::qwen

namespace firefly::model
{

void register_qwen_models(ModelRegistry& registry)
{
    registry.register_factory("Qwen2ForCausalLM", [](const ModelConfig& config)
                              { return std::make_unique<qwen::QwenModel>(config); });
    registry.register_factory("Qwen3ForCausalLM", [](const ModelConfig& config)
                              { return std::make_unique<qwen::QwenModel>(config); });
}

}  // namespace firefly::model
