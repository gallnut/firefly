#include <cuda_fp16.h>

#include <iostream>

#include "firefly/kernels.h"
#include "firefly/model/qwen_model.h"
#include "firefly/model_registry.h"

#define CHECK_CUDA_LAST_ERROR(msg)                                                                             \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t _err = cudaGetLastError();                                                                 \
        if (_err != cudaSuccess)                                                                               \
        {                                                                                                      \
            std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(_err) << " (" << _err << ")\n"; \
        }                                                                                                      \
    } while (0)

namespace
{
__global__ void append_kv_cache(const half* k_src, const half* v_src, half* k_cache, half* v_cache,
                                const int* block_table, const int* context_lens, int max_blocks, int num_kv_heads,
                                int head_dim)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int d = threadIdx.x;

    if (d < head_dim)
    {
        int context_start = context_lens ? context_lens[b] : 0;
        int pos = context_start + t;
        int block_idx = pos / 16;
        int block_offset = pos % 16;

        int phys_block = block_table[b * max_blocks + block_idx];

        int64_t src_idx = ((int64_t)b * gridDim.x + t) * num_kv_heads * head_dim + h * head_dim + d;

        int64_t dst_idx = (int64_t)phys_block * 16 * num_kv_heads * head_dim +
                          (int64_t)block_offset * num_kv_heads * head_dim + (int64_t)h * head_dim + d;

        k_cache[dst_idx] = k_src[src_idx];
        v_cache[dst_idx] = v_src[src_idx];
    }
}
}  // namespace

namespace firefly::model
{

QwenModel::QwenModel(const ModelConfig& cfg) : config(cfg) { layers.resize(config.num_hidden_layers); }

void QwenModel::load_weights(std::unordered_map<std::string, Tensor>& weights)
{
    // 1. Embeddings
    if (weights.count("model.embed_tokens.weight"))
    {
        token_embeddings = std::move(weights.at("model.embed_tokens.weight"));
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

Tensor QwenModel::forward(const Tensor& input_ids, const Tensor& context_lens, std::vector<Tensor>& k_caches,
                          std::vector<Tensor>& v_caches, int* block_table, int max_blocks)
{
    int batch_size = input_ids.shape()[0];
    int seq_len = input_ids.shape()[1];

    Tensor hidden_states({(long)batch_size, seq_len, (long)config.hidden_size}, DType::F16, Device::CUDA);
    kernels::embedding_lookup(input_ids, token_embeddings, hidden_states);

    for (int i = 0; i < config.num_hidden_layers; ++i)
    {
        const auto& layer = layers[i];

        Tensor norm_out({(long)batch_size, seq_len, (long)config.hidden_size}, DType::F16, Device::CUDA);
        kernels::rms_norm(hidden_states, layer.input_layernorm, norm_out, config.rms_norm_eps);

        Tensor q({(long)batch_size, seq_len, (long)config.num_attention_heads, (long)config.head_dim}, DType::F16,
                 Device::CUDA);
        Tensor k({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, DType::F16,
                 Device::CUDA);
        Tensor v({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, DType::F16,
                 Device::CUDA);

        kernels::matmul(norm_out, layer.attention.q_proj, q);
        kernels::matmul(norm_out, layer.attention.k_proj, k);
        kernels::matmul(norm_out, layer.attention.v_proj, v);

        if (layer.attention.q_norm.data() != nullptr && layer.attention.k_norm.data() != nullptr)
        {
            kernels::rms_norm(q, layer.attention.q_norm, q, config.rms_norm_eps);
            kernels::rms_norm(k, layer.attention.k_norm, k, config.rms_norm_eps);
        }

        const int* context_lens_ptr = nullptr;
        if (context_lens.data() != nullptr)
        {
            context_lens_ptr = static_cast<const int*>(context_lens.data());
        }

        kernels::apply_rope(q, k, config.head_dim, seq_len, config.rope_theta, context_lens_ptr);

        // Append K/V to cache
        dim3 grid(seq_len, config.num_key_value_heads, batch_size);
        dim3 block(128);  // assuming head_dim <= 128
        append_kv_cache<<<grid, block, 0, firefly::kernels::get_default_stream()>>>(
            (const half*)k.data(), (const half*)v.data(), (half*)k_caches[i].data(), (half*)v_caches[i].data(),
            block_table, context_lens_ptr, max_blocks, config.num_key_value_heads, config.head_dim);

        Tensor attn_out({(long)batch_size, seq_len, (long)(config.num_attention_heads * config.head_dim)}, DType::F16,
                        Device::CUDA);
        if (context_lens_ptr == nullptr ||
            batch_size == 1 && seq_len > 1)  // Using simple heuristic for Prefill for now (1 element config)
        {
            kernels::attention(q, k, v, attn_out, nullptr, config.num_key_value_heads, seq_len, max_blocks, nullptr);
        }
        else
        {
            kernels::attention(q, k_caches[i], v_caches[i], attn_out, block_table, config.num_key_value_heads, seq_len,
                               max_blocks, context_lens_ptr);
        }

        Tensor o_out({(long)batch_size, seq_len, (long)config.hidden_size}, DType::F16, Device::CUDA);
        kernels::matmul(attn_out, layer.attention.o_proj, o_out);
        kernels::add_inplace(hidden_states, o_out);

        kernels::rms_norm(hidden_states, layer.post_attention_layernorm, norm_out, config.rms_norm_eps);

        Tensor gate({(long)batch_size, seq_len, (long)config.intermediate_size}, DType::F16, Device::CUDA);
        Tensor up({(long)batch_size, seq_len, (long)config.intermediate_size}, DType::F16, Device::CUDA);
        Tensor mlp_intermediate({(long)batch_size, seq_len, (long)config.intermediate_size}, DType::F16, Device::CUDA);

        kernels::matmul(norm_out, layer.mlp.gate_proj, gate);
        kernels::matmul(norm_out, layer.mlp.up_proj, up);
        kernels::swiglu(gate, up, mlp_intermediate);

        Tensor mlp_out({(long)batch_size, seq_len, (long)config.hidden_size}, DType::F16, Device::CUDA);
        kernels::matmul(mlp_intermediate, layer.mlp.down_proj, mlp_out);
        kernels::add_inplace(hidden_states, mlp_out);
    }

    Tensor final_norm_out({(long)batch_size, 1, (long)config.hidden_size}, DType::F16, Device::CUDA);

    // If context_len == 0, hidden_states is [batch, seq_len, hidden_size]
    // We only need the last token to calculate the logits for the *next* token
    half*  last_token_hidden = (half*)hidden_states.data() + (seq_len - 1) * config.hidden_size;
    Tensor last_hidden =
        Tensor::from_external(last_token_hidden, {batch_size, 1, (long)config.hidden_size}, DType::F16, Device::CUDA);

    kernels::rms_norm(last_hidden, norm, final_norm_out, config.rms_norm_eps);

    Tensor logits({(long)batch_size, 1, (long)config.vocab_size}, DType::F16, Device::CUDA);
    kernels::matmul(final_norm_out, lm_head, logits);

    return logits;
}

namespace
{
struct QwenModel_MultiRegistrar
{
    QwenModel_MultiRegistrar()
    {
        ::firefly::model::ModelRegistry::get().register_factory("Qwen2ForCausalLM",
                                                                [](const ::firefly::model::ModelConfig& config)
                                                                { return std::make_unique<QwenModel>(config); });
        ::firefly::model::ModelRegistry::get().register_factory("Qwen3ForCausalLM",
                                                                [](const ::firefly::model::ModelConfig& config)
                                                                { return std::make_unique<QwenModel>(config); });
    }
};
static QwenModel_MultiRegistrar _multi_registrar;
}  // namespace

}  // namespace firefly::model
