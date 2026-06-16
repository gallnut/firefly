#include <cuda_fp16.h>

#include <cstddef>
#include <iostream>

#include "firefly/cuda_dtype.cuh"
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
template <typename scalar_t>
__global__ void append_kv_cache(const scalar_t* k_src, const scalar_t* v_src, scalar_t* k_cache, scalar_t* v_cache,
                                const int* block_table, const int* context_lens, int max_blocks, int num_kv_heads,
                                int head_dim)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int lane = threadIdx.x;

    for (int d = lane; d < head_dim; d += blockDim.x)
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

template <typename scalar_t>
void launch_append_kv_cache(const firefly::Tensor& k, const firefly::Tensor& v, firefly::Tensor& k_cache,
                            firefly::Tensor& v_cache, const int* block_table, const int* context_lens, int max_blocks,
                            int num_kv_heads, int head_dim, dim3 grid, dim3 block)
{
    append_kv_cache<scalar_t><<<grid, block, 0, firefly::kernels::get_default_stream()>>>(
        static_cast<const scalar_t*>(k.data()), static_cast<const scalar_t*>(v.data()),
        static_cast<scalar_t*>(k_cache.data()), static_cast<scalar_t*>(v_cache.data()), block_table, context_lens,
        max_blocks, num_kv_heads, head_dim);
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
        config.dtype = token_embeddings.dtype();
        firefly::kernels::require_float16_or_bfloat16(config.dtype, "QwenModel");
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
                          std::vector<Tensor>& v_caches, int* block_table, int max_blocks, bool compute_logits,
                          bool prefer_split_decode, int max_decode_context_len, int min_context_len)
{
    int batch_size = input_ids.shape()[0];
    int seq_len = input_ids.shape()[1];

    Tensor hidden_states({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA);
    kernels::embedding_lookup(input_ids, token_embeddings, hidden_states);

    for (int i = 0; i < config.num_hidden_layers; ++i)
    {
        const auto& layer = layers[i];

        Tensor norm_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA);
        kernels::rms_norm(hidden_states, layer.input_layernorm, norm_out, config.rms_norm_eps);

        Tensor q({(long)batch_size, seq_len, (long)config.num_attention_heads, (long)config.head_dim}, config.dtype,
                 Device::CUDA);
        Tensor k({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, config.dtype,
                 Device::CUDA);
        Tensor v({(long)batch_size, seq_len, (long)config.num_key_value_heads, (long)config.head_dim}, config.dtype,
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
        dim3 block(256);
        if (config.dtype == DType::BF16)
        {
            launch_append_kv_cache<__nv_bfloat16>(k, v, k_caches[i], v_caches[i], block_table, context_lens_ptr,
                                                  max_blocks, config.num_key_value_heads, config.head_dim, grid, block);
        }
        else
        {
            launch_append_kv_cache<half>(k, v, k_caches[i], v_caches[i], block_table, context_lens_ptr, max_blocks,
                                         config.num_key_value_heads, config.head_dim, grid, block);
        }

        Tensor attn_out({(long)batch_size, seq_len, (long)(config.num_attention_heads * config.head_dim)}, config.dtype,
                        Device::CUDA);
        if (context_lens_ptr == nullptr || block_table == nullptr)
        {
            kernels::attention(q, k, v, attn_out, nullptr, config.num_key_value_heads, seq_len, max_blocks, nullptr);
        }
        else
        {
            auto backend = kernels::get_attention_backend();
            if ((backend == kernels::AttentionBackend::Contiguous || backend == kernels::AttentionBackend::External) &&
                min_context_len == 0 && seq_len > 1)
            {
                kernels::attention_ex(q, k, v, attn_out, nullptr, config.num_key_value_heads, seq_len, max_blocks,
                                      nullptr, backend, prefer_split_decode, max_decode_context_len, 0);
            }
            else
            {
                int prefill_context_len = -1;
                if (seq_len > 1 && min_context_len >= 0)
                {
                    prefill_context_len = min_context_len;
                }
                else if (seq_len == 1 && max_decode_context_len > 0)
                {
                    prefill_context_len = max_decode_context_len;
                }
                kernels::attention_ex(q, k_caches[i], v_caches[i], attn_out, block_table, config.num_key_value_heads,
                                      seq_len, max_blocks, context_lens_ptr, backend, prefer_split_decode,
                                      max_decode_context_len, prefill_context_len);
            }
        }

        Tensor o_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA);
        kernels::matmul(attn_out, layer.attention.o_proj, o_out);
        kernels::add_inplace(hidden_states, o_out);

        kernels::rms_norm(hidden_states, layer.post_attention_layernorm, norm_out, config.rms_norm_eps);

        Tensor gate({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA);
        Tensor up({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype, Device::CUDA);
        Tensor mlp_intermediate({(long)batch_size, seq_len, (long)config.intermediate_size}, config.dtype,
                                Device::CUDA);

        kernels::matmul(norm_out, layer.mlp.gate_proj, gate);
        kernels::matmul(norm_out, layer.mlp.up_proj, up);
        kernels::swiglu(gate, up, mlp_intermediate);

        Tensor mlp_out({(long)batch_size, seq_len, (long)config.hidden_size}, config.dtype, Device::CUDA);
        kernels::matmul(mlp_intermediate, layer.mlp.down_proj, mlp_out);
        kernels::add_inplace(hidden_states, mlp_out);
    }

    if (!compute_logits)
    {
        return Tensor();
    }

    Tensor final_norm_out({(long)batch_size, 1, (long)config.hidden_size}, config.dtype, Device::CUDA);

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
        compact_last_hidden = Tensor({(long)batch_size, 1, (long)config.hidden_size}, config.dtype, Device::CUDA);
        cudaMemcpy2DAsync(compact_last_hidden.data(), config.hidden_size * dtype_size(config.dtype),
                          static_cast<std::byte*>(hidden_states.data()) +
                              (seq_len - 1) * config.hidden_size * dtype_size(config.dtype),
                          seq_len * config.hidden_size * dtype_size(config.dtype),
                          config.hidden_size * dtype_size(config.dtype), batch_size, cudaMemcpyDeviceToDevice,
                          firefly::kernels::get_default_stream());
        last_hidden =
            Tensor::from_external(compact_last_hidden.data(), {batch_size, 1, (long)config.hidden_size}, config.dtype,
                                  Device::CUDA);
    }

    kernels::rms_norm(last_hidden, norm, final_norm_out, config.rms_norm_eps);

    Tensor logits({(long)batch_size, 1, (long)config.vocab_size}, config.dtype, Device::CUDA);
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
