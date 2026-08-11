#include "firefly/kernels/attention/attention.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "firefly/core/logging.h"
#include "firefly/kernels/attention/detail/flashinfer.h"
#include "firefly/kernels/attention/detail/launch.h"

namespace firefly::kernels
{
namespace
{
attention_detail::DecodeConfig decode_config()
{
    static const attention_detail::DecodeConfig config = []
    {
        attention_detail::DecodeConfig result;
        if (const char* backend = std::getenv("FIREFLY_DECODE_BACKEND"))
        {
            result.force_single = std::string(backend) == "single";
            result.force_split = std::string(backend) == "split";
        }

        if (const char* split_size = std::getenv("FIREFLY_DECODE_SPLIT_SIZE"))
        {
            if (std::string(split_size).empty()) return result;
            int value = std::atoi(split_size);
            if (value == 128 || value == 256 || value == 512)
            {
                result.split_size = value;
            }
            else
            {
                FIREFLY_LOG_WARN("attention", "unsupported decode split size value={} fallback=256", split_size);
            }
        }
        return result;
    }();
    return config;
}

void check_launch(const char* operation)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

void validate_common(const Tensor& query, const Tensor& output, const AttentionOptions& options)
{
    require_float16_or_bfloat16(query.dtype(), "attention");
    require_same_dtype(query.dtype(), output.dtype(), "attention");
    if (query.shape().size() != 4) throw std::runtime_error("attention expects rank-4 query tensor");
    if (query.shape()[1] <= 0) throw std::runtime_error("attention requires a non-empty sequence");
    if (options.kv_head_count <= 0 || query.shape()[2] % options.kv_head_count != 0)
    {
        throw std::runtime_error("attention requires query heads to be divisible by KV heads");
    }
    if (output.numel() != query.numel())
    {
        throw std::runtime_error("attention output element count does not match query shape");
    }
}

void validate_contiguous(const Tensor& query, const Tensor& key, const Tensor& value, const AttentionOptions& options)
{
    require_same_dtype(query.dtype(), key.dtype(), "attention");
    require_same_dtype(query.dtype(), value.dtype(), "attention");
    const auto& shape = query.shape();
    if (key.shape().size() != 4 || value.shape().size() != 4 || key.shape()[0] != shape[0] ||
        value.shape()[0] != shape[0] || key.shape()[1] != shape[1] || value.shape()[1] != shape[1] ||
        key.shape()[2] != options.kv_head_count || value.shape()[2] != options.kv_head_count ||
        key.shape()[3] != shape[3] || value.shape()[3] != shape[3])
    {
        throw std::runtime_error("contiguous attention expects KV shape [batch, sequence, KV heads, head dim]");
    }
}

void validate_paged(const Tensor& query, const Tensor& key, const Tensor& value, const AttentionOptions& options)
{
    if (options.block_table == nullptr || options.context_lengths == nullptr)
    {
        throw std::runtime_error("paged attention requires a block table and context lengths");
    }
    const int head_dim = query.shape()[3];
    if (key.shape().size() != 4 || value.shape().size() != 4 || key.shape()[1] != 16 || value.shape()[1] != 16 ||
        key.shape()[2] != options.kv_head_count || value.shape()[2] != options.kv_head_count ||
        key.shape()[3] != head_dim || value.shape()[3] != head_dim)
    {
        throw std::runtime_error("paged attention expects KV cache shape [blocks, 16, KV heads, head dim]");
    }

    if (options.kv_scales == nullptr)
    {
        require_same_dtype(query.dtype(), key.dtype(), "attention");
        require_same_dtype(query.dtype(), value.dtype(), "attention");
    }
    else if (key.dtype() != DType::I8 || value.dtype() != DType::I8 || options.kv_scales->dtype() != DType::F32)
    {
        throw std::runtime_error("quantized KV attention requires I8 caches and F32 scales");
    }
}
}  // namespace

const char* attention_backend_name(AttentionBackend backend)
{
    switch (backend)
    {
        case AttentionBackend::Auto:
            return "auto";
        case AttentionBackend::Paged:
            return "paged";
        case AttentionBackend::Contiguous:
            return "contiguous";
        case AttentionBackend::FlashInfer:
            return "flashinfer";
    }
    return "unknown";
}

AttentionBackend get_attention_backend()
{
    static const AttentionBackend backend = []
    {
        const char* environment = std::getenv("FIREFLY_ATTENTION_BACKEND");
        if (environment == nullptr || std::string(environment).empty() || std::string(environment) == "auto")
        {
#ifdef FIREFLY_USE_FLASHINFER
            return AttentionBackend::FlashInfer;
#else
            return AttentionBackend::Auto;
#endif
        }
        std::string value(environment);
        if (value == "paged") return AttentionBackend::Paged;
        if (value == "contiguous") return AttentionBackend::Contiguous;
        if (value == "flashinfer") return AttentionBackend::FlashInfer;
        FIREFLY_LOG_WARN("attention", "unknown backend value={} fallback=auto", value);
        return AttentionBackend::Auto;
    }();
    return backend;
}

void reserve_paged_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim)
{
    attention_detail::reserve_decode_scratch(batch_size, num_heads, max_context_blocks, head_dim,
                                             decode_config().split_size);
}

void prepare_attention_prefill(const int* block_table, int batch_size, int sequence_length, int context_length,
                               int max_context_blocks, int num_query_heads, int num_kv_heads, int head_dim,
                               const device::Context& context)
{
#ifdef FIREFLY_USE_FLASHINFER
    if (get_attention_backend() == AttentionBackend::FlashInfer)
    {
        attention_backend::prepare_flashinfer_prefill(block_table, batch_size, sequence_length, context_length,
                                                      max_context_blocks, num_query_heads, num_kv_heads, head_dim,
                                                      context.stream());
    }
#else
    (void)block_table;
    (void)batch_size;
    (void)sequence_length;
    (void)context_length;
    (void)max_context_blocks;
    (void)num_query_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)context;
#endif
}

bool prepare_attention_prefill_ragged(const std::vector<int>& q_indptr, const std::vector<int>& kv_indptr,
                                      const std::vector<int>& last_page_len, int max_context_blocks,
                                      int num_query_heads, int num_kv_heads, int head_dim,
                                      const device::Context& context)
{
#ifdef FIREFLY_USE_FLASHINFER
    if (get_attention_backend() == AttentionBackend::FlashInfer)
    {
        attention_backend::prepare_flashinfer_prefill_ragged(q_indptr, kv_indptr, last_page_len, max_context_blocks,
                                                             num_query_heads, num_kv_heads, head_dim,
                                                             context.stream());
        return true;
    }
#else
    (void)q_indptr;
    (void)kv_indptr;
    (void)last_page_len;
    (void)max_context_blocks;
    (void)num_query_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)context;
#endif
    return false;
}

bool launch_attention_prefill_ragged(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                     const int* block_table, int kv_head_count, int max_context_blocks, float scale,
                                     const device::Context& context)
{
#ifdef FIREFLY_USE_FLASHINFER
    if (get_attention_backend() == AttentionBackend::FlashInfer)
    {
        return attention_backend::launch_flashinfer_prefill_ragged(
            query, key_cache, value_cache, output, block_table, kv_head_count, max_context_blocks, scale, context);
    }
#else
    (void)query;
    (void)key_cache;
    (void)value_cache;
    (void)output;
    (void)block_table;
    (void)kv_head_count;
    (void)max_context_blocks;
    (void)scale;
    (void)context;
#endif
    return false;
}

void attention(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
               const device::Context& context)
{
    validate_common(query, output, options);
    AttentionBackend backend = options.backend;
    bool             requested_flashinfer = backend == AttentionBackend::FlashInfer;
    if (backend == AttentionBackend::Auto)
    {
        backend = options.block_table == nullptr ? AttentionBackend::Contiguous : AttentionBackend::Paged;
    }
    if (options.kv_scales != nullptr || (backend == AttentionBackend::Contiguous && options.block_table != nullptr))
    {
        backend = AttentionBackend::Paged;
    }

    const int   sequence_length = query.shape()[1];
    const int   head_dim = query.shape()[3];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (backend == AttentionBackend::FlashInfer)
    {
        bool launched = false;
        if (options.block_table == nullptr && sequence_length > 1)
        {
            validate_contiguous(query, key, value, options);
            launched = attention_backend::launch_flashinfer_contiguous_prefill(query, key, value, output,
                                                                               options.kv_head_count, scale, context);
        }
        else if (options.block_table != nullptr && sequence_length == 1)
        {
            validate_paged(query, key, value, options);
            launched = attention_backend::launch_flashinfer_decode(query, key, value, output, options.kv_head_count,
                                                                   head_dim, scale, context);
        }
        else if (options.block_table != nullptr && options.prefill_context_length >= 0)
        {
            validate_paged(query, key, value, options);
            launched = attention_backend::launch_flashinfer_prefill(
                query, key, value, output, options.block_table, options.kv_head_count, sequence_length,
                options.max_context_blocks, options.prefill_context_length, scale, context);
        }
        if (launched)
        {
            check_launch("FlashInfer attention launch");
            return;
        }

        static bool warned = false;
        if (!warned)
        {
            FIREFLY_LOG_WARN("attention", "FlashInfer unsupported attention shape fallback=firefly");
            warned = true;
        }
        backend = options.block_table == nullptr ? AttentionBackend::Contiguous : AttentionBackend::Paged;
    }

    if (requested_flashinfer && options.kv_scales != nullptr && sequence_length > 1 &&
        options.prefill_context_length >= 0 &&
        attention_backend::launch_flashinfer_quantized_prefill(
            query, key, value, const_cast<Tensor&>(*options.kv_scales), options.block_table, output,
            options.kv_head_count, options.max_context_blocks, options.prefill_context_length, scale, context))
    {
        check_launch("FlashInfer quantized prefill launch");
        return;
    }

    if (backend == AttentionBackend::Contiguous)
    {
        validate_contiguous(query, key, value, options);
        attention_detail::launch_contiguous(query, key, value, output, options, scale, context);
    }
    else if (backend == AttentionBackend::Paged)
    {
        validate_paged(query, key, value, options);
        if (options.kv_scales != nullptr)
        {
            attention_detail::launch_quantized_paged(query, key, value, output, options, decode_config(), scale,
                                                     context);
        }
        else
        {
            attention_detail::launch_paged(query, key, value, output, options, decode_config(), scale, context);
        }
    }
    else
    {
        throw std::runtime_error("unsupported attention backend");
    }
    check_launch("attention launch");
}
}  // namespace firefly::kernels
