#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

#include <vector>

namespace firefly::kernels
{
enum class AttentionBackend
{
    Auto,
    Paged,
    Contiguous,
    FlashInfer
};

struct QuantizedQuery
{
    const Tensor* values = nullptr;
    const Tensor* scales = nullptr;

    [[nodiscard]] bool valid() const { return values != nullptr && scales != nullptr; }
};

struct AttentionOptions
{
    AttentionBackend backend = AttentionBackend::Auto;
    const int*       block_table = nullptr;
    const int*       context_lengths = nullptr;
    const Tensor*    kv_scales = nullptr;
    QuantizedQuery   quantized_query;
    int              kv_head_count = 0;
    int              max_context_blocks = 0;
    int              max_decode_context_length = 0;
    int              prefill_context_length = -1;
    bool             prefer_split_decode = false;
};

void attention(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
               const device::Context& context = {});
void prepare_attention_decode(const int* context_lens, const int* block_table, int batch_size, int max_context_blocks,
                              int num_query_heads, int num_kv_heads, int head_dim,
                              const device::Context& context = {});
void prepare_attention_prefill(const int* block_table, int batch_size, int sequence_length, int context_length,
                               int max_context_blocks, int num_query_heads, int num_kv_heads, int head_dim,
                               const device::Context& context = {});
bool prepare_attention_prefill_ragged(const std::vector<int>& q_indptr, const std::vector<int>& kv_indptr,
                                      const std::vector<int>& last_page_len, int max_context_blocks,
                                      int num_query_heads, int num_kv_heads, int head_dim,
                                      const device::Context& context = {});
bool launch_attention_prefill_ragged(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                     const int* block_table, int kv_head_count, int max_context_blocks, float scale,
                                     const device::Context& context = {});
void reserve_paged_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim);
AttentionBackend get_attention_backend();
const char*      attention_backend_name(AttentionBackend backend);
}  // namespace firefly::kernels
