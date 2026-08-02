#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

namespace firefly::kernels
{
enum class AttentionBackend
{
    Auto,
    Paged,
    Contiguous,
    FlashInfer
};

void attention(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* block_table, int kv_head_num, int seq_len,
               int max_context_blocks, const int* context_lens, bool prefer_split_decode = false,
               int max_decode_context_len = 0, const device::Context& context = {});
void attention_ex(Tensor& q, Tensor& k, Tensor& v, Tensor& output, void* block_table, int kv_head_num, int seq_len,
                  int max_context_blocks, const int* context_lens, AttentionBackend backend,
                  bool prefer_split_decode = false, int max_decode_context_len = 0, int prefill_context_len = -1,
                  const device::Context& context = {});
void prepare_attention_decode(const int* context_lens, const int* block_table, int batch_size,
                              int max_context_blocks, int num_query_heads, const device::Context& context = {});
AttentionBackend get_attention_backend();
const char* attention_backend_name(AttentionBackend backend);
}
