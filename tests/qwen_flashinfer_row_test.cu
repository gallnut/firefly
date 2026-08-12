#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
#include "firefly/kernels/attention/attention.h"
#include "test_support.h"

using namespace firefly;

namespace
{
using firefly::Device;
using firefly::DType;
using firefly::Tensor;

Tensor make_tensor(std::vector<int64_t> shape, DType dtype)
{
    return firefly::test::require_tensor(Tensor::create(std::move(shape), dtype, Device::CUDA));
}

float random_value(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (static_cast<float>(state & 0xffff) / 32767.5f - 1.0f) * 0.5f;
}

void fill_bf16(Tensor& tensor, uint32_t seed)
{
    std::vector<__nv_bfloat16> values(tensor.numel());
    for (auto& value : values) value = __float2bfloat16(random_value(seed));
    cudaMemcpy(tensor.data(), values.data(), tensor.nbytes(), cudaMemcpyHostToDevice);
}

void fill_int(Tensor& tensor, const std::vector<int>& values)
{
    cudaMemcpy(tensor.data(), values.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice);
}

bool close(float a, float b) { return std::abs(a - b) <= 1.0e-2f; }
}  // namespace

int main()
{
    constexpr int seq_len = 22;
    constexpr int query_heads = 16;
    constexpr int kv_heads = 4;
    constexpr int head_dim = 128;
    constexpr int max_blocks = 2;
    constexpr int context_len = 0;
    constexpr int page_size = 16;

    Tensor q = make_tensor({1, seq_len, query_heads, head_dim}, DType::BF16);
    Tensor k_cache = make_tensor({max_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor v_cache = make_tensor({max_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor out_row = make_tensor({1, seq_len, query_heads * head_dim}, DType::BF16);
    Tensor out_batch = make_tensor({2, seq_len, query_heads * head_dim}, DType::BF16);
    Tensor q_batch = make_tensor({2, seq_len, query_heads, head_dim}, DType::BF16);
    Tensor block_table_row = make_tensor({max_blocks}, DType::I32);
    Tensor block_table_batch = make_tensor({2 * max_blocks}, DType::I32);
    Tensor context_lens_row = make_tensor({1}, DType::I32);
    Tensor context_lens_batch = make_tensor({2}, DType::I32);

    fill_bf16(q, 0x1234U);
    fill_bf16(k_cache, 0x5678U);
    fill_bf16(v_cache, 0x9abcU);
    cudaMemcpy(q_batch.data(), q.data(), q.nbytes(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<__nv_bfloat16*>(q_batch.data()) + q.numel(), q.data(), q.nbytes(),
               cudaMemcpyDeviceToDevice);
    fill_int(block_table_row, {0, 1});
    fill_int(block_table_batch, {0, 1, 0, 1});
    fill_int(context_lens_row, {0});
    fill_int(context_lens_batch, {0, 0});

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    firefly::device::Context context;

    kernels::prepare_attention_prefill(static_cast<const int*>(block_table_row.data()), 1, seq_len, context_len,
                                       max_blocks, query_heads, kv_heads, head_dim, context);
    kernels::AttentionOptions row_options{
        .backend = kernels::AttentionBackend::FlashInfer,
        .block_table = static_cast<const int*>(block_table_row.data()),
        .context_lengths = static_cast<const int*>(context_lens_row.data()),
        .kv_head_count = kv_heads,
        .max_context_blocks = max_blocks,
        .prefill_context_length = context_len,
    };
    kernels::attention(q, k_cache, v_cache, out_row, row_options, context);

    kernels::prepare_attention_prefill(static_cast<const int*>(block_table_batch.data()), 2, seq_len, context_len,
                                       max_blocks, query_heads, kv_heads, head_dim, context);
    kernels::AttentionOptions batch_options{
        .backend = kernels::AttentionBackend::FlashInfer,
        .block_table = static_cast<const int*>(block_table_batch.data()),
        .context_lengths = static_cast<const int*>(context_lens_batch.data()),
        .kv_head_count = kv_heads,
        .max_context_blocks = max_blocks,
        .prefill_context_length = context_len,
    };
    kernels::attention(q_batch, k_cache, v_cache, out_batch, batch_options, context);

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 1;

    std::vector<__nv_bfloat16> row(out_row.numel());
    std::vector<__nv_bfloat16> batch(out_batch.numel());
    cudaMemcpy(row.data(), out_row.data(), out_row.nbytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(batch.data(), out_batch.data(), out_batch.nbytes(), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    int   bad = 0;
    for (int i = 0; i < static_cast<int>(row.size()); ++i)
    {
        const float a = __bfloat162float(row[i]);
        const float b = __bfloat162float(batch[i]);
        const float error = std::abs(a - b);
        if (error > max_error) max_error = error;
        if (!close(a, b)) ++bad;
    }
    std::printf("per-row vs batch prefill max_error=%.6f mismatches=%d\n", max_error, bad);
    return bad == 0 ? 0 : 1;
}
