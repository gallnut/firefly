#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/cache/kv_cache.h"
#include "test_support.h"

namespace
{
using firefly::Device;
using firefly::DType;
using firefly::Tensor;

Tensor make_tensor(std::vector<int64_t> shape, DType dtype)
{
    return firefly::test::require_tensor(Tensor::create(std::move(shape), dtype, Device::CUDA));
}

constexpr int page_size = 16;
constexpr int query_heads = 16;
constexpr int kv_heads = 8;
constexpr int head_dim = 128;

__global__ void fill_bf16(__nv_bfloat16* data, int64_t count, uint32_t seed, float scale)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = static_cast<uint32_t>(index) ^ seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    float normalized = static_cast<float>(value & 0xffff) / 32767.5f - 1.0f;
    data[index] = __float2bfloat16(normalized * scale);
}

std::vector<float> copy_bf16(const Tensor& tensor)
{
    std::vector<__nv_bfloat16> raw(tensor.numel());
    cudaMemcpy(raw.data(), tensor.data(), tensor.nbytes(), cudaMemcpyDeviceToHost);
    std::vector<float> result(raw.size());
    std::transform(raw.begin(), raw.end(), result.begin(),
                   [](__nv_bfloat16 value) { return __bfloat162float(value); });
    return result;
}

float measure_ms(const std::function<void()>& operation)
{
    constexpr int warmup_iterations = 20;
    constexpr int measured_iterations = 200;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) operation();
    cudaDeviceSynchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iteration = 0; iteration < measured_iterations; ++iteration) operation();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return elapsed_ms / measured_iterations;
}

bool run_case(int batch_size, int context_length)
{
    // Decode of the next token sees context_length + 1 visible tokens (the new
    // token's own K/V page is included), so the cache/block table must cover
    // that many tokens. The last slot is filled with zeros so both paths read
    // identical, safe cache contents.
    int token_count = context_length + 1;
    int blocks_per_sequence = (token_count + page_size - 1) / page_size;
    int total_blocks = batch_size * blocks_per_sequence;

    Tensor query = make_tensor({batch_size, 1, query_heads, head_dim}, DType::BF16);
    Tensor key = make_tensor({batch_size, token_count, kv_heads, head_dim}, DType::BF16);
    Tensor value = make_tensor({batch_size, token_count, kv_heads, head_dim}, DType::BF16);
    Tensor bf16_key_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor bf16_value_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor int8_key_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::I8);
    Tensor int8_value_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::I8);
    Tensor scale_cache = make_tensor({total_blocks, kv_heads, page_size, 2}, DType::F32);
    Tensor bf16_output = make_tensor({batch_size, 1, query_heads, head_dim}, DType::BF16);
    Tensor int8_output = make_tensor({batch_size, 1, query_heads, head_dim}, DType::BF16);
    Tensor context_lengths_device = make_tensor({batch_size}, DType::I32);
    Tensor block_table_device = make_tensor({batch_size * blocks_per_sequence}, DType::I32);

    int threads = 256;
    fill_bf16<<<(query.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(query.data()), query.numel(), 0x12345678U + context_length, 0.25f);
    fill_bf16<<<(key.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(key.data()), key.numel(), 0x87654321U + context_length, 0.25f);
    fill_bf16<<<(value.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(value.data()), value.numel(), 0x13579bdfU + context_length, 0.5f);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        int64_t last_token_offset =
            (static_cast<int64_t>(batch) * token_count + token_count - 1) * kv_heads * head_dim;
        cudaMemset(static_cast<__nv_bfloat16*>(key.data()) + last_token_offset, 0,
                   sizeof(__nv_bfloat16) * kv_heads * head_dim);
        cudaMemset(static_cast<__nv_bfloat16*>(value.data()) + last_token_offset, 0,
                   sizeof(__nv_bfloat16) * kv_heads * head_dim);
    }

    std::vector<int> context_lengths(batch_size, context_length);
    std::vector<int> block_table(batch_size * blocks_per_sequence);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        for (int block = 0; block < blocks_per_sequence; ++block)
            block_table[batch * blocks_per_sequence + block] = batch * blocks_per_sequence + block;
    }
    cudaMemcpy(context_lengths_device.data(), context_lengths.data(), context_lengths_device.nbytes(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(block_table_device.data(), block_table.data(), block_table_device.nbytes(), cudaMemcpyHostToDevice);

    firefly::kernels::append_paged_kv(
        key, value, bf16_key_cache, bf16_value_cache, static_cast<const int*>(block_table_device.data()), nullptr,
        blocks_per_sequence);
    firefly::kernels::append_paged_kv(
        key, value, int8_key_cache, int8_value_cache, static_cast<const int*>(block_table_device.data()), nullptr,
        blocks_per_sequence, &scale_cache);
    cudaDeviceSynchronize();

    firefly::kernels::AttentionOptions bf16_options{
        .backend = firefly::kernels::AttentionBackend::FlashInfer,
        .block_table = static_cast<const int*>(block_table_device.data()),
        .context_lengths = static_cast<const int*>(context_lengths_device.data()),
        .kv_head_count = kv_heads,
        .max_context_blocks = blocks_per_sequence,
        .max_decode_context_length = context_length,
        .prefer_split_decode = true,
    };
    firefly::kernels::AttentionOptions int8_options = bf16_options;
    int8_options.backend = firefly::kernels::AttentionBackend::Paged;
    int8_options.kv_scales = &scale_cache;

    firefly::kernels::prepare_attention_decode(context_lengths.data(), block_table.data(), batch_size,
                                               blocks_per_sequence, query_heads, kv_heads, head_dim);
    firefly::kernels::reserve_paged_decode_scratch(batch_size, query_heads, blocks_per_sequence, head_dim);
    firefly::kernels::attention(query, bf16_key_cache, bf16_value_cache, bf16_output, bf16_options);
    firefly::kernels::attention(query, int8_key_cache, int8_value_cache, int8_output, int8_options);
    cudaDeviceSynchronize();

    auto expected = copy_bf16(bf16_output);
    auto actual = copy_bf16(int8_output);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double squared_actual = 0.0;
    double dot = 0.0;
    float max_abs = 0.0f;
    for (size_t index = 0; index < expected.size(); ++index)
    {
        float error = actual[index] - expected[index];
        max_abs = std::max(max_abs, std::abs(error));
        squared_error += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
        squared_actual += static_cast<double>(actual[index]) * actual[index];
        dot += static_cast<double>(expected[index]) * actual[index];
    }
    double relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1e-30));
    double cosine = dot / std::sqrt(std::max(squared_reference * squared_actual, 1e-30));

    float bf16_ms = measure_ms([&]
                               { firefly::kernels::attention(query, bf16_key_cache, bf16_value_cache, bf16_output,
                                                             bf16_options); });
    float int8_ms = measure_ms([&]
                               { firefly::kernels::attention(query, int8_key_cache, int8_value_cache, int8_output,
                                                             int8_options); });

    size_t bf16_bytes = bf16_key_cache.nbytes() + bf16_value_cache.nbytes();
    size_t int8_bytes = int8_key_cache.nbytes() + int8_value_cache.nbytes() + scale_cache.nbytes();
    double memory_reduction = 1.0 - static_cast<double>(int8_bytes) / bf16_bytes;
    double speedup = bf16_ms / int8_ms;
    bool passed = max_abs <= 0.02f && relative_l2 <= 0.05 && cosine >= 0.999;

    std::cout << std::fixed << std::setprecision(6) << "batch=" << batch_size << " context=" << context_length
              << " max_abs=" << max_abs << " relative_l2=" << relative_l2 << " cosine=" << cosine
              << " bf16_ms=" << bf16_ms << " int8_ms=" << int8_ms << " speedup=" << speedup
              << " memory_reduction=" << memory_reduction * 100.0 << "%" << (passed ? " PASS" : " FAIL")
              << '\n';
    return passed;
}
}  // namespace

int main()
{
    bool passed = true;
    for (int batch_size : {1, 8})
    {
        for (int context_length : {128, 1024, 2048, 4096})
            passed = run_case(batch_size, context_length) && passed;
    }
    return passed ? 0 : 1;
}
