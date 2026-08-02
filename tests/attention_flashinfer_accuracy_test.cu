#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "firefly/kernels/attention/attention.h"

namespace
{
using firefly::DType;
using firefly::Device;
using firefly::Tensor;

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
    std::transform(raw.begin(), raw.end(), result.begin(), [](__nv_bfloat16 value) {
        return __bfloat162float(value);
    });
    return result;
}

bool run_case(int batch_size, int context_len)
{
    constexpr int num_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int page_size = 16;
    int pages = (context_len + 1 + page_size - 1) / page_size;
    int max_blocks = pages + 2;
    int total_blocks = batch_size * max_blocks;

    Tensor q({batch_size, 1, num_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor k_cache({total_blocks, page_size, kv_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor v_cache({total_blocks, page_size, kv_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor reference({batch_size, 1, num_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor candidate({batch_size, 1, num_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor context_device({batch_size}, DType::I32, Device::CUDA);
    Tensor blocks_device({batch_size * max_blocks}, DType::I32, Device::CUDA);

    int threads = 256;
    fill_bf16<<<(q.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(q.data()), q.numel(), 0x12345678U + context_len, 0.25f);
    fill_bf16<<<(k_cache.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(k_cache.data()), k_cache.numel(), 0x87654321U + context_len, 0.25f);
    fill_bf16<<<(v_cache.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(v_cache.data()), v_cache.numel(), 0x13579bdfU + context_len, 0.5f);

    std::vector<int> context_lens(batch_size, context_len);
    std::vector<int> block_table(batch_size * max_blocks);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        for (int page = 0; page < max_blocks; ++page)
        {
            block_table[batch * max_blocks + page] = batch * max_blocks + page;
        }
    }
    cudaMemcpy(context_device.data(), context_lens.data(), context_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blocks_device.data(), block_table.data(), block_table.size() * sizeof(int), cudaMemcpyHostToDevice);

    firefly::kernels::attention_ex(q, k_cache, v_cache, reference, blocks_device.data(), kv_heads, 1, max_blocks,
                                   static_cast<const int*>(context_device.data()),
                                   firefly::kernels::AttentionBackend::Paged, true, context_len + 1, context_len);
    firefly::kernels::prepare_attention_decode(context_lens.data(), block_table.data(), batch_size, max_blocks,
                                               num_heads);
    firefly::kernels::attention_ex(q, k_cache, v_cache, candidate, blocks_device.data(), kv_heads, 1, max_blocks,
                                   static_cast<const int*>(context_device.data()),
                                   firefly::kernels::AttentionBackend::FlashInfer, true, context_len + 1, context_len);
    cudaDeviceSynchronize();

    auto expected = copy_bf16(reference);
    auto actual = copy_bf16(candidate);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double dot = 0.0;
    double squared_actual = 0.0;
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
    bool passed = max_abs <= 0.003f && relative_l2 <= 0.01 && cosine >= 0.9999;
    std::cout << "batch=" << batch_size << " context=" << context_len << " max_abs=" << max_abs
              << " relative_l2=" << relative_l2 << " cosine=" << cosine << (passed ? " PASS" : " FAIL")
              << '\n';
    return passed;
}

bool run_prefill_case(int batch_size, int context_len, int seq_len)
{
    constexpr int num_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int page_size = 16;
    int pages = (context_len + seq_len + page_size - 1) / page_size;
    int max_blocks = pages + 2;
    int total_blocks = batch_size * max_blocks;

    Tensor q({batch_size, seq_len, num_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor k_cache({total_blocks, page_size, kv_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor v_cache({total_blocks, page_size, kv_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor reference({batch_size, seq_len, num_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor candidate({batch_size, seq_len, num_heads, head_dim}, DType::BF16, Device::CUDA);
    Tensor context_device({batch_size}, DType::I32, Device::CUDA);
    Tensor blocks_device({batch_size * max_blocks}, DType::I32, Device::CUDA);

    int threads = 256;
    fill_bf16<<<(q.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(q.data()), q.numel(), 0x2468ace0U + context_len + seq_len, 0.25f);
    fill_bf16<<<(k_cache.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(k_cache.data()), k_cache.numel(), 0xfdb97531U + context_len, 0.25f);
    fill_bf16<<<(v_cache.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(v_cache.data()), v_cache.numel(), 0x10293847U + seq_len, 0.5f);

    std::vector<int> context_lens(batch_size, context_len);
    std::vector<int> block_table(batch_size * max_blocks);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        for (int page = 0; page < max_blocks; ++page)
            block_table[batch * max_blocks + page] = batch * max_blocks + page;
    }
    cudaMemcpy(context_device.data(), context_lens.data(), context_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blocks_device.data(), block_table.data(), block_table.size() * sizeof(int), cudaMemcpyHostToDevice);

    firefly::kernels::attention_ex(q, k_cache, v_cache, reference, blocks_device.data(), kv_heads, seq_len,
                                   max_blocks, static_cast<const int*>(context_device.data()),
                                   firefly::kernels::AttentionBackend::Paged, false, context_len + seq_len,
                                   context_len);
    firefly::kernels::attention_ex(q, k_cache, v_cache, candidate, blocks_device.data(), kv_heads, seq_len,
                                   max_blocks, static_cast<const int*>(context_device.data()),
                                   firefly::kernels::AttentionBackend::FlashInfer, false, context_len + seq_len,
                                   context_len);
    cudaDeviceSynchronize();

    auto expected = copy_bf16(reference);
    auto actual = copy_bf16(candidate);
    double squared_error = 0.0, squared_reference = 0.0, squared_actual = 0.0, dot = 0.0;
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
    bool passed = max_abs <= 0.004f && relative_l2 <= 0.012 && cosine >= 0.9999;
    std::cout << "prefill batch=" << batch_size << " context=" << context_len << " seq=" << seq_len
              << " max_abs=" << max_abs << " relative_l2=" << relative_l2 << " cosine=" << cosine
              << (passed ? " PASS" : " FAIL") << '\n';
    return passed;
}
}  // namespace

int main()
{
    bool passed = true;
    for (int batch_size : {1, 2})
    {
        for (int context_len : {15, 16, 31, 257, 1025})
        {
            passed = run_case(batch_size, context_len) && passed;
        }
    }
    for (int batch_size : {1, 2})
    {
        for (int context_len : {0, 31, 257})
        {
            for (int seq_len : {17, 64}) passed = run_prefill_case(batch_size, context_len, seq_len) && passed;
        }
    }
    return passed ? 0 : 1;
}
