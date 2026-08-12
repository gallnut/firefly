#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "firefly/kernels/attention/attention.h"
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

__global__ void fill_int8(int8_t* data, int64_t count, uint32_t seed)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = static_cast<uint32_t>(index) ^ seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    data[index] = static_cast<int8_t>(static_cast<int>(value % 127) - 63);
}

__global__ void fill_kv_scales(float* scales, int64_t token_head_count)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= token_head_count) return;
    scales[index * 2] = 0.0025f + static_cast<float>(index % 7) * 0.0001f;
    scales[index * 2 + 1] = 0.005f + static_cast<float>(index % 11) * 0.0001f;
}

std::vector<float> copy_bf16(const Tensor& tensor)
{
    std::vector<__nv_bfloat16> raw(tensor.numel());
    cudaMemcpy(raw.data(), tensor.data(), tensor.nbytes(), cudaMemcpyDeviceToHost);
    std::vector<float> result(raw.size());
    std::transform(raw.begin(), raw.end(), result.begin(), [](__nv_bfloat16 value) { return __bfloat162float(value); });
    return result;
}

bool run_case(int batch_size, int context_len, int num_heads = 16, int kv_heads = 8, int head_dim = 128)
{
    constexpr int page_size = 16;
    int           pages = (context_len + 1 + page_size - 1) / page_size;
    int           max_blocks = pages + 2;
    int           total_blocks = batch_size * max_blocks;

    Tensor q = make_tensor({batch_size, 1, num_heads, head_dim}, DType::BF16);
    Tensor k_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor v_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor reference = make_tensor({batch_size, 1, num_heads, head_dim}, DType::BF16);
    Tensor candidate = make_tensor({batch_size, 1, num_heads, head_dim}, DType::BF16);
    Tensor context_device = make_tensor({batch_size}, DType::I32);
    Tensor blocks_device = make_tensor({batch_size * max_blocks}, DType::I32);

    int threads = 256;
    fill_bf16<<<(q.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(q.data()), q.numel(),
                                                                0x12345678U + context_len, 0.25f);
    fill_bf16<<<(k_cache.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(k_cache.data()), k_cache.numel(), 0x87654321U + context_len, 0.25f);
    fill_bf16<<<(v_cache.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(v_cache.data()),
                                                                      v_cache.numel(), 0x13579bdfU + context_len, 0.5f);

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

    firefly::kernels::AttentionOptions reference_options{
        .backend = firefly::kernels::AttentionBackend::Paged,
        .block_table = static_cast<const int*>(blocks_device.data()),
        .context_lengths = static_cast<const int*>(context_device.data()),
        .kv_head_count = kv_heads,
        .max_context_blocks = max_blocks,
        .max_decode_context_length = context_len + 1,
        .prefill_context_length = context_len,
        .prefer_split_decode = true,
    };
    firefly::kernels::attention(q, k_cache, v_cache, reference, reference_options);
    firefly::kernels::prepare_attention_decode(context_lens.data(), block_table.data(), batch_size, max_blocks,
                                               num_heads, kv_heads, head_dim);
    auto candidate_options = reference_options;
    candidate_options.backend = firefly::kernels::AttentionBackend::FlashInfer;
    firefly::kernels::attention(q, k_cache, v_cache, candidate, candidate_options);
    cudaDeviceSynchronize();

    auto   expected = copy_bf16(reference);
    auto   actual = copy_bf16(candidate);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double dot = 0.0;
    double squared_actual = 0.0;
    float  max_abs = 0.0f;
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
    bool   passed = max_abs <= 0.003f && relative_l2 <= 0.01 && cosine >= 0.9999;
    std::cout << "batch=" << batch_size << " context=" << context_len << " max_abs=" << max_abs
              << " relative_l2=" << relative_l2 << " cosine=" << cosine << (passed ? " PASS" : " FAIL") << '\n';
    return passed;
}

bool run_prefill_case(int batch_size, int context_len, int seq_len)
{
    constexpr int num_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int page_size = 16;
    int           pages = (context_len + seq_len + page_size - 1) / page_size;
    int           max_blocks = pages + 2;
    int           total_blocks = batch_size * max_blocks;

    Tensor q = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor k_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor v_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::BF16);
    Tensor reference = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor candidate = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor context_device = make_tensor({batch_size}, DType::I32);
    Tensor blocks_device = make_tensor({batch_size * max_blocks}, DType::I32);

    int threads = 256;
    fill_bf16<<<(q.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(q.data()), q.numel(),
                                                                0x2468ace0U + context_len + seq_len, 0.25f);
    fill_bf16<<<(k_cache.numel() + threads - 1) / threads, threads>>>(
        static_cast<__nv_bfloat16*>(k_cache.data()), k_cache.numel(), 0xfdb97531U + context_len, 0.25f);
    fill_bf16<<<(v_cache.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(v_cache.data()),
                                                                      v_cache.numel(), 0x10293847U + seq_len, 0.5f);

    std::vector<int> context_lens(batch_size, context_len);
    std::vector<int> block_table(batch_size * max_blocks);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        for (int page = 0; page < max_blocks; ++page)
            block_table[batch * max_blocks + page] = batch * max_blocks + page;
    }
    cudaMemcpy(context_device.data(), context_lens.data(), context_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blocks_device.data(), block_table.data(), block_table.size() * sizeof(int), cudaMemcpyHostToDevice);

    firefly::kernels::AttentionOptions reference_options{
        .backend = firefly::kernels::AttentionBackend::Paged,
        .block_table = static_cast<const int*>(blocks_device.data()),
        .context_lengths = static_cast<const int*>(context_device.data()),
        .kv_head_count = kv_heads,
        .max_context_blocks = max_blocks,
        .max_decode_context_length = context_len + seq_len,
        .prefill_context_length = context_len,
    };
    firefly::kernels::attention(q, k_cache, v_cache, reference, reference_options);
    auto candidate_options = reference_options;
    candidate_options.backend = firefly::kernels::AttentionBackend::FlashInfer;
    firefly::kernels::attention(q, k_cache, v_cache, candidate, candidate_options);
    cudaDeviceSynchronize();

    auto   expected = copy_bf16(reference);
    auto   actual = copy_bf16(candidate);
    double squared_error = 0.0, squared_reference = 0.0, squared_actual = 0.0, dot = 0.0;
    float  max_abs = 0.0f;
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
    bool   passed = max_abs <= 0.004f && relative_l2 <= 0.012 && cosine >= 0.9999;
    std::cout << "prefill batch=" << batch_size << " context=" << context_len << " seq=" << seq_len
              << " max_abs=" << max_abs << " relative_l2=" << relative_l2 << " cosine=" << cosine
              << (passed ? " PASS" : " FAIL") << '\n';
    return passed;
}

bool run_contiguous_prefill_case(int batch_size, int seq_len)
{
    constexpr int num_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    Tensor q = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor k = make_tensor({batch_size, seq_len, kv_heads, head_dim}, DType::BF16);
    Tensor v = make_tensor({batch_size, seq_len, kv_heads, head_dim}, DType::BF16);
    Tensor reference = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor candidate = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);

    int threads = 256;
    fill_bf16<<<(q.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(q.data()), q.numel(),
                                                                0x31415926U + seq_len, 0.25f);
    fill_bf16<<<(k.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(k.data()), k.numel(),
                                                                0x27182818U + seq_len, 0.25f);
    fill_bf16<<<(v.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(v.data()), v.numel(),
                                                                0x16180339U + seq_len, 0.5f);

    firefly::kernels::AttentionOptions reference_options{
        .backend = firefly::kernels::AttentionBackend::Contiguous,
        .kv_head_count = kv_heads,
    };
    firefly::kernels::attention(q, k, v, reference, reference_options);
    auto candidate_options = reference_options;
    candidate_options.backend = firefly::kernels::AttentionBackend::FlashInfer;
    firefly::kernels::attention(q, k, v, candidate, candidate_options);
    cudaDeviceSynchronize();

    auto   expected = copy_bf16(reference);
    auto   actual = copy_bf16(candidate);
    double squared_error = 0.0, squared_reference = 0.0, squared_actual = 0.0, dot = 0.0;
    float  max_abs = 0.0f;
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
    bool   passed = max_abs <= 0.004f && relative_l2 <= 0.012 && cosine >= 0.9999;
    std::cout << "contiguous prefill batch=" << batch_size << " seq=" << seq_len << " max_abs=" << max_abs
              << " relative_l2=" << relative_l2 << " cosine=" << cosine << (passed ? " PASS" : " FAIL") << '\n';
    return passed;
}

bool run_quantized_prefill_case(int batch_size, int context_len, int seq_len)
{
    constexpr int num_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int page_size = 16;
    int           pages = (context_len + seq_len + page_size - 1) / page_size;
    int           max_blocks = pages + 2;
    int           total_blocks = batch_size * max_blocks;

    Tensor q = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor k_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::I8);
    Tensor v_cache = make_tensor({total_blocks, page_size, kv_heads, head_dim}, DType::I8);
    Tensor scales = make_tensor({total_blocks, kv_heads, page_size, 2}, DType::F32);
    Tensor reference = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor candidate = make_tensor({batch_size, seq_len, num_heads, head_dim}, DType::BF16);
    Tensor context_device = make_tensor({batch_size}, DType::I32);
    Tensor blocks_device = make_tensor({batch_size * max_blocks}, DType::I32);

    int threads = 256;
    fill_bf16<<<(q.numel() + threads - 1) / threads, threads>>>(static_cast<__nv_bfloat16*>(q.data()), q.numel(),
                                                                0x42424242U + context_len + seq_len, 0.25f);
    fill_int8<<<(k_cache.numel() + threads - 1) / threads, threads>>>(static_cast<int8_t*>(k_cache.data()),
                                                                      k_cache.numel(), 0x12344321U + context_len);
    fill_int8<<<(v_cache.numel() + threads - 1) / threads, threads>>>(static_cast<int8_t*>(v_cache.data()),
                                                                      v_cache.numel(), 0x56788765U + seq_len);
    int64_t token_head_count = static_cast<int64_t>(total_blocks) * page_size * kv_heads;
    fill_kv_scales<<<(token_head_count + threads - 1) / threads, threads>>>(static_cast<float*>(scales.data()),
                                                                            token_head_count);

    std::vector<int> context_lens(batch_size, context_len);
    std::vector<int> block_table(batch_size * max_blocks);
    for (int batch = 0; batch < batch_size; ++batch)
    {
        for (int page = 0; page < max_blocks; ++page)
            block_table[batch * max_blocks + page] = batch * max_blocks + page;
    }
    cudaMemcpy(context_device.data(), context_lens.data(), context_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blocks_device.data(), block_table.data(), block_table.size() * sizeof(int), cudaMemcpyHostToDevice);

    firefly::kernels::AttentionOptions reference_options{
        .backend = firefly::kernels::AttentionBackend::Paged,
        .block_table = static_cast<const int*>(blocks_device.data()),
        .context_lengths = static_cast<const int*>(context_device.data()),
        .kv_scales = &scales,
        .kv_head_count = kv_heads,
        .max_context_blocks = max_blocks,
        .max_decode_context_length = context_len + seq_len,
        .prefill_context_length = context_len,
    };
    firefly::kernels::attention(q, k_cache, v_cache, reference, reference_options);
    auto candidate_options = reference_options;
    candidate_options.backend = firefly::kernels::AttentionBackend::FlashInfer;
    firefly::kernels::attention(q, k_cache, v_cache, candidate, candidate_options);
    cudaDeviceSynchronize();

    auto   expected = copy_bf16(reference);
    auto   actual = copy_bf16(candidate);
    double squared_error = 0.0, squared_reference = 0.0, squared_actual = 0.0, dot = 0.0;
    float  max_abs = 0.0f;
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
    bool   passed = max_abs <= 0.004f && relative_l2 <= 0.012 && cosine >= 0.9999;
    std::cout << "INT8 bridge prefill batch=" << batch_size << " context=" << context_len << " seq=" << seq_len
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
    for (int context_len : {31, 257, 1025, 8191, 11469})
        passed = run_case(1, context_len, 8, 2, 256) && passed;
    for (int batch_size : {1, 2})
    {
        for (int context_len : {0, 31, 257})
        {
            for (int seq_len : {17, 64}) passed = run_prefill_case(batch_size, context_len, seq_len) && passed;
        }
        for (int seq_len : {17, 64}) passed = run_contiguous_prefill_case(batch_size, seq_len) && passed;
        for (int context_len : {31, 257})
        {
            for (int seq_len : {17, 64})
                passed = run_quantized_prefill_case(batch_size, context_len, seq_len) && passed;
        }
    }
    return passed ? 0 : 1;
}
