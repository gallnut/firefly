#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "firefly/kernels/linear_attention/gated_delta_net.h"

namespace
{
using firefly::Device;
using firefly::DType;
using firefly::Tensor;

float random_value(uint32_t& state, float scale, float bias = 0.0f)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return bias + (static_cast<float>(state & 0xffff) / 32767.5f - 1.0f) * scale;
}

void fill_bf16(Tensor& tensor, uint32_t seed, float scale, float bias = 0.0f)
{
    std::vector<__nv_bfloat16> values(tensor.numel());
    for (auto& value : values) value = __float2bfloat16(random_value(seed, scale, bias));
    cudaMemcpy(tensor.data(), values.data(), tensor.nbytes(), cudaMemcpyHostToDevice);
}

void fill_float(Tensor& tensor, uint32_t seed, float scale, float bias = 0.0f)
{
    std::vector<float> values(tensor.numel());
    for (auto& value : values) value = random_value(seed, scale, bias);
    cudaMemcpy(tensor.data(), values.data(), tensor.nbytes(), cudaMemcpyHostToDevice);
}

template <typename T>
std::vector<T> copy_to_host(const Tensor& tensor)
{
    std::vector<T> values(tensor.numel());
    cudaMemcpy(values.data(), tensor.data(), tensor.nbytes(), cudaMemcpyDeviceToHost);
    return values;
}

struct ErrorMetrics
{
    float max_abs = 0.0f;
    double relative_l2 = 0.0;
};

template <typename Expected, typename Actual, typename ConvertExpected, typename ConvertActual>
ErrorMetrics compare(const std::vector<Expected>& expected, const std::vector<Actual>& actual,
                     ConvertExpected convert_expected, ConvertActual convert_actual)
{
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float max_abs = 0.0f;
    for (size_t index = 0; index < expected.size(); ++index)
    {
        const float reference = convert_expected(expected[index]);
        const float candidate = convert_actual(actual[index]);
        const float error = candidate - reference;
        max_abs = std::max(max_abs, std::abs(error));
        squared_error += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(reference) * reference;
    }
    return {max_abs, std::sqrt(squared_error / std::max(squared_reference, 1.0e-30))};
}

bool run_case(int sequence_length, int initial_context, int head_count = 2)
{
    constexpr int batch = 1;
    constexpr int head_dimension = 128;
    const int projected_width = head_count * head_dimension * 3;
    const int output_width = head_count * head_dimension;

    Tensor mixed_qkv({batch, sequence_length, projected_width}, DType::BF16, Device::CUDA);
    Tensor gate({batch, sequence_length, output_width}, DType::BF16, Device::CUDA);
    Tensor decay({batch, sequence_length, head_count}, DType::BF16, Device::CUDA);
    Tensor beta({batch, sequence_length, head_count}, DType::BF16, Device::CUDA);
    Tensor decay_log({head_count}, DType::F32, Device::CUDA);
    Tensor decay_bias({head_count}, DType::BF16, Device::CUDA);
    Tensor norm_weight({head_dimension}, DType::F32, Device::CUDA);
    Tensor initial_state({batch, head_count, head_dimension, head_dimension}, DType::BF16, Device::CUDA);
    Tensor prefill_state({batch, head_count, head_dimension, head_dimension}, DType::BF16, Device::CUDA);
    Tensor reference_state({batch, head_count, head_dimension, head_dimension}, DType::BF16, Device::CUDA);
    Tensor prefill_output({batch, sequence_length, output_width}, DType::BF16, Device::CUDA);
    Tensor reference_output({batch, sequence_length, output_width}, DType::BF16, Device::CUDA);
    firefly::kernels::linear_attention::GatedDeltaNetWorkspace workspace;
    Tensor state_slots({batch}, DType::I32, Device::CUDA);
    Tensor context_lengths({batch}, DType::I32, Device::CUDA);

    fill_bf16(mixed_qkv, 0x12345678U + sequence_length, 0.35f);
    fill_bf16(gate, 0x23456789U + sequence_length, 0.5f);
    fill_bf16(decay, 0x3456789aU + sequence_length, 0.6f, -0.2f);
    fill_bf16(beta, 0x456789abU + sequence_length, 0.75f);
    fill_float(decay_log, 0x56789abcU, 0.2f, -1.5f);
    fill_bf16(decay_bias, 0x6789abcdU, 0.25f, 0.4f);
    fill_float(norm_weight, 0x789abcdeU, 0.1f, 1.0f);
    fill_bf16(initial_state, 0x89abcdefU, 0.02f);
    cudaMemcpy(prefill_state.data(), initial_state.data(), initial_state.nbytes(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(reference_state.data(), initial_state.data(), initial_state.nbytes(), cudaMemcpyDeviceToDevice);
    const int slot = 0;
    cudaMemcpy(state_slots.data(), &slot, sizeof(slot), cudaMemcpyHostToDevice);
    cudaMemcpy(context_lengths.data(), &initial_context, sizeof(initial_context), cudaMemcpyHostToDevice);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    firefly::kernels::linear_attention::gated_delta_net(
        mixed_qkv, gate, decay, beta, decay_log, decay_bias, norm_weight, prefill_state,
        static_cast<const int*>(state_slots.data()), static_cast<const int*>(context_lengths.data()), prefill_output,
        &workspace, 1.0e-6, {});
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    for (int token = 0; token < sequence_length; ++token)
    {
        const int context = initial_context == 0 && token == 0 ? 0 : initial_context + token;
        cudaMemcpy(context_lengths.data(), &context, sizeof(context), cudaMemcpyHostToDevice);
        Tensor token_qkv = Tensor::from_external(
            static_cast<__nv_bfloat16*>(mixed_qkv.data()) + static_cast<int64_t>(token) * projected_width,
            {batch, 1, projected_width}, DType::BF16, Device::CUDA);
        Tensor token_gate = Tensor::from_external(
            static_cast<__nv_bfloat16*>(gate.data()) + static_cast<int64_t>(token) * output_width,
            {batch, 1, output_width}, DType::BF16, Device::CUDA);
        Tensor token_decay = Tensor::from_external(
            static_cast<__nv_bfloat16*>(decay.data()) + static_cast<int64_t>(token) * head_count,
            {batch, 1, head_count}, DType::BF16, Device::CUDA);
        Tensor token_beta = Tensor::from_external(
            static_cast<__nv_bfloat16*>(beta.data()) + static_cast<int64_t>(token) * head_count,
            {batch, 1, head_count}, DType::BF16, Device::CUDA);
        Tensor token_output = Tensor::from_external(
            static_cast<__nv_bfloat16*>(reference_output.data()) + static_cast<int64_t>(token) * output_width,
            {batch, 1, output_width}, DType::BF16, Device::CUDA);
        firefly::kernels::linear_attention::gated_delta_net(
            token_qkv, token_gate, token_decay, token_beta, decay_log, decay_bias, norm_weight, reference_state,
            static_cast<const int*>(state_slots.data()), static_cast<const int*>(context_lengths.data()), token_output,
            nullptr, 1.0e-6, {});
    }
    cudaDeviceSynchronize();

    const auto expected_output = copy_to_host<__nv_bfloat16>(reference_output);
    const auto actual_output = copy_to_host<__nv_bfloat16>(prefill_output);
    const auto output_error = compare(expected_output, actual_output,
                                      [](__nv_bfloat16 value) { return __bfloat162float(value); },
                                      [](__nv_bfloat16 value) { return __bfloat162float(value); });
    const auto expected_state = copy_to_host<float>(reference_state);
    const auto actual_state = copy_to_host<float>(prefill_state);
    const auto state_error = compare(expected_state, actual_state, [](float value) { return value; },
                                     [](float value) { return value; });
    const bool passed = output_error.max_abs <= 0.008f && output_error.relative_l2 <= 0.012 &&
                        state_error.max_abs <= 0.01f && state_error.relative_l2 <= 0.012;
    std::cout << "sequence=" << sequence_length << " heads=" << head_count
              << " initial_context=" << initial_context << " kernel_ms=" << elapsed_ms
              << " output_max_abs=" << output_error.max_abs << " output_relative_l2=" << output_error.relative_l2
              << " state_max_abs=" << state_error.max_abs << " state_relative_l2=" << state_error.relative_l2
              << (passed ? " PASS" : " FAIL") << '\n';
    return passed;
}

bool run_convolution_case(int sequence_length, int initial_context)
{
    constexpr int batch = 1;
    constexpr int channels = 96;
    constexpr int kernel_size = 4;
    Tensor input({batch, sequence_length, channels}, DType::BF16, Device::CUDA);
    Tensor weight({channels, kernel_size}, DType::BF16, Device::CUDA);
    Tensor initial_state({batch, channels, kernel_size}, DType::BF16, Device::CUDA);
    Tensor prefill_state({batch, channels, kernel_size}, DType::BF16, Device::CUDA);
    Tensor reference_state({batch, channels, kernel_size}, DType::BF16, Device::CUDA);
    Tensor prefill_output({batch, sequence_length, channels}, DType::BF16, Device::CUDA);
    Tensor reference_output({batch, sequence_length, channels}, DType::BF16, Device::CUDA);
    Tensor state_slots({batch}, DType::I32, Device::CUDA);
    Tensor context_lengths({batch}, DType::I32, Device::CUDA);
    fill_bf16(input, 0x10203040U + sequence_length, 0.4f);
    fill_bf16(weight, 0x20304050U, 0.3f);
    fill_bf16(initial_state, 0x30405060U, 0.2f);
    cudaMemcpy(prefill_state.data(), initial_state.data(), initial_state.nbytes(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(reference_state.data(), initial_state.data(), initial_state.nbytes(), cudaMemcpyDeviceToDevice);
    const int slot = 0;
    cudaMemcpy(state_slots.data(), &slot, sizeof(slot), cudaMemcpyHostToDevice);
    cudaMemcpy(context_lengths.data(), &initial_context, sizeof(initial_context), cudaMemcpyHostToDevice);
    firefly::kernels::linear_attention::causal_convolution(
        input, weight, prefill_state, static_cast<const int*>(state_slots.data()),
        static_cast<const int*>(context_lengths.data()), prefill_output, {});
    for (int token = 0; token < sequence_length; ++token)
    {
        const int context = initial_context == 0 && token == 0 ? 0 : initial_context + token;
        cudaMemcpy(context_lengths.data(), &context, sizeof(context), cudaMemcpyHostToDevice);
        Tensor token_input = Tensor::from_external(
            static_cast<__nv_bfloat16*>(input.data()) + static_cast<int64_t>(token) * channels,
            {batch, 1, channels}, DType::BF16, Device::CUDA);
        Tensor token_output = Tensor::from_external(
            static_cast<__nv_bfloat16*>(reference_output.data()) + static_cast<int64_t>(token) * channels,
            {batch, 1, channels}, DType::BF16, Device::CUDA);
        firefly::kernels::linear_attention::causal_convolution(
            token_input, weight, reference_state, static_cast<const int*>(state_slots.data()),
            static_cast<const int*>(context_lengths.data()), token_output, {});
    }
    cudaDeviceSynchronize();
    const auto expected_output = copy_to_host<__nv_bfloat16>(reference_output);
    const auto actual_output = copy_to_host<__nv_bfloat16>(prefill_output);
    const auto output_error = compare(expected_output, actual_output,
                                      [](__nv_bfloat16 value) { return __bfloat162float(value); },
                                      [](__nv_bfloat16 value) { return __bfloat162float(value); });
    const auto expected_state = copy_to_host<__nv_bfloat16>(reference_state);
    const auto actual_state = copy_to_host<__nv_bfloat16>(prefill_state);
    const auto state_error = compare(expected_state, actual_state,
                                     [](__nv_bfloat16 value) { return __bfloat162float(value); },
                                     [](__nv_bfloat16 value) { return __bfloat162float(value); });
    const bool passed = output_error.max_abs == 0.0f && state_error.max_abs == 0.0f;
    std::cout << "convolution_sequence=" << sequence_length << " initial_context=" << initial_context
              << " output_max_abs=" << output_error.max_abs << " state_max_abs=" << state_error.max_abs
              << (passed ? " PASS" : " FAIL") << '\n';
    return passed;
}

}  // namespace

int main()
{
    bool passed = run_convolution_case(17, 0) && run_convolution_case(3, 128);
    passed = run_case(17, 0) && passed;
    passed = run_case(33, 128) && passed;
    if (std::getenv("FIREFLY_GDN_TEST_LONG")) passed = run_case(512, 128, 16) && passed;
    return passed ? 0 : 1;
}
