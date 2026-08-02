#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "firefly/core/tensor.h"
#include "firefly/device/stream.h"
#include "firefly/kernels/sampling/argmax.h"

namespace
{
__global__ void delay_kernel(uint64_t cycles)
{
    uint64_t start = clock64();
    while (clock64() - start < cycles)
    {
    }
}

void require_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

firefly::device::Context make_context()
{
    auto stream_result = firefly::device::Stream::create();
    if (!stream_result) throw std::runtime_error(stream_result.error().description());
    firefly::device::Stream stream = std::move(stream_result.value());
    return stream.context();
}
}  // namespace

int main()
{
    try
    {
        firefly::device::Context context = make_context();
        firefly::Tensor logits({1, 4}, firefly::DType::BF16, firefly::Device::CUDA, context);
        firefly::Tensor output({1}, firefly::DType::I32, firefly::Device::CUDA, context);

        std::array<__nv_bfloat16, 4> initial = {
            __float2bfloat16(4.0f), __float2bfloat16(3.0f), __float2bfloat16(2.0f), __float2bfloat16(1.0f)};
        std::array<__nv_bfloat16, 4> updated = {
            __float2bfloat16(1.0f), __float2bfloat16(2.0f), __float2bfloat16(3.0f), __float2bfloat16(4.0f)};

        require_cuda(cudaMemcpy(logits.data(), initial.data(), logits.nbytes(), cudaMemcpyHostToDevice),
                     "initial logits copy");
        delay_kernel<<<1, 1, 0, context.stream()>>>(50'000'000);
        require_cuda(cudaMemcpyAsync(logits.data(), updated.data(), logits.nbytes(), cudaMemcpyHostToDevice,
                                     context.stream()),
                     "updated logits copy");
        firefly::kernels::argmax(logits, output, context);

        int token = -1;
        require_cuda(cudaMemcpyAsync(&token, output.data(), sizeof(token), cudaMemcpyDeviceToHost, context.stream()),
                     "output token copy");
        require_cuda(cudaStreamSynchronize(context.stream()), "context synchronization");
        if (token != 3)
        {
            std::cerr << "Expected argmax token 3, got " << token << '\n';
            return 1;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
