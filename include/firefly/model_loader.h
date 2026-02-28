#pragma once

#include <cuda_fp16.h>

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/hal/stream.h"
#include "firefly/io/safetensors.h"
#include "firefly/mm/model_weight_pool.h"
#include "firefly/tensor.h"

namespace firefly
{

/**
 * @brief High-level loader to ingest models from disk (SafeTensors) into GPU memory.
 */
class ModelLoader
{
public:
    struct Options
    {
        bool   validate_checksums;
        size_t alignment;

        Options() : validate_checksums(false), alignment(256) {}
    };

    /**
     * @brief Loads all tensors from a SafeTensors file into the provided GPU memory pool.
     *
     * @param filepath Path to the .safetensors file.
     * @param pool The GPU memory pool to allocate weights in.
     * @param options Loading options.
     * @return Result<std::unordered_map<std::string, Tensor>> Map of loaded tensors on GPU.
     */
    static hal::Result<std::unordered_map<std::string, Tensor>> load_safetensors(
        const std::string& filepath, mm::ModelWeightPool<Device::CUDA>& pool, const Options& options = {})
    {
        try
        {
            io::SafetensorsLoader loader(filepath);
            const auto&           keys = loader.keys();

            size_t total_size = 0;

            for (const auto& name : keys)
            {
                Tensor cpu_tensor = loader.get_tensor(name);
                size_t aligned_size =
                    (cpu_tensor.nbytes() + options.alignment - 1) / options.alignment * options.alignment;
                total_size += aligned_size;
            }

            pool.reserve(total_size);

            auto stream_res = hal::Stream::create();
            if (!stream_res) return hal::unexpected(stream_res.error());
            hal::Stream& stream = stream_res.value();

            std::unordered_map<std::string, Tensor> gpu_tensors;
            gpu_tensors.reserve(keys.size());

            for (const auto& name : keys)
            {
                Tensor cpu_tensor = loader.get_tensor(name);
                size_t bytes = cpu_tensor.nbytes();

                void* d_ptr = pool.allocate(bytes, options.alignment);
                if (!d_ptr)
                {
                    return hal::unexpected(
                        hal::Error{cudaErrorMemoryAllocation, "Failed to allocate memory for tensor: " + name});
                }

                if (cpu_tensor.dtype() == DType::BF16)
                {
                    // Convert BF16 to FP16
                    size_t            count = cpu_tensor.numel();
                    std::vector<half> fp16_data(count);
                    const uint16_t*   src = reinterpret_cast<const uint16_t*>(cpu_tensor.data());

                    for (size_t i = 0; i < count; ++i)
                    {
                        uint32_t temp = static_cast<uint32_t>(src[i]) << 16;
                        float    f;
                        std::memcpy(&f, &temp, sizeof(float));
                        fp16_data[i] = __float2half(f);
                    }

                    cudaError_t err = cudaMemcpy(d_ptr, fp16_data.data(), bytes, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                        return hal::unexpected(hal::Error{err, "Failed to copy tensor to device: " + name});
                }
                else
                {
                    cudaError_t err = cudaMemcpy(d_ptr, cpu_tensor.data(), bytes, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess)
                        return hal::unexpected(hal::Error{err, "Failed to copy tensor to device: " + name});
                }

                gpu_tensors.emplace(name, Tensor::from_external(d_ptr, cpu_tensor.shape(), DType::F16, Device::CUDA));
            }

            cudaError_t err = cudaStreamSynchronize(stream.get());
            if (err != cudaSuccess)
            {
                return hal::unexpected(hal::Error{err, "Failed to synchronize stream after loading"});
            }

            return gpu_tensors;
        }
        catch (const std::exception& e)
        {
            return hal::unexpected(hal::Error{-1, hal::ErrorCategory::Runtime, e.what()});
        }
    }
};

}  // namespace firefly
