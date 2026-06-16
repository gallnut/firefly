#pragma once

#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>
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
                DType  dtype = cpu_tensor.dtype();
                size_t bytes = cpu_tensor.nbytes();

                if (bytes == 0)
                {
                    return hal::unexpected(
                        hal::Error{-1, hal::ErrorCategory::Runtime, "Unsupported tensor dtype for tensor: " + name});
                }

                void* d_ptr = pool.allocate(bytes, options.alignment);
                if (!d_ptr)
                {
                    return hal::unexpected(
                        hal::Error{cudaErrorMemoryAllocation, "Failed to allocate memory for tensor: " + name});
                }

                cudaError_t err =
                    cudaMemcpyAsync(d_ptr, cpu_tensor.data(), bytes, cudaMemcpyHostToDevice, stream.get());
                if (err != cudaSuccess)
                    return hal::unexpected(hal::Error{err, "Failed to copy tensor to device: " + name});

                gpu_tensors.emplace(name, Tensor::from_external(d_ptr, cpu_tensor.shape(), dtype, Device::CUDA));
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
