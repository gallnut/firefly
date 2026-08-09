#pragma once

#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/device/stream.h"
#include "firefly/model/weight_pool.h"
#include "firefly/storage/safetensors.h"

namespace firefly::model
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
        std::function<bool(std::string_view)> include_weight;
        std::function<bool(std::string_view)> retain_weight;

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
    static device::Result<std::unordered_map<std::string, Tensor>> load_safetensors(
        const std::string& filepath, ModelWeightPool<Device::CUDA>& pool, const Options& options = {})
    {
        try
        {
            storage::SafetensorsLoader loader(filepath);
            const auto&           keys = loader.keys();

            size_t total_size = 0;

            for (const auto& name : keys)
            {
                if (options.include_weight && !options.include_weight(name)) continue;
                Tensor cpu_tensor = loader.get_tensor(name);
                if (options.retain_weight && !options.retain_weight(name)) continue;
                size_t aligned_size =
                    (cpu_tensor.nbytes() + options.alignment - 1) / options.alignment * options.alignment;
                total_size += aligned_size;
            }

            pool.reserve(total_size);

            auto stream_res = device::Stream::create();
            if (!stream_res) return device::unexpected(stream_res.error());
            device::Stream& stream = stream_res.value();

            std::unordered_map<std::string, Tensor> gpu_tensors;
            gpu_tensors.reserve(keys.size());

            for (const auto& name : keys)
            {
                if (options.include_weight && !options.include_weight(name)) continue;
                Tensor cpu_tensor = loader.get_tensor(name);
                DType  dtype = cpu_tensor.dtype();
                size_t bytes = cpu_tensor.nbytes();

                if (bytes == 0)
                {
                    return device::unexpected(device::Error{
                        -1, device::ErrorCategory::Runtime, "Unsupported tensor dtype for tensor: " + name});
                }

                const bool retain = !options.retain_weight || options.retain_weight(name);
                Tensor gpu_tensor;
                void* d_ptr = nullptr;
                if (retain)
                {
                    d_ptr = pool.allocate(bytes, options.alignment);
                    if (!d_ptr)
                    {
                        return device::unexpected(device::Error{cudaErrorMemoryAllocation,
                                                                "Failed to allocate memory for tensor: " + name});
                    }
                    gpu_tensor = Tensor::from_external(d_ptr, cpu_tensor.shape(), dtype, Device::CUDA);
                }
                else
                {
                    gpu_tensor = Tensor(cpu_tensor.shape(), dtype, Device::CUDA, stream.context());
                    d_ptr = gpu_tensor.data();
                }

                cudaError_t err =
                    cudaMemcpyAsync(d_ptr, cpu_tensor.data(), bytes, cudaMemcpyHostToDevice, stream.get());
                if (err != cudaSuccess)
                    return device::unexpected(device::Error{err, "Failed to copy tensor to device: " + name});

                gpu_tensors.emplace(name, std::move(gpu_tensor));
            }

            cudaError_t err = cudaStreamSynchronize(stream.get());
            if (err != cudaSuccess)
            {
                return device::unexpected(device::Error{err, "Failed to synchronize stream after loading"});
            }

            return gpu_tensors;
        }
        catch (const std::exception& e)
        {
            return device::unexpected(device::Error{-1, device::ErrorCategory::Runtime, e.what()});
        }
    }

    static device::Result<std::unordered_map<std::string, Tensor>> load_model(
        const std::string& model_directory, ModelWeightPool<Device::CUDA>& pool, const Options& options = {})
    {
        try
        {
            namespace fs = std::filesystem;
            fs::path directory(model_directory);
            fs::path single_file = directory / "model.safetensors";
            fs::path index_file = directory / "model.safetensors.index.json";
            if (fs::exists(single_file)) return load_safetensors(single_file.string(), pool, options);
            if (!fs::exists(index_file))
            {
                return device::unexpected(device::Error{-1, device::ErrorCategory::Runtime,
                                                        "model.safetensors or model.safetensors.index.json not found"});
            }

            std::ifstream input(index_file);
            auto          index = nlohmann::json::parse(input);
            std::set<std::string> shard_names;
            for (const auto& [name, shard] : index.at("weight_map").items())
            {
                if (!options.include_weight || options.include_weight(name)) shard_names.insert(shard.get<std::string>());
            }

            size_t total_size = 0;
            for (const auto& shard_name : shard_names)
            {
                storage::SafetensorsLoader loader((directory / shard_name).string());
                for (const auto& name : loader.keys())
                {
                    if (options.include_weight && !options.include_weight(name)) continue;
                    Tensor tensor = loader.get_tensor(name);
                    if (options.retain_weight && !options.retain_weight(name)) continue;
                    total_size += (tensor.nbytes() + options.alignment - 1) / options.alignment * options.alignment;
                }
            }
            pool.reserve(total_size);

            auto stream_result = device::Stream::create();
            if (!stream_result) return device::unexpected(stream_result.error());
            device::Stream& stream = stream_result.value();
            std::unordered_map<std::string, Tensor> weights;
            for (const auto& shard_name : shard_names)
            {
                storage::SafetensorsLoader loader((directory / shard_name).string());
                for (const auto& name : loader.keys())
                {
                    if (options.include_weight && !options.include_weight(name)) continue;
                    Tensor cpu_tensor = loader.get_tensor(name);
                    const bool retain = !options.retain_weight || options.retain_weight(name);
                    Tensor gpu_tensor;
                    void* destination = nullptr;
                    if (retain)
                    {
                        destination = pool.allocate(cpu_tensor.nbytes(), options.alignment);
                        if (!destination)
                        {
                            return device::unexpected(device::Error{cudaErrorMemoryAllocation,
                                                                    "Failed to allocate memory for tensor: " + name});
                        }
                        gpu_tensor = Tensor::from_external(destination, cpu_tensor.shape(), cpu_tensor.dtype(),
                                                           Device::CUDA);
                    }
                    else
                    {
                        gpu_tensor = Tensor(cpu_tensor.shape(), cpu_tensor.dtype(), Device::CUDA, stream.context());
                        destination = gpu_tensor.data();
                    }
                    cudaError_t error = cudaMemcpyAsync(destination, cpu_tensor.data(), cpu_tensor.nbytes(),
                                                        cudaMemcpyHostToDevice, stream.get());
                    if (error != cudaSuccess) return device::unexpected(device::Error{error, "Failed to copy: " + name});
                    weights.emplace(name, std::move(gpu_tensor));
                }
            }
            cudaError_t error = cudaStreamSynchronize(stream.get());
            if (error != cudaSuccess) return device::unexpected(device::Error{error, "Failed to synchronize weights"});
            return weights;
        }
        catch (const std::exception& error)
        {
            return device::unexpected(device::Error{-1, device::ErrorCategory::Runtime, error.what()});
        }
    }
};

}  // namespace firefly::model
