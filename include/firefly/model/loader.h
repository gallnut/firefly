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
    /** @brief Weight-selection, retention, checksum, and allocation-alignment policy. */
    struct Options
    {
        bool   validate_checksums; ///< Whether storage loaders should validate optional checksums.
        size_t alignment; ///< Byte alignment used for persistent pooled allocations.
        std::function<bool(std::string_view)> include_weight; ///< Optional predicate filtering checkpoint names.
        std::function<bool(std::string_view)> retain_weight; ///< Optional predicate selecting persistent pool storage.

        /** @brief Constructs default loading policy with checksum validation disabled and 256-byte alignment. */
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
    static Result<std::unordered_map<std::string, Tensor>> load_safetensors(
        const std::string& filepath, ModelWeightPool<Device::CUDA>& pool, const Options& options = {})
    {
        return load_shards({filepath}, pool, options);
    }

    /**
     * @brief Loads a single-file or sharded Hugging Face SafeTensors model directory.
     * @param model_directory Directory containing `model.safetensors`, one discoverable file, or an index JSON.
     * @param pool Persistent CUDA weight pool.
     * @param options Weight selection and allocation policy.
     * @return Device tensor map or a recoverable loading error.
     */
    static Result<std::unordered_map<std::string, Tensor>> load_model(
        const std::string& model_directory, ModelWeightPool<Device::CUDA>& pool, const Options& options = {})
    {
        namespace fs = std::filesystem;
        const fs::path directory(model_directory);
        const fs::path single_file = directory / "model.safetensors";
        const fs::path index_file = directory / "model.safetensors.index.json";
        std::error_code filesystem_error;
        if (fs::exists(single_file, filesystem_error))
            return load_safetensors(single_file.string(), pool, options);
        if (filesystem_error)
            return unexpected(Error{ErrorCode::Io, "failed to inspect model directory: " +
                                                       filesystem_error.message(), filesystem_error.value()});

        if (!fs::exists(index_file, filesystem_error))
        {
            fs::path discovered_file;
            int discovered_count = 0;
            fs::directory_iterator iterator(directory, filesystem_error);
            if (filesystem_error)
                return unexpected(Error{ErrorCode::Io, "failed to enumerate model directory: " +
                                                           filesystem_error.message(), filesystem_error.value()});
            for (const auto& entry : iterator)
            {
                if (entry.is_regular_file(filesystem_error) && entry.path().extension() == ".safetensors")
                {
                    discovered_file = entry.path();
                    ++discovered_count;
                }
            }
            if (discovered_count == 1) return load_safetensors(discovered_file.string(), pool, options);
            return unexpected(Error{ErrorCode::NotFound,
                                    "model.safetensors or model.safetensors.index.json not found in " +
                                        model_directory});
        }

        std::ifstream input(index_file);
        if (!input.is_open())
            return unexpected(Error{ErrorCode::Io, "failed to open SafeTensors index: " + index_file.string()});
        nlohmann::json index;
        try
        {
            index = nlohmann::json::parse(input);
        }
        catch (const nlohmann::json::exception& exception)
        {
            return unexpected(Error{ErrorCode::Parse, "invalid SafeTensors index: " +
                                                          std::string(exception.what())});
        }
        if (!index.contains("weight_map") || !index["weight_map"].is_object())
            return unexpected(Error{ErrorCode::Parse, "SafeTensors index has no object weight_map"});

        std::set<std::string> shard_names;
        try
        {
            for (const auto& [name, shard] : index.at("weight_map").items())
                if (!options.include_weight || options.include_weight(name))
                    shard_names.insert(shard.get<std::string>());
        }
        catch (const nlohmann::json::exception& exception)
        {
            return unexpected(Error{ErrorCode::Parse, "invalid SafeTensors weight_map: " +
                                                          std::string(exception.what())});
        }
        std::vector<std::string> shard_paths;
        shard_paths.reserve(shard_names.size());
        for (const auto& shard_name : shard_names) shard_paths.push_back((directory / shard_name).string());
        return load_shards(shard_paths, pool, options);
    }

private:
    /**
     * @brief Loads selected tensors from one or more checkpoint shards through one stream and pool reservation.
     * @param paths Ordered shard file paths.
     * @param pool Persistent CUDA allocation pool used for retained tensors.
     * @param options Selection, retention, checksum, and alignment policy.
     * @return Device tensor map or a structured storage, allocation, or CUDA error.
     */
    static Result<std::unordered_map<std::string, Tensor>> load_shards(
        const std::vector<std::string>& paths, ModelWeightPool<Device::CUDA>& pool, const Options& options)
    {
        if (options.alignment == 0 || (options.alignment & (options.alignment - 1)) != 0)
            return unexpected(Error{ErrorCode::InvalidArgument,
                                    "model weight alignment must be a nonzero power of two"});
        std::vector<storage::SafetensorsLoader> loaders;
        loaders.reserve(paths.size());
        size_t total_size = 0;
        for (const auto& path : paths)
        {
            loaders.emplace_back(FIREFLY_TRY_CONTEXT(storage::SafetensorsLoader::create(path),
                                                      "open model checkpoint shard"));
            for (const auto& name : loaders.back().keys())
            {
                if (options.include_weight && !options.include_weight(name)) continue;
                if (options.retain_weight && !options.retain_weight(name)) continue;
                Tensor tensor = FIREFLY_TRY(loaders.back().get_tensor(name));
                total_size += (tensor.nbytes() + options.alignment - 1) / options.alignment * options.alignment;
            }
        }
        if (total_size > 0) FIREFLY_TRY(pool.reserve(total_size));
        device::Stream stream = FIREFLY_TRY(device::Stream::create());
        std::unordered_map<std::string, Tensor> weights;
        for (const auto& loader : loaders)
        {
            for (const auto& name : loader.keys())
            {
                if (options.include_weight && !options.include_weight(name)) continue;
                Tensor cpu_tensor = FIREFLY_TRY(loader.get_tensor(name));
                if (cpu_tensor.nbytes() == 0)
                    return unexpected(Error{ErrorCode::Parse, "checkpoint tensor has empty storage: " + name});
                const bool retain = !options.retain_weight || options.retain_weight(name);
                Tensor gpu_tensor;
                void* destination = nullptr;
                if (retain)
                {
                    destination = FIREFLY_TRY_CONTEXT(pool.allocate(cpu_tensor.nbytes(), options.alignment),
                                                      "allocate persistent model tensor " + name);
                    gpu_tensor = Tensor::from_external(destination, cpu_tensor.shape(), cpu_tensor.dtype(),
                                                       Device::CUDA);
                }
                else
                {
                    gpu_tensor = FIREFLY_TRY(Tensor::create(cpu_tensor.shape(), cpu_tensor.dtype(), Device::CUDA,
                                                            stream.context()));
                    destination = gpu_tensor.data();
                }
                FIREFLY_TRY(device::check_cuda(
                    cudaMemcpyAsync(destination, cpu_tensor.data(), cpu_tensor.nbytes(), cudaMemcpyHostToDevice,
                                    stream.get()),
                    "copy model tensor to device: " + name));
                weights.emplace(name, std::move(gpu_tensor));
            }
        }
        FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(stream.get()),
                                       "synchronize model weight loading"));
        return weights;
    }
};

}  // namespace firefly::model
