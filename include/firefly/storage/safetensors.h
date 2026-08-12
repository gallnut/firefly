#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>

#include "firefly/core/tensor.h"
#include "firefly/core/types.h"
#include "firefly/storage/mapped_file.h"

namespace firefly::storage
{

/**
 * @brief Zero-copy reader for tensors stored in one SafeTensors file.
 *
 * Tensor views returned by `get_tensor` borrow the underlying mapped file and become
 * invalid when the loader is moved or destroyed. Pinning is attempted to accelerate
 * asynchronous host-to-device checkpoint copies.
 */
class SafetensorsLoader
{
public:
    /**
     * @brief Maps a SafeTensors file and parses its JSON header.
     * @param filepath Path to a complete single-shard SafeTensors file.
     * @return Loader or a structured mapping, bounds, metadata, or JSON error.
     */
    [[nodiscard]] static Result<SafetensorsLoader> create(const std::string &filepath)
    {
        SafetensorsLoader loader(FIREFLY_TRY(PinnedMappedFile::create(filepath)));
        FIREFLY_TRY(loader.parse_header());
        return loader;
    }

    /**
     * @brief Returns a non-owning CPU tensor view for a named checkpoint tensor.
     * @param name Exact SafeTensors key.
     * @return Borrowed tensor view into the mapped data buffer.
     * @return Borrowed tensor view or `ErrorCode::NotFound` when the name is absent.
     */
    Result<Tensor> get_tensor(const std::string &name) const
    {
        auto it = metadata_.find(name);
        if (it == metadata_.end())
        {
            return unexpected(Error{ErrorCode::NotFound, "SafeTensors tensor not found: " + name});
        }

        const auto &meta = it->second;
        // Pinned memory pointer safe for async copy
        void *ptr = const_cast<std::byte *>(file_.data() + buffer_start_offset_ + meta.start_offset);

        return Tensor::from_external(ptr, meta.shape, meta.dtype, Device::CPU);
    }

    /**
     * @brief Returns tensor keys in header iteration order, excluding `__metadata__`.
     * @return Borrowed key vector valid until the loader is moved or destroyed.
     */
    const std::vector<std::string> &keys() const { return keys_; }

private:
    /** @brief Constructs a loader around an already mapped file before header parsing. */
    explicit SafetensorsLoader(PinnedMappedFile file) : file_(std::move(file)) {}
    /** @brief Parsed metadata needed to create a tensor view into the mapped data buffer. */
    struct TensorMeta
    {
        DType                dtype; ///< Firefly scalar type parsed from the SafeTensors dtype string.
        std::vector<int64_t> shape; ///< Logical tensor dimensions.
        std::size_t          start_offset; ///< First data byte relative to the SafeTensors buffer region.
        std::size_t          end_offset; ///< One-past-last data byte relative to the buffer region.
    };

    /**
     * @brief Validates the fixed header prefix, parses JSON metadata, and indexes tensor entries.
     * @return Success or a structured bounds, JSON, tensor shape, or dtype error.
     */
    Status parse_header()
    {
        if (file_.size() < 8) [[unlikely]]
        {
            return unexpected(Error{ErrorCode::Parse, "file is too small to be SafeTensors"});
        }

        uint64_t header_size = 0;
        std::memcpy(&header_size, file_.data(), sizeof(uint64_t));

        if (8 + header_size > file_.size())
        {
            return unexpected(Error{ErrorCode::Parse, "SafeTensors header exceeds file size"});
        }

        std::string_view json_str(reinterpret_cast<const char *>(file_.data() + 8), header_size);
        nlohmann::json j;
        try
        {
            j = nlohmann::json::parse(json_str);
        }
        catch (const nlohmann::json::exception& exception)
        {
            return unexpected(Error{ErrorCode::Parse, "invalid SafeTensors JSON header: " +
                                                          std::string(exception.what())});
        }
        buffer_start_offset_ = 8 + header_size;

        for (auto &[key, value] : j.items())
        {
            if (key == "__metadata__") continue;

            TensorMeta meta;
            meta.dtype = parse_dtype(value["dtype"].get<std::string>());
            if (meta.dtype == DType::UNKNOWN)
                return unexpected(Error{ErrorCode::Parse, "unsupported SafeTensors dtype for tensor: " + key});

            for (auto dim : value["shape"])
            {
                meta.shape.push_back(dim.get<int64_t>());
            }

            meta.start_offset = value["data_offsets"][0].get<size_t>();
            meta.end_offset = value["data_offsets"][1].get<size_t>();
            if (meta.start_offset > meta.end_offset ||
                meta.end_offset > file_.size() - buffer_start_offset_)
                return unexpected(Error{ErrorCode::Parse, "invalid SafeTensors data offsets for tensor: " + key});

            metadata_[key] = meta;
            keys_.push_back(key);
        }
        return {};
    }

    /**
     * @brief Converts a SafeTensors dtype identifier to Firefly's scalar enum.
     * @param dtype_str Canonical SafeTensors dtype name such as `F16` or `BF16`.
     * @return Matching Firefly scalar type, or `DType::UNKNOWN` for unsupported names.
     */
    DType parse_dtype(const std::string &dtype_str) const
    {
        if (dtype_str == "F32") return DType::F32;
        if (dtype_str == "F16") return DType::F16;
        if (dtype_str == "BF16") return DType::BF16;
        if (dtype_str == "I8") return DType::I8;
        if (dtype_str == "I32") return DType::I32;
        if (dtype_str == "I64") return DType::I64;
        if (dtype_str == "U8") return DType::U8;
        return DType::UNKNOWN;
    }

    PinnedMappedFile                            file_; ///< Owned mapping backing every returned tensor view.
    std::size_t buffer_start_offset_{0}; ///< Absolute byte offset at which tensor payload data begins.
    std::unordered_map<std::string, TensorMeta> metadata_; ///< Parsed tensor metadata indexed by key.
    std::vector<std::string>                    keys_; ///< Header iteration order excluding metadata entries.
};

}  // namespace firefly::storage
