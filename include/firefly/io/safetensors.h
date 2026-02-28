#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "firefly/mm/memory_mapper.h"
#include "firefly/tensor.h"
#include "firefly/types.h"

namespace firefly::io
{

class SafetensorsLoader
{
public:
    explicit SafetensorsLoader(const std::string &filepath) : file_(filepath) { parse_header(); }

    Tensor get_tensor(const std::string &name) const
    {
        auto it = metadata_.find(name);
        if (it == metadata_.end())
        {
            throw std::runtime_error("Tensor not found: " + name);
        }

        const auto &meta = it->second;
        // Pinned memory pointer safe for async copy
        void *ptr = const_cast<std::byte *>(file_.data() + buffer_start_offset_ + meta.start_offset);

        return Tensor::from_external(ptr, meta.shape, meta.dtype, Device::CPU);
    }

    const std::vector<std::string> &keys() const { return keys_; }

private:
    struct TensorMeta
    {
        DType                dtype;
        std::vector<int64_t> shape;
        std::size_t          start_offset;
        std::size_t          end_offset;
    };

    void parse_header()
    {
        if (file_.size() < 8) [[unlikely]]
        {
            throw std::runtime_error("File too small to be a safetensors file.");
        }

        uint64_t header_size = 0;
        std::memcpy(&header_size, file_.data(), sizeof(uint64_t));

        if (8 + header_size > file_.size())
        {
            throw std::runtime_error("Invalid safetensors header size.");
        }

        std::string_view json_str(reinterpret_cast<const char *>(file_.data() + 8), header_size);
        auto             j = nlohmann::json::parse(json_str);
        buffer_start_offset_ = 8 + header_size;

        for (auto &[key, value] : j.items())
        {
            if (key == "__metadata__") continue;

            TensorMeta meta;
            meta.dtype = parse_dtype(value["dtype"].get<std::string>());

            for (auto dim : value["shape"])
            {
                meta.shape.push_back(dim.get<int64_t>());
            }

            meta.start_offset = value["data_offsets"][0].get<size_t>();
            meta.end_offset = value["data_offsets"][1].get<size_t>();

            metadata_[key] = meta;
            keys_.push_back(key);
        }
    }

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

    mm::PinnedMappedFile                        file_;
    std::size_t                                 buffer_start_offset_{0};
    std::unordered_map<std::string, TensorMeta> metadata_;
    std::vector<std::string>                    keys_;
};

}  // namespace firefly::io