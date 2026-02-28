#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace firefly
{

enum class Device : uint8_t
{
    CPU,
    CUDA
};

enum class DType : uint8_t
{
    F32,
    F16,
    BF16,
    I8,
    I32,
    I64,
    U8,
    UNKNOWN
};

[[nodiscard]]
constexpr std::string_view dtype_to_string(DType dtype) noexcept
{
    switch (dtype)
    {
        using enum DType;
        case F32:
            return "Float32";
        case F16:
            return "Float16";
        case BF16:
            return "BFloat16";
        case I8:
            return "Int8";
        case I32:
            return "Int32";
        case I64:
            return "Int64";
        case U8:
            return "UInt8";
        default:
            return "UNKNOWN";
    }
}

[[nodiscard]]
constexpr size_t dtype_size(DType dtype)
{
    switch (dtype)
    {
        using enum DType;
        case F32:
        case I32:
            return 4;
        case F16:
        case BF16:
            return 2;
        case I64:
            return 8;
        case I8:
        case U8:
            return 1;
        default:
            return 0;  // Or throw?
    }
}

[[nodiscard]]
constexpr size_t element_size(DType dtype) noexcept
{
    switch (dtype)
    {
        case DType::F32:
            return 4;
        case DType::I32:
            return 4;
        case DType::I64:
            return 8;
        case DType::F16:
            return 2;
        case DType::BF16:
            return 2;
        case DType::I8:
            return 1;
        default:
            return 0;
    }
}

}  // namespace firefly