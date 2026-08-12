#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "firefly/core/error.h"

namespace firefly
{

/** @brief Identifies the memory and execution domain that owns a tensor. */
enum class Device : uint8_t
{
    CPU,  ///< Host-accessible system memory.
    CUDA  ///< CUDA device memory associated with the active CUDA device.
};

/** @brief Enumerates scalar storage formats supported by Firefly tensors and kernels. */
enum class DType : uint8_t
{
    F32,      ///< IEEE-754 single-precision floating point.
    F16,      ///< IEEE-754 half-precision floating point.
    BF16,     ///< Brain floating-point with an eight-bit exponent.
    I8,       ///< Signed eight-bit integer, commonly used for quantized storage.
    I32,      ///< Signed 32-bit integer.
    I64,      ///< Signed 64-bit integer.
    U8,       ///< Unsigned eight-bit integer.
    UNKNOWN   ///< Sentinel used when no valid scalar format is known.
};

/**
 * @brief Returns the stable human-readable name of a scalar type.
 * @param dtype Scalar type to describe.
 * @return A string literal whose lifetime extends for the duration of the program.
 */
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

/**
 * @brief Returns the storage width of one value of the requested scalar type.
 * @param dtype Scalar type whose width is requested.
 * @return The number of bytes per element, or zero for `DType::UNKNOWN`.
 */
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
            return 0;
    }
}

/**
 * @brief Returns the element width used by low-level kernels.
 * @param dtype Scalar type whose width is requested.
 * @return The number of bytes per element, or zero when the type is unsupported.
 * @note This function is retained as a kernel-facing synonym of `dtype_size`.
 */
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

/**
 * @brief Validates that a kernel input uses a supported 16-bit floating-point format.
 * @param dtype Scalar type to validate.
 * @param operation Null-terminated operation name included in the diagnostic.
 * @return Success or `ErrorCode::InvalidArgument` for an unsupported scalar type.
 */
inline Status require_float16_or_bfloat16(DType dtype, const char* operation)
{
    if (dtype != DType::F16 && dtype != DType::BF16)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                std::string(operation) + " only supports F16/BF16 tensors"});
    return {};
}

/**
 * @brief Validates that two tensors use identical scalar storage formats.
 * @param first Scalar type of the first tensor.
 * @param second Scalar type of the second tensor.
 * @param operation Null-terminated operation name included in the diagnostic.
 * @return Success or `ErrorCode::InvalidArgument` when the scalar types differ.
 */
inline Status require_same_dtype(DType first, DType second, const char* operation)
{
    if (first != second)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                std::string(operation) + " requires tensors with the same dtype"});
    return {};
}

}  // namespace firefly
