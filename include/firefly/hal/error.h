#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <variant>

#if __has_include(<expected>) && __cplusplus > 202002L
#include <expected>
#endif

namespace firefly::hal
{

enum class ErrorCategory
{
    CUDA,
    Runtime,
    System
};

// Simple error wrapper to avoid exceptions
struct Error
{
    int                  code;      // Error code (cudaError_t or other)
    ErrorCategory        category;  // Category of the error
    std::string          message;   // Description
    std::source_location location;  // Where it happened

    // Constructor for CUDA errors
    Error(cudaError_t c, std::string_view msg = "", std::source_location loc = std::source_location::current())
        : code(static_cast<int>(c)), category(ErrorCategory::CUDA), message(msg), location(loc)
    {
    }

    // Constructor for generic runtime errors
    Error(int c, ErrorCategory cat, std::string_view msg = "",
          std::source_location loc = std::source_location::current())
        : code(c), category(cat), message(msg), location(loc)
    {
    }

    [[nodiscard]] const char* name() const
    {
        if (category == ErrorCategory::CUDA)
        {
            return cudaGetErrorName(static_cast<cudaError_t>(code));
        }
        return "RuntimeError";
    }

    [[nodiscard]] const char* description() const
    {
        if (category == ErrorCategory::CUDA)
        {
            return cudaGetErrorString(static_cast<cudaError_t>(code));
        }
        return message.c_str();
    }
};

// Minimal polyfill for std::expected if missing
#if defined(__cpp_lib_expected) || (__has_include(<expected>) && __cplusplus > 202002L)

template <typename T>
using Result = std::expected<T, Error>;

using Unexpected = std::unexpected<Error>;

#else

// Simple Unexpected wrapper
template <typename E>
struct Unexpected
{
    E error;
    explicit Unexpected(E e) : error(std::move(e)) {}
};

// Deduction guide
template <typename E>
Unexpected(E) -> Unexpected<E>;

// Minimal Result implementation using std::variant
template <typename T>
class Result
{
public:
    // Success constructor
    Result(T value) : storage_(std::move(value)) {}
    Result() = default;  // Default construct T

    // Error constructor
    Result(Unexpected<Error> u) : storage_(std::move(u.error)) {}

    [[nodiscard]] bool has_value() const { return std::holds_alternative<T>(storage_); }
    explicit           operator bool() const { return has_value(); }

    T& value()
    {
        if (auto* val = std::get_if<T>(&storage_)) return *val;
        std::abort();  // Should throw bad_expected_access but keeping simple
    }

    const T& value() const
    {
        if (auto* val = std::get_if<T>(&storage_)) return *val;
        std::abort();
    }

    const Error& error() const
    {
        if (auto* err = std::get_if<Error>(&storage_)) return *err;
        std::abort();
    }

    Error& error()
    {
        if (auto* err = std::get_if<Error>(&storage_)) return *err;
        std::abort();
    }

private:
    std::variant<T, Error> storage_;
};

// Specialization for void
template <>
class Result<void>
{
public:
    Result() = default;  // Success
    Result(Unexpected<Error> u) : error_(std::move(u.error)) {}

    [[nodiscard]] bool has_value() const { return !error_.has_value(); }
    explicit           operator bool() const { return has_value(); }

    void value() const
    {
        if (!has_value()) std::abort();
    }

    const Error& error() const
    {
        if (error_) return *error_;
        std::abort();
    }

private:
    std::optional<Error> error_;
};

#endif

// Helper alias for usage consistency
// using firefly::hal::Unexpected;

// Helper to create unexpected error easily if needed, matching std usage
// But since we have Unexpected struct, we can just use that.
// We also need to make sure std::unexpected works if using std::expected.
// So we alias std::unexpected to hal::Unexpected if using std.

// If using std::expected, we need to bring std::unexpected into scope or usage
#if defined(__cpp_lib_expected) || (__has_include(<expected>) && __cplusplus > 202002L)
// No-op, std::unexpected is used directly or via alias if we defined one,
// but wait, std::unexpected is a class template.
// We can't easily alias strict usage unless we wrap it.
// Let's just use std::unexpected when available.
#else
// When using polyfill, functions returning Result<T> can return Unexpected(Error{...})
// But std::unexpected is a specific type.
// We should bring our Unexpected into std namespace? No, that's undefined behavior.
// We should use firefly::hal::Unexpected in our code.
#endif

// To make code cleaner, let's inject `unexpected` into this namespace
// compatible with both.

#if defined(__cpp_lib_expected) || (__has_include(<expected>) && __cplusplus > 202002L)
using std::unexpected;
#else
template <typename E>
using unexpected = Unexpected<E>;
#endif

// ... macros ...

// Alias for void result
using Status = Result<void>;

// Helper macro: if expression returns error, propagate it (like Rust's ?)
#define FIREFLY_TRY(expression)                                       \
    ({                                                                \
        auto&& _result = (expression);                                \
        if (!_result.has_value()) return unexpected(_result.error()); \
        std::forward<decltype(_result)>(_result).value();             \
    })

// Helper: check raw CUDA API call. If error, return unexpected(Error(err))
inline Status check_cuda(cudaError_t err, std::source_location loc = std::source_location::current())
{
    if (err != cudaSuccess)
    {
        return unexpected(Error{err, "", loc});
    }
    return {};
}

// Fatal check: for unrecoverable errors (e.g. in destructors)
inline void check_cuda_fatal(cudaError_t err, std::source_location loc = std::source_location::current())
{
    if (__builtin_expect(err != cudaSuccess, 0))
    {
        fprintf(stderr, "Fatal CUDA Error at %s:%d: %s (%s)\n", loc.file_name(), loc.line(), cudaGetErrorString(err),
                cudaGetErrorName(err));
        std::abort();
    }
}

}  // namespace firefly::hal
