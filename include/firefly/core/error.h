#pragma once

#include <expected>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace firefly
{

/** @brief Stable machine-readable classification for every recoverable Firefly failure. */
enum class ErrorCode
{
    InvalidArgument,     ///< Caller supplied an unsupported value, shape, dtype, or combination.
    InvalidState,        ///< An operation was requested before required runtime state was initialized.
    NotFound,            ///< A requested file, model, tensor, token, or registry entry does not exist.
    AlreadyExists,       ///< Creation or registration conflicts with an existing resource.
    ResourceExhausted,   ///< Host memory, device memory, cache blocks, or another finite resource is exhausted.
    Io,                  ///< File-system or operating-system I/O operation failed.
    Parse,               ///< Configuration, tokenizer, or checkpoint metadata could not be parsed.
    Cuda,                ///< CUDA Runtime API operation failed.
    Cublas,              ///< cuBLAS operation failed.
    Model,               ///< Model weights, metadata, or runtime behavior is incompatible.
    Kernel,              ///< A kernel launch, tensor contract, or backend operation failed.
    Cancelled,           ///< Work was cancelled before successful completion.
    Unavailable,         ///< A requested backend or capability is not available in this build or environment.
    Internal             ///< An invariant failed without a more specific public classification.
};

/** @brief Converts an error code to a stable lowercase diagnostic name. */
[[nodiscard]] constexpr std::string_view error_code_name(ErrorCode code) noexcept
{
    switch (code)
    {
        case ErrorCode::InvalidArgument: return "invalid_argument";
        case ErrorCode::InvalidState: return "invalid_state";
        case ErrorCode::NotFound: return "not_found";
        case ErrorCode::AlreadyExists: return "already_exists";
        case ErrorCode::ResourceExhausted: return "resource_exhausted";
        case ErrorCode::Io: return "io";
        case ErrorCode::Parse: return "parse";
        case ErrorCode::Cuda: return "cuda";
        case ErrorCode::Cublas: return "cublas";
        case ErrorCode::Model: return "model";
        case ErrorCode::Kernel: return "kernel";
        case ErrorCode::Cancelled: return "cancelled";
        case ErrorCode::Unavailable: return "unavailable";
        case ErrorCode::Internal: return "internal";
    }
    return "unknown";
}

/**
 * @brief Structured, value-semantic error propagated through `std::expected`.
 *
 * The primary message describes the lowest-level failure. Higher layers append context
 * without replacing the original classification, native subsystem code, or source
 * location, preserving a useful causal chain for logs and service responses.
 */
class Error
{
public:
    /**
     * @brief Constructs a structured failure.
     * @param code Stable Firefly classification.
     * @param message Human-readable description of the root failure.
     * @param native_code Optional CUDA, cuBLAS, errno, ICU, or other subsystem status.
     * @param location Source location at which the root failure was observed.
     */
    Error(ErrorCode code, std::string message, int native_code = 0,
          std::source_location location = std::source_location::current())
        : code_(code), native_code_(native_code), message_(std::move(message)), location_(location)
    {
    }

    /** @brief Returns the stable Firefly error classification. */
    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    /** @brief Returns the optional subsystem-native numeric status. */
    [[nodiscard]] int native_code() const noexcept { return native_code_; }
    /** @brief Returns the root-cause message without appended operation context. */
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    /** @brief Returns where the root cause was first converted to a Firefly error. */
    [[nodiscard]] const std::source_location& location() const noexcept { return location_; }
    /** @brief Returns outer-to-inner operation context accumulated during propagation. */
    [[nodiscard]] const std::vector<std::string>& context() const noexcept { return context_; }

    /**
     * @brief Appends higher-level operation context while preserving the root cause.
     * @param operation Description of the operation that could not complete.
     * @return Rvalue reference suitable for immediate propagation through `std::unexpected`.
     */
    Error&& with_context(std::string operation) &&
    {
        context_.push_back(std::move(operation));
        return std::move(*this);
    }

    /** @brief Formats the classification, operation chain, root cause, native code, and source location. */
    [[nodiscard]] std::string describe() const
    {
        std::string result = std::format("{}: ", error_code_name(code_));
        for (auto iterator = context_.rbegin(); iterator != context_.rend(); ++iterator)
            result += *iterator + ": ";
        result += message_;
        if (native_code_ != 0) result += std::format(" (native_code={})", native_code_);
        result += std::format(" [{}:{}]", location_.file_name(), location_.line());
        return result;
    }

private:
    ErrorCode                code_; ///< Stable programmatic classification.
    int                      native_code_; ///< Optional subsystem-native status code.
    std::string              message_; ///< Root-cause diagnostic.
    std::source_location     location_; ///< Root-cause source location.
    std::vector<std::string> context_; ///< Higher-level operation chain.
};

/** @brief Success value or structured Firefly error using the C++23 standard implementation. */
template <typename T>
using Result = std::expected<T, Error>;

/** @brief Result alias for operations that only report success or failure. */
using Status = Result<void>;

/** @brief Constructs a failed `Result` from a structured Firefly error. */
[[nodiscard]] inline std::unexpected<Error> unexpected(Error error)
{
    return std::unexpected<Error>(std::move(error));
}

}  // namespace firefly

/**
 * @brief Propagates a failed `Result` and adds the current operation to its error context.
 * @param expression Expression yielding `firefly::Result<T>`.
 * @param operation Human-readable operation description appended during propagation.
 */
#define FIREFLY_TRY_CONTEXT(expression, operation)                            \
    ({                                                                        \
        auto _firefly_result = (expression);                                  \
        if (!_firefly_result)                                                 \
            return ::firefly::unexpected(                                     \
                std::move(_firefly_result.error()).with_context(operation));  \
        std::move(_firefly_result).value();                                   \
    })

/**
 * @brief Propagates a failed `Result` without adding context.
 * @param expression Expression yielding `firefly::Result<T>`.
 */
#define FIREFLY_TRY(expression)                                               \
    ({                                                                        \
        auto _firefly_result = (expression);                                  \
        if (!_firefly_result)                                                 \
            return ::firefly::unexpected(std::move(_firefly_result.error())); \
        std::move(_firefly_result).value();                                   \
    })
