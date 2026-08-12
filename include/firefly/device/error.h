#pragma once

#include <cuda_runtime.h>

#include "firefly/core/error.h"
#include "firefly/core/logging.h"

namespace firefly::device
{

/** @brief Device-namespace compatibility alias for the project-wide structured error. */
using ::firefly::Error;
/** @brief Device-namespace compatibility alias for the project-wide error classification. */
using ::firefly::ErrorCode;
/** @brief Device-namespace compatibility alias for the project-wide result template. */
using ::firefly::Result;
/** @brief Device-namespace compatibility alias for the project-wide status type. */
using ::firefly::Status;
/** @brief Device-namespace compatibility import for constructing failed results. */
using ::firefly::unexpected;

/**
 * @brief Builds a structured error from a CUDA Runtime API status.
 * @param error Native CUDA status.
 * @param operation Operation that produced the status.
 * @param location Source location at which the status was observed.
 * @return Resource-exhaustion or CUDA-classified error preserving the native status.
 */
[[nodiscard]] inline Error cuda_error(cudaError_t error, std::string_view operation,
                                      std::source_location location = std::source_location::current())
{
    return Error{error == cudaErrorMemoryAllocation ? ErrorCode::ResourceExhausted : ErrorCode::Cuda,
                 std::string(operation) + ": " + cudaGetErrorString(error), static_cast<int>(error), location};
}

/**
 * @brief Converts a CUDA Runtime API status into a structured Firefly status.
 * @param error CUDA status to inspect.
 * @param operation Operation associated with the API call.
 * @param location Source location associated with the API call.
 * @return Success for `cudaSuccess`; otherwise an `ErrorCode::Cuda` failure.
 */
inline Status check_cuda(cudaError_t error, std::string_view operation = "CUDA operation",
                         std::source_location location = std::source_location::current())
{
    if (error == cudaSuccess) return {};
    return unexpected(cuda_error(error, operation, location));
}

/**
 * @brief Logs and aborts after an unrecoverable CUDA failure in a no-fail boundary such as a destructor.
 * @param error CUDA status to inspect.
 * @param operation Operation associated with the API call.
 * @param location Source location associated with the API call.
 */
inline void check_cuda_fatal(cudaError_t error, std::string_view operation = "CUDA operation",
                             std::source_location location = std::source_location::current())
{
    if (__builtin_expect(error != cudaSuccess, 0))
    {
        Error failure{ErrorCode::Cuda, std::string(operation) + ": " + cudaGetErrorString(error),
                      static_cast<int>(error), location};
        log::submit_sync(log::Level::Critical, "cuda", failure.describe(), location);
        std::abort();
    }
}

}  // namespace firefly::device
