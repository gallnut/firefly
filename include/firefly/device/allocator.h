#pragma once
#include <cstddef>

#include "firefly/core/error.h"
#include "firefly/core/types.h"

namespace firefly
{

/**
 * @brief Device-specific byte allocator selected at compile time.
 * @tparam D Allocation domain.
 */
template <Device D>
struct DeviceAllocator;

/** @brief Host-memory specialization backed by the process allocator. */
template <>
struct DeviceAllocator<Device::CPU>
{
    /** @brief Allocates at least `bytes` bytes of host memory. */
    static Result<void*> allocate(size_t bytes);
    /** @brief Releases a pointer returned by the CPU allocator; accepts `nullptr`. */
    static void  free(void* ptr);
};

/** @brief CUDA-memory specialization backed by the active device memory pool. */
template <>
struct DeviceAllocator<Device::CUDA>
{
    /**
     * @brief Initialize memory pool configuration for device.
     * @param device_id Device ID (-1 for current).
     * @param release_threshold_bytes Release threshold (UINT64_MAX to never release).
     */
    static Status init(int device_id = -1, size_t release_threshold_bytes = UINT64_MAX);

    /**
     * @brief Allocates device memory, optionally ordered on a CUDA stream.
     * @param bytes Number of bytes to allocate.
     * @param stream Optional `cudaStream_t` passed as an opaque pointer.
     * @return Device pointer, or a structured allocation error; zero bytes produce a null success value.
     */
    static Result<void*> allocate(size_t bytes, void* stream = nullptr);
    /**
     * @brief Releases device memory, optionally ordered on a CUDA stream.
     * @param ptr Pointer returned by this allocator; `nullptr` is accepted.
     * @param stream Optional `cudaStream_t` passed as an opaque pointer.
     */
    static void  free(void* ptr, void* stream = nullptr);
};

}  // namespace firefly
