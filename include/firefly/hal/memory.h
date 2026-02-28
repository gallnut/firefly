#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "firefly/hal/error.h"

namespace firefly::hal::memory
{

/**
 * @brief Allocates device memory asynchronously on a stream.
 *
 * @param bytes Size in bytes to allocate.
 * @param stream The stream to perform allocation on.
 * @return Result<void*> The allocated pointer or error.
 */
inline Result<void*> allocate(size_t bytes, cudaStream_t stream = nullptr)
{
    void*       ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
    if (err != cudaSuccess)
    {
        return unexpected(Error{err, "CUDA OOM: Failed to allocate device memory"});
    }
    return ptr;
}

/**
 * @brief Frees device memory asynchronously on a stream.
 *
 * @param ptr Pointer to free.
 * @param stream The stream to perform deallocation on.
 */
inline void free(void* ptr, cudaStream_t stream = nullptr)
{
    if (ptr)
    {
        cudaFreeAsync(ptr, stream);
    }
}

/**
 * @brief Sets the memory pool release threshold for a device.
 *
 * @param device_id The device ID (-1 for current).
 * @param threshold_bytes The threshold (UINT64_MAX to prevent releasing).
 * @return Result<void> Success or failure.
 */
inline Result<void> set_release_threshold(int device_id, size_t threshold_bytes)
{
    int current_device = device_id;
    if (current_device < 0)
    {
        cudaError_t err = cudaGetDevice(&current_device);
        if (err != cudaSuccess) return unexpected(Error{err});
    }

    cudaMemPool_t mem_pool;
    cudaError_t   err = cudaDeviceGetDefaultMemPool(&mem_pool, current_device);
    if (err != cudaSuccess) return unexpected(Error{err});

    uint64_t threshold = static_cast<uint64_t>(threshold_bytes);
    err = cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (err != cudaSuccess) return unexpected(Error{err});

    return {};
}

/**
 * @brief Enables peer access for the memory pool of the current device.
 *
 * @param current_device The current device ID.
 * @param peer_device The peer device ID to allow access from.
 * @return Result<void> Success or failure.
 */
inline Result<void> enable_peer_access(int current_device, int peer_device)
{
    cudaMemPool_t mem_pool;
    cudaError_t   err = cudaDeviceGetDefaultMemPool(&mem_pool, current_device);
    if (err != cudaSuccess) return unexpected(Error{err});

    int can_access = 0;
    err = cudaDeviceCanAccessPeer(&can_access, current_device, peer_device);
    if (err != cudaSuccess) return unexpected(Error{err});

    if (can_access)
    {
        cudaMemAccessDesc desc;
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = peer_device;
        desc.flags = cudaMemAccessFlagsProtReadWrite;

        err = cudaMemPoolSetAccess(mem_pool, &desc, 1);
        if (err != cudaSuccess) return unexpected(Error{err});
    }

    return {};
}

}  // namespace firefly::hal::memory
