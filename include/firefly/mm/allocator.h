#pragma once
#include <cstddef>

#include "firefly/types.h"

namespace firefly
{

template <Device D>
struct DeviceAllocator;

template <>
struct DeviceAllocator<Device::CPU>
{
    static void* allocate(size_t bytes);
    static void  free(void* ptr);
};

template <>
struct DeviceAllocator<Device::CUDA>
{
    /**
     * @brief Initialize memory pool configuration for device.
     * @param device_id Device ID (-1 for current).
     * @param release_threshold_bytes Release threshold (UINT64_MAX to never release).
     */
    static void init(int device_id = -1, size_t release_threshold_bytes = UINT64_MAX);

    static void set_default_stream(void* stream);

    static void* allocate(size_t bytes, void* stream = nullptr);
    static void  free(void* ptr, void* stream = nullptr);
};

}  // namespace firefly