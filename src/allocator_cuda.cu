#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "firefly/hal/memory.h"
#include "firefly/mm/allocator.h"

namespace firefly
{

using namespace firefly::hal;

void DeviceAllocator<Device::CUDA>::init(int device_id, size_t release_threshold_bytes)
{
    // Use HAL memory pool management
    auto result = memory::set_release_threshold(device_id, release_threshold_bytes);
    if (!result)
    {
        throw std::runtime_error("Failed to set memory pool release threshold: " +
                                 std::string(result.error().description()));
    }

    // Enable peer access if multiple devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 1)
    {
        int current_device = device_id;
        if (current_device < 0) cudaGetDevice(&current_device);

        for (int i = 0; i < device_count; ++i)
        {
            if (i != current_device)
            {
                // Attempt to enable peer access for memory pool optimization.
                // Errors are ignored as P2P might not be supported on all pairs.
                (void)memory::enable_peer_access(current_device, i);
            }
        }
    }
}

static thread_local cudaStream_t g_default_stream = nullptr;

void DeviceAllocator<Device::CUDA>::set_default_stream(void* stream)
{
    g_default_stream = static_cast<cudaStream_t>(stream);
}

void* DeviceAllocator<Device::CUDA>::allocate(size_t bytes, void* stream)
{
    cudaStream_t s = (stream == nullptr) ? g_default_stream : static_cast<cudaStream_t>(stream);

    auto result = memory::allocate(bytes, s);
    if (!result)
    {
        // Convert HAL error to exception to maintain interface compatibility
        throw std::runtime_error(std::string(result.error().description()));
    }
    return result.value();
}

void DeviceAllocator<Device::CUDA>::free(void* ptr, void* stream)
{
    cudaStream_t s = (stream == nullptr) ? g_default_stream : static_cast<cudaStream_t>(stream);
    memory::free(ptr, s);
}

}  // namespace firefly