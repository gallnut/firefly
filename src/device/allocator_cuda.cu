#include <cuda_runtime.h>

#include "firefly/device/allocator.h"
#include "firefly/device/memory.h"

namespace firefly
{

using namespace firefly::device;

Status DeviceAllocator<Device::CUDA>::init(int device_id, size_t release_threshold_bytes)
{
    FIREFLY_TRY_CONTEXT(memory::set_release_threshold(device_id, release_threshold_bytes),
                        "configure CUDA memory pool release threshold");

    int device_count = 0;
    FIREFLY_TRY(device::check_cuda(cudaGetDeviceCount(&device_count), "query CUDA device count"));
    if (device_count > 1)
    {
        int current_device = device_id;
        if (current_device < 0)
            FIREFLY_TRY(device::check_cuda(cudaGetDevice(&current_device), "query current CUDA device"));

        for (int i = 0; i < device_count; ++i)
        {
            if (i != current_device)
            {
                auto peer_status = memory::enable_peer_access(current_device, i);
                if (!peer_status && peer_status.error().code() != ErrorCode::Unavailable)
                    return unexpected(std::move(peer_status.error()).with_context("enable CUDA memory-pool peer access"));
            }
        }
    }
    return {};
}

Result<void*> DeviceAllocator<Device::CUDA>::allocate(size_t bytes, void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    return memory::allocate(bytes, s);
}

void DeviceAllocator<Device::CUDA>::free(void* ptr, void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    memory::free(ptr, s);
}

}  // namespace firefly
