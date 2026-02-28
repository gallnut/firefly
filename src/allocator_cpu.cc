#include <cstdlib>
#include <stdexcept>

#include "firefly/mm/allocator.h"

namespace firefly
{

void* DeviceAllocator<Device::CPU>::allocate(size_t bytes)
{
    void* ptr = std::malloc(bytes);
    if (!ptr && bytes > 0) throw std::runtime_error("CPU OOM");
    return ptr;
}

void DeviceAllocator<Device::CPU>::free(void* ptr) { std::free(ptr); }

}  // namespace firefly