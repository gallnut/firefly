#include <cstdlib>
#include "firefly/device/allocator.h"

namespace firefly
{

Result<void*> DeviceAllocator<Device::CPU>::allocate(size_t bytes)
{
    void* ptr = std::malloc(bytes);
    if (!ptr && bytes > 0)
        return unexpected(Error{ErrorCode::ResourceExhausted, "failed to allocate host memory"});
    return ptr;
}

void DeviceAllocator<Device::CPU>::free(void* ptr) { std::free(ptr); }

}  // namespace firefly
