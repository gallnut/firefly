#include <cuda_runtime.h>

#include <cstddef>

#include "firefly/execution/engine.h"

namespace firefly::execution
{
void Engine::copy_prefix_cow_blocks(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context)
{
    size_t block_bytes = 16 * static_cast<size_t>(runtime_requirements_.kv_cache_head_count) *
                         static_cast<size_t>(runtime_requirements_.kv_cache_head_dim) *
                         dtype_size(options_.kv_cache_format == KVCacheFormat::Int8 ? DType::I8 : config_.dtype);
    cudaStream_t stream = context.stream();

    for (const auto& req : requests)
    {
        if (req->prefix_cow_copied || req->prefix_cow_source_block < 0 || req->prefix_cow_private_block < 0)
        {
            continue;
        }

        size_t src_offset = static_cast<size_t>(req->prefix_cow_source_block) * block_bytes;
        size_t dst_offset = static_cast<size_t>(req->prefix_cow_private_block) * block_bytes;
        for (size_t layer = 0; layer < k_caches_.size(); ++layer)
        {
            auto* k_base = static_cast<std::byte*>(k_caches_[layer].data());
            auto* v_base = static_cast<std::byte*>(v_caches_[layer].data());
            cudaMemcpyAsync(k_base + dst_offset, k_base + src_offset, block_bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_base + dst_offset, v_base + src_offset, block_bytes, cudaMemcpyDeviceToDevice, stream);
            if (options_.kv_cache_format == KVCacheFormat::Int8)
            {
                size_t scale_block_bytes =
                    16 * static_cast<size_t>(runtime_requirements_.kv_cache_head_count) * 2 * sizeof(float);
                auto*  scale_base = static_cast<std::byte*>(kv_scale_caches_[layer].data());
                size_t scale_offset = static_cast<size_t>(req->prefix_cow_private_block) * scale_block_bytes;
                size_t scale_src_offset = static_cast<size_t>(req->prefix_cow_source_block) * scale_block_bytes;
                cudaMemcpyAsync(scale_base + scale_offset, scale_base + scale_src_offset, scale_block_bytes,
                                cudaMemcpyDeviceToDevice, stream);
            }
        }
        req->prefix_cow_copied = true;
    }
}
}  // namespace firefly::execution
