#pragma once

#include <vector>

namespace firefly::scheduler
{

/** @brief Forward declaration of the cache consulted when free blocks are insufficient. */
class PrefixCache;

/** @brief Scheduler-owned allocator for integer paged-KV block identifiers. */
class BlockAllocator
{
public:
    /**
     * @brief Initializes the free list and optional prefix-cache eviction delegate.
     * @param total_blocks Number of physical block identifiers in the managed pool.
     * @param prefix_cache Borrowed cache asked to evict unreferenced blocks under pressure.
     */
    void init(int total_blocks, PrefixCache* prefix_cache = nullptr);
    /**
     * @brief Allocates blocks, evicting unused prefixes if required.
     * @param num_blocks Number of physical block identifiers requested.
     * @param out_blocks Destination receiving allocated identifiers on success.
     * @return `true` when the complete request was satisfied; partial allocations are not exposed.
     */
    bool allocate(int num_blocks, std::vector<int>& out_blocks);
    /**
     * @brief Returns block identifiers to the free list.
     * @param blocks Identifiers no longer referenced by requests or the prefix cache.
     */
    void free(const std::vector<int>& blocks);
    /**
     * @brief Returns the number of immediately available block identifiers.
     * @return Free-list size without triggering prefix-cache eviction.
     */
    int  free_block_count() const;

private:
    std::vector<int> free_blocks_; ///< Stack of physical block identifiers available for allocation.
    PrefixCache*     prefix_cache_ = nullptr; ///< Borrowed eviction delegate, or null when caching is disabled.
};

}  // namespace firefly::scheduler
