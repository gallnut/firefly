#include "firefly/scheduler/block_allocator.h"

#include "firefly/scheduler/prefix_cache.h"

namespace firefly::scheduler
{

void BlockAllocator::init(int total_blocks, PrefixCache* prefix_cache)
{
    prefix_cache_ = prefix_cache;
    free_blocks_.clear();
    free_blocks_.reserve(total_blocks);
    for (int block = 0; block < total_blocks; ++block) free_blocks_.push_back(block);
}

bool BlockAllocator::allocate(int num_blocks, std::vector<int>& out_blocks)
{
    if (free_blocks_.size() < static_cast<size_t>(num_blocks) && prefix_cache_)
        prefix_cache_->evict(num_blocks - static_cast<int>(free_blocks_.size()));
    if (free_blocks_.size() < static_cast<size_t>(num_blocks)) return false;

    for (int index = 0; index < num_blocks; ++index)
    {
        out_blocks.push_back(free_blocks_.back());
        free_blocks_.pop_back();
    }
    return true;
}

void BlockAllocator::free(const std::vector<int>& blocks)
{
    free_blocks_.insert(free_blocks_.end(), blocks.begin(), blocks.end());
}

int BlockAllocator::free_block_count() const { return static_cast<int>(free_blocks_.size()); }

}  // namespace firefly::scheduler
