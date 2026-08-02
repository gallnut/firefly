#pragma once

#include <vector>

namespace firefly::scheduler
{

class PrefixCache;

class BlockAllocator
{
public:
    void init(int total_blocks, PrefixCache* prefix_cache = nullptr);
    bool allocate(int num_blocks, std::vector<int>& out_blocks);
    void free(const std::vector<int>& blocks);
    int  free_block_count() const;

private:
    std::vector<int> free_blocks_;
    PrefixCache*     prefix_cache_ = nullptr;
};

}  // namespace firefly::scheduler
