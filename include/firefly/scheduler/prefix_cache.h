#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace firefly::scheduler
{

class BlockAllocator;

struct PrefixCacheNode
{
    std::vector<int> tokens;
    std::vector<int> block_indices;
    std::atomic<int> ref_count{0};
    uint64_t         last_access_time{0};
    std::unordered_map<int, std::shared_ptr<PrefixCacheNode>> children;
};

class PrefixCache
{
public:
    explicit PrefixCache(BlockAllocator* allocator);

    struct MatchResult
    {
        std::vector<std::shared_ptr<PrefixCacheNode>> matched_nodes;
        std::vector<int>                              matched_blocks;
        int                                           matched_tokens = 0;
    };

    struct InsertResult
    {
        int cached_blocks = 0;
        int adopted_block_start = 0;
    };

    MatchResult match(const std::vector<int>& tokens);
    InsertResult insert(const std::vector<int>& tokens, const std::vector<int>& blocks);
    void release(const std::vector<std::shared_ptr<PrefixCacheNode>>& nodes);
    int  evict(int num_blocks_needed);
    void print_stats() const;

private:
    std::shared_ptr<PrefixCacheNode> root_;
    BlockAllocator*                  allocator_;
    mutable std::mutex               mutex_;
    uint64_t                         current_time_ = 0;

    void collect_evictable_leaves(
        std::shared_ptr<PrefixCacheNode> node,
        std::vector<std::pair<std::shared_ptr<PrefixCacheNode>, std::shared_ptr<PrefixCacheNode>>>& leaves,
        std::shared_ptr<PrefixCacheNode> parent);
};

}  // namespace firefly::scheduler
