#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace firefly::scheduler
{

/** @brief Forward declaration of the allocator that owns cached physical blocks. */
class BlockAllocator;

/** @brief One radix-tree node representing a cacheable full KV page of tokens. */
struct PrefixCacheNode
{
    std::vector<int> tokens; ///< Token page represented by this node.
    std::vector<int> block_indices; ///< Physical KV blocks materializing the path through this node.
    std::atomic<int> ref_count{0}; ///< Number of live sequences borrowing this cached prefix.
    uint64_t         last_access_time{0}; ///< Monotonic timestamp used for LRU eviction.
    std::unordered_map<int, std::shared_ptr<PrefixCacheNode>> children; ///< Next-page branches keyed by first token.
};

/** @brief Thread-safe radix-tree cache that shares immutable full KV pages across requests. */
class PrefixCache
{
public:
    /**
     * @brief Constructs an empty prefix tree borrowing its physical block allocator.
     * @param allocator Allocator that receives blocks released by cache eviction.
     */
    explicit PrefixCache(BlockAllocator* allocator);

    /** @brief Result of longest-prefix lookup and reference acquisition. */
    struct MatchResult
    {
        std::vector<std::shared_ptr<PrefixCacheNode>> matched_nodes; ///< Nodes whose references were acquired.
        std::vector<int>                              matched_blocks; ///< Physical blocks forming the cached prefix.
        int                                           matched_tokens = 0; ///< Number of prompt tokens already materialized.
    };

    /** @brief Result of inserting materialized full pages into the radix tree. */
    struct InsertResult
    {
        int cached_blocks = 0; ///< Number of leading block-table entries represented by the tree.
        int adopted_block_start = 0; ///< First caller-owned block whose ownership transferred to the tree.
    };

    /**
     * @brief Finds the longest full-page prefix and acquires references to matched nodes.
     * @param tokens Complete request prompt used for page-wise radix lookup.
     * @return Matched path, physical blocks, and reusable token count.
     */
    MatchResult match(const std::vector<int>& tokens);
    /**
     * @brief Inserts full materialized pages and may adopt caller-owned physical blocks.
     * @param tokens Prompt tokens whose full pages have been materialized.
     * @param blocks Physical blocks corresponding to consecutive prompt pages.
     * @return Cache coverage and the first block whose ownership transferred from the caller.
     */
    InsertResult insert(const std::vector<int>& tokens, const std::vector<int>& blocks);
    /**
     * @brief Releases references previously acquired by `match`.
     * @param nodes Matched nodes retained by one retiring request.
     */
    void release(const std::vector<std::shared_ptr<PrefixCacheNode>>& nodes);
    /**
     * @brief Evicts least-recently-used unreferenced leaves until enough blocks are freed.
     * @param num_blocks_needed Target number of blocks to return to the allocator.
     * @return Actual number of physical blocks released.
     */
    int  evict(int num_blocks_needed);
    /** @brief Emits current tree and hit statistics for diagnostics. */
    void print_stats() const;

private:
    std::shared_ptr<PrefixCacheNode> root_; ///< Sentinel root of the page-wise radix tree.
    BlockAllocator*                  allocator_; ///< Borrowed owner of physical block identifiers.
    mutable std::mutex               mutex_; ///< Serializes tree, reference, timestamp, and statistic mutations.
    uint64_t                         current_time_ = 0; ///< Logical clock incremented on cache accesses.

    /**
     * @brief Collects unreferenced leaf candidates and their parents for LRU eviction.
     * @param node Current subtree root.
     * @param leaves Destination list of evictable child-parent pairs.
     * @param parent Parent of `node`, or null at the sentinel root.
     * @pre The caller holds `mutex_`.
     */
    void collect_evictable_leaves(
        std::shared_ptr<PrefixCacheNode> node,
        std::vector<std::pair<std::shared_ptr<PrefixCacheNode>, std::shared_ptr<PrefixCacheNode>>>& leaves,
        std::shared_ptr<PrefixCacheNode> parent);
};

}  // namespace firefly::scheduler
