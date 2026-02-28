#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace firefly
{

class BlockAllocator;

struct TreeNode
{
    std::vector<int> tokens;
    std::vector<int> block_indices;
    std::atomic<int> ref_count{0};
    uint64_t         last_access_time{0};

    std::unordered_map<int, std::shared_ptr<TreeNode>> children;

    TreeNode() = default;
};

class RadixTree
{
public:
    RadixTree(BlockAllocator* allocator);

    struct MatchResult
    {
        std::vector<std::shared_ptr<TreeNode>> matched_nodes;
        std::vector<int>                       matched_blocks;
        int                                    matched_tokens = 0;
    };

    /**
     * @brief Finds the longest matching prefix for a sequence of tokens.
     *
     * Automatically increments the ref_count of the matched nodes.
     *
     * @param tokens The sequence of tokens to match.
     * @return MatchResult The result of the match.
     */
    MatchResult match_prefix(const std::vector<int>& tokens);

    /**
     * @brief Inserts a sequence of tokens and their corresponding blocks.
     *
     * Handles splitting existing nodes if a partial match diverges.
     * Automatically decrements the ref_count if nodes were previously matched.
     *
     * @param tokens The full sequence of tokens to insert.
     * @param blocks The corresponding block indices.
     */
    void insert(const std::vector<int>& tokens, const std::vector<int>& blocks);

    /**
     * @brief Decrements the ref_count of given nodes.
     *
     * Call when a request finishes or is freed.
     *
     * @param nodes The nodes to decrement references for.
     */
    void decrement_ref_counts(const std::vector<std::shared_ptr<TreeNode>>& nodes);

    /**
     * @brief Evicts leaf nodes using LRU policy.
     *
     * Evicts leaf nodes with ref_count == 0 until `num_blocks_needed` is satisfied.
     *
     * @param num_blocks_needed The number of memory blocks required.
     * @return int The number of blocks successfully freed.
     */
    int evict(int num_blocks_needed);

    /**
     * @brief Prints caching statistics for the radix tree.
     */
    void print_stats() const;

private:
    std::shared_ptr<TreeNode> root_;
    BlockAllocator*           allocator_;
    mutable std::mutex        mutex_;

    uint64_t current_time_ = 0;

    /**
     * @brief Recursive eviction helper.
     *
     * @param node The node to evaluate.
     * @param num_blocks_needed Target blocks needed.
     * @return int Blocks freed.
     */
    int evict_recursive(std::shared_ptr<TreeNode> node, int num_blocks_needed);

    /**
     * @brief Helper to collect evictable leaves using LRU policy.
     *
     * @param node The current node to scan.
     * @param leaves Vector storing evictable leaf nodes.
     * @param parent The parent of the current node.
     */
    void collect_evictable_leaves(std::shared_ptr<TreeNode>                                                     node,
                                  std::vector<std::pair<std::shared_ptr<TreeNode>, std::shared_ptr<TreeNode>>>& leaves,
                                  std::shared_ptr<TreeNode>                                                     parent);
};

}  // namespace firefly
