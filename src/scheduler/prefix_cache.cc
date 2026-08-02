#include "firefly/scheduler/prefix_cache.h"

#include <algorithm>
#include <iostream>

#include "firefly/scheduler/block_allocator.h"

namespace firefly::scheduler
{

PrefixCache::PrefixCache(BlockAllocator* allocator) : allocator_(allocator)
{
    root_ = std::make_shared<PrefixCacheNode>();
}

PrefixCache::MatchResult PrefixCache::match(const std::vector<int>& tokens)
{
    std::lock_guard<std::mutex> lock(mutex_);
    current_time_++;

    MatchResult               result;
    std::shared_ptr<PrefixCacheNode> current = root_;
    int                       token_idx = 0;

    while (token_idx < (int)tokens.size())
    {
        int next_token = tokens[token_idx];
        if (current->children.find(next_token) == current->children.end())
        {
            break;  // No further matching path
        }

        auto child = current->children[next_token];

        // Check how much of the edge matches
        int match_len = 0;
        int max_match = std::min(child->tokens.size(), tokens.size() - token_idx);
        while (match_len < max_match && child->tokens[match_len] == tokens[token_idx + match_len])
        {
            match_len++;
        }

        int block_aligned_match = (match_len / 16) * 16;
        if (block_aligned_match == 0)
        {
            // Cannot even safely match one block, so we must stop matching early
            // completely avoiding shared physical block corruption.
            break;
        }

        if (block_aligned_match < (int)child->tokens.size())
        {
            // We matched partially but cleanly on a block boundary.
            // Split the node precisely at `block_aligned_match`.
            auto new_child = std::make_shared<PrefixCacheNode>();
            new_child->tokens = std::vector<int>(child->tokens.begin() + block_aligned_match, child->tokens.end());

            int blocks_before_split = block_aligned_match / 16;

            new_child->block_indices =
                std::vector<int>(child->block_indices.begin() + blocks_before_split, child->block_indices.end());
            child->block_indices.resize(blocks_before_split);

            new_child->children = std::move(child->children);
            new_child->ref_count.store(child->ref_count.load());
            new_child->last_access_time = child->last_access_time;

            child->tokens.resize(block_aligned_match);
            child->children.clear();
            child->children[new_child->tokens[0]] = new_child;

            // Re-point child since we dynamically split it for the exact block match
            child->ref_count++;
            child->last_access_time = current_time_;
            result.matched_nodes.push_back(child);
            result.matched_blocks.insert(result.matched_blocks.end(), child->block_indices.begin(),
                                         child->block_indices.end());
            result.matched_tokens += child->tokens.size();

            // Token idx advanced to the divergence point
            token_idx += block_aligned_match;
            break;
        }

        // Full match on this node's edge
        child->ref_count++;
        child->last_access_time = current_time_;
        result.matched_nodes.push_back(child);
        result.matched_blocks.insert(result.matched_blocks.end(), child->block_indices.begin(),
                                     child->block_indices.end());
        result.matched_tokens += child->tokens.size();

        token_idx += match_len;
        current = child;
    }

    return result;
}

void PrefixCache::release(const std::vector<std::shared_ptr<PrefixCacheNode>>& nodes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    current_time_++;
    for (auto& node : nodes)
    {
        if (node->ref_count > 0)
        {
            node->ref_count--;
            node->last_access_time = current_time_;
        }
    }
}

PrefixCache::InsertResult PrefixCache::insert(const std::vector<int>& tokens, const std::vector<int>& blocks)
{
    std::lock_guard<std::mutex> lock(mutex_);
    current_time_++;

    int cacheable_blocks = std::min<int>(blocks.size(), tokens.size() / 16);
    int cacheable_tokens = cacheable_blocks * 16;
    if (cacheable_blocks == 0)
    {
        return {};
    }

    std::shared_ptr<PrefixCacheNode> current = root_;
    int                       token_idx = 0;
    int                       block_idx = 0;

    while (token_idx < cacheable_tokens)
    {
        int next_token = tokens[token_idx];
        if (current->children.find(next_token) == current->children.end())
        {
            break;
        }

        auto child = current->children[next_token];

        int match_len = 0;
        int max_match = std::min<int>(child->tokens.size(), cacheable_tokens - token_idx);
        while (match_len < max_match && child->tokens[match_len] == tokens[token_idx + match_len])
        {
            match_len++;
        }

        if (match_len < (int)child->tokens.size())
        {
            int block_aligned_match = (match_len / 16) * 16;
            if (block_aligned_match == 0)
            {
                return {block_idx, block_idx};
            }

            auto new_child = std::make_shared<PrefixCacheNode>();
            new_child->tokens = std::vector<int>(child->tokens.begin() + block_aligned_match, child->tokens.end());

            int blocks_before_split = block_aligned_match / 16;

            if (blocks_before_split < (int)child->block_indices.size())
            {
                new_child->block_indices =
                    std::vector<int>(child->block_indices.begin() + blocks_before_split, child->block_indices.end());
                child->block_indices.resize(blocks_before_split);
            }

            new_child->children = std::move(child->children);
            new_child->ref_count.store(child->ref_count.load());
            new_child->last_access_time = child->last_access_time;

            child->tokens.resize(block_aligned_match);
            child->children.clear();
            child->children[new_child->tokens[0]] = new_child;

            token_idx += block_aligned_match;
            block_idx += blocks_before_split;
            current = child;
            continue;
        }

        token_idx += match_len;
        block_idx += match_len / 16;
        current = child;
    }

    // Insert new suffix if any
    if (token_idx < cacheable_tokens)
    {
        auto new_node = std::make_shared<PrefixCacheNode>();
        new_node->tokens = std::vector<int>(tokens.begin() + token_idx, tokens.begin() + cacheable_tokens);

        if (block_idx < cacheable_blocks)
        {
            new_node->block_indices = std::vector<int>(blocks.begin() + block_idx, blocks.begin() + cacheable_blocks);
        }

        new_node->last_access_time = current_time_;
        current->children[new_node->tokens[0]] = new_node;
        return {cacheable_blocks, block_idx};
    }

    return {block_idx, block_idx};
}

void PrefixCache::collect_evictable_leaves(
    std::shared_ptr<PrefixCacheNode> node,
    std::vector<std::pair<std::shared_ptr<PrefixCacheNode>, std::shared_ptr<PrefixCacheNode>>>& leaves,
    std::shared_ptr<PrefixCacheNode> parent)
{
    if (node->children.empty() && node != root_)
    {
        if (node->ref_count == 0)
        {
            leaves.push_back({node, parent});
        }
        return;
    }
    for (auto& pair : node->children)
    {
        collect_evictable_leaves(pair.second, leaves, node);
    }
}

int PrefixCache::evict(int num_blocks_needed)
{
    std::lock_guard<std::mutex> lock(mutex_);
    int                         freed_blocks = 0;

    while (freed_blocks < num_blocks_needed)
    {
        std::vector<std::pair<std::shared_ptr<PrefixCacheNode>, std::shared_ptr<PrefixCacheNode>>> evictable_leaves;
        collect_evictable_leaves(root_, evictable_leaves, nullptr);

        if (evictable_leaves.empty())
        {
            break;  // Nothing left to evict
        }

        // Sort by LRU (oldest first). Tie-break by deepest block depth if needed.
        std::sort(evictable_leaves.begin(), evictable_leaves.end(),
                  [](const auto& a, const auto& b) { return a.first->last_access_time < b.first->last_access_time; });

        auto target = evictable_leaves.front();
        auto leaf = target.first;
        auto parent = target.second;

        freed_blocks += leaf->block_indices.size();
        allocator_->free(leaf->block_indices);

        if (parent)
        {
            parent->children.erase(leaf->tokens[0]);
        }
    }

    return freed_blocks;
}

void PrefixCache::print_stats() const { std::cout << "[PrefixCache] stats unavailable\n"; }

}  // namespace firefly::scheduler
