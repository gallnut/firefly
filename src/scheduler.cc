#include "firefly/scheduler.h"

#include <algorithm>
#include <iostream>

#include "firefly/radix_tree.h"

namespace firefly
{

BlockAllocator::BlockAllocator() : total_blocks_(0) {}

void BlockAllocator::init(int total_blocks, RadixTree* tree)
{
    total_blocks_ = total_blocks;
    radix_tree_ = tree;
    free_blocks_.clear();
    for (int i = 0; i < total_blocks; ++i)
    {
        free_blocks_.push_back(i);
    }
}

void BlockAllocator::set_radix_tree(RadixTree* tree) { radix_tree_ = tree; }

bool BlockAllocator::allocate(int num_blocks, std::vector<int>& out_blocks)
{
    if (free_blocks_.size() < (size_t)num_blocks)
    {
        if (radix_tree_)
        {
            int needed = num_blocks - free_blocks_.size();
            radix_tree_->evict(needed);
        }
        if (free_blocks_.size() < (size_t)num_blocks) return false;
    }

    for (int i = 0; i < num_blocks; ++i)
    {
        out_blocks.push_back(free_blocks_.back());
        free_blocks_.pop_back();
    }
    return true;
}

void BlockAllocator::free(const std::vector<int>& blocks)
{
    for (int b : blocks)
    {
        free_blocks_.push_back(b);
    }
}

int BlockAllocator::get_free_blocks() const { return free_blocks_.size(); }

Scheduler::Scheduler() {}

void Scheduler::init(int max_context_blocks, int max_batch_size_limit, int max_prefill_chunk_size)
{
    max_batch_size_limit_ = max_batch_size_limit;
    max_prefill_chunk_size_ = max_prefill_chunk_size;
    radix_tree_ = std::make_unique<RadixTree>(&block_allocator_);
    block_allocator_.init(max_context_blocks, radix_tree_.get());
}

void Scheduler::add_request(RequestPtr req)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        match = radix_tree_->match_prefix(req->prompt_tokens);
    req->block_table = match.matched_blocks;
    req->context_len = match.matched_tokens;
    req->radix_nodes = match.matched_nodes;
    pending_requests_.push_back(req);
}

bool Scheduler::has_unfinished_requests()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !pending_requests_.empty() || !active_requests_.empty();
}

SchedulerBatch Scheduler::step()
{
    std::lock_guard<std::mutex> lock(mutex_);
    SchedulerBatch              batch;

    auto pending_it = pending_requests_.begin();
    while (pending_it != pending_requests_.end())
    {
        auto req = *pending_it;
        if (req->is_cancelled())
        {
            req->status = RequestStatus::FAILED;
            req->error_message = "request cancelled";
            release_owned_blocks(req);
            pending_it = pending_requests_.erase(pending_it);
            continue;
        }
        ++pending_it;
    }

    // 1. Prioritize allocating 1 block for ACTIVE requests that need it
    auto active_it = active_requests_.begin();
    while (active_it != active_requests_.end())
    {
        auto req = *active_it;
        if (req->is_cancelled())
        {
            req->status = RequestStatus::FAILED;
            req->error_message = "request cancelled";
            release_owned_blocks(req);
            active_it = active_requests_.erase(active_it);
            continue;
        }

        int  current_blocks = req->block_table.size();
        int  tokens_to_process = 1;

        if (req->generated_tokens.empty())
        {
            int unmatched_tokens = req->prompt_tokens.size() - req->context_len;
            if (unmatched_tokens <= 0)
            {
                req->status = RequestStatus::FAILED;
                req->error_message = "active prefill request has no remaining prompt tokens";
                release_owned_blocks(req);
                batch.failed_requests.push_back(req);
                active_it = active_requests_.erase(active_it);
                continue;
            }
            tokens_to_process = std::min(unmatched_tokens, max_prefill_chunk_size_);
        }

        int target_len = req->context_len + tokens_to_process;
        int required_blocks = (target_len + block_size_ - 1) / block_size_ - current_blocks;
        required_blocks = std::max(required_blocks, 0);

        if (required_blocks > 0)
        {
            std::vector<int> new_block;
            if (block_allocator_.allocate(required_blocks, new_block))
            {
                req->block_table.insert(req->block_table.end(), new_block.begin(), new_block.end());
                batch.requests.push_back(req);
            }
            else
            {
                req->status = RequestStatus::FAILED;
                req->error_message = "KV cache exhausted while extending active request";
                release_owned_blocks(req);
                batch.failed_requests.push_back(req);
                active_it = active_requests_.erase(active_it);
                continue;
            }
        }
        else
        {
            batch.requests.push_back(req);
        }

        ++active_it;
    }

    // 2. Schedule PENDING requests for prefix if block memory allows
    // Cache-aware scheduling: sort pending by longest matched cache prefix
    pending_requests_.sort([](const RequestPtr& a, const RequestPtr& b) { return a->context_len > b->context_len; });

    auto it = pending_requests_.begin();
    while (it != pending_requests_.end())
    {
        auto req = *it;
        int  unmatched_tokens = req->prompt_tokens.size() - req->context_len;

        // Ensure at least 1 token is sent to the Engine to compute logits
        if (unmatched_tokens == 0 && !req->radix_nodes.empty())
        {
            auto last_node = req->radix_nodes.back();
            req->context_len -= last_node->tokens.size();
            req->block_table.resize(req->block_table.size() - last_node->block_indices.size());
            last_node->ref_count--;  // explicitly decrement
            req->radix_nodes.pop_back();
            unmatched_tokens = req->prompt_tokens.size() - req->context_len;
        }
        if (unmatched_tokens <= 0)
        {
            req->status = RequestStatus::FAILED;
            req->error_message = "request has no prompt tokens to prefill";
            release_owned_blocks(req);
            batch.failed_requests.push_back(req);
            it = pending_requests_.erase(it);
            continue;
        }

        // Chunk prefilling limit calculation
        // We only allocate blocks sufficient for the NEW chunk we can process.
        int chunk_size = std::min(unmatched_tokens, max_prefill_chunk_size_);
        int target_len = req->context_len + chunk_size;
        int required_blocks = (target_len + block_size_ - 1) / block_size_ - req->block_table.size();
        required_blocks = std::max(required_blocks, 0);

        // Safety bound: allow admission if we don't blow past active request limits
        // Wait, prefill requests do not immediately join the "decode_requests" block, but
        // they will become active during execution.
        if (active_requests_.size() < max_batch_size_limit_)
        {
            std::vector<int> blocks;
            if (required_blocks == 0 || block_allocator_.allocate(required_blocks, blocks))
            {
                req->block_table.insert(req->block_table.end(), blocks.begin(), blocks.end());
                req->status = RequestStatus::ACTIVE;
                active_requests_.push_back(req);
                batch.requests.push_back(req);
                it = pending_requests_.erase(it);
                continue;
            }
        }

        // KV Cache is too full to allocate blocks for this chunk, stop admitting pending requests
        break;
    }

    return batch;
}

void Scheduler::release_owned_blocks(RequestPtr req)
{
    size_t cached_block_count = 0;
    for (const auto& node : req->radix_nodes)
    {
        cached_block_count += node->block_indices.size();
    }
    cached_block_count = std::min(cached_block_count, req->block_table.size());

    if (cached_block_count < req->block_table.size())
    {
        std::vector<int> owned_blocks(req->block_table.begin() + cached_block_count, req->block_table.end());
        block_allocator_.free(owned_blocks);
    }

    if (radix_tree_)
    {
        radix_tree_->decrement_ref_counts(req->radix_nodes);
    }
    req->radix_nodes.clear();
    req->block_table.clear();
}

void Scheduler::finish_request(RequestPtr req)
{
    std::lock_guard<std::mutex> lock(mutex_);
    req->status = RequestStatus::FINISHED;

    // Decrement tree ref counts
    if (radix_tree_) radix_tree_->decrement_ref_counts(req->radix_nodes);
    req->radix_nodes.clear();

    // Insert new generated sequence back into RadixTree to be cached
    std::vector<int> all_tokens = req->prompt_tokens;
    all_tokens.insert(all_tokens.end(), req->generated_tokens.begin(), req->generated_tokens.end());
    int cached_blocks = 0;
    if (radix_tree_) cached_blocks = radix_tree_->insert(all_tokens, req->block_table);

    cached_blocks = std::min<int>(cached_blocks, req->block_table.size());
    if (cached_blocks < (int)req->block_table.size())
    {
        std::vector<int> uncached_blocks(req->block_table.begin() + cached_blocks, req->block_table.end());
        block_allocator_.free(uncached_blocks);
    }

    req->block_table.clear();  // blocks are now owned exclusively by the tree until evicted

    auto it = std::find(active_requests_.begin(), active_requests_.end(), req);
    if (it != active_requests_.end())
    {
        active_requests_.erase(it);
    }
}

void Scheduler::abort_request(RequestPtr req)
{
    std::lock_guard<std::mutex> lock(mutex_);
    req->status = RequestStatus::FAILED;
    release_owned_blocks(req);

    auto active_it = std::find(active_requests_.begin(), active_requests_.end(), req);
    if (active_it != active_requests_.end())
    {
        active_requests_.erase(active_it);
    }

    auto pending_it = std::find(pending_requests_.begin(), pending_requests_.end(), req);
    if (pending_it != pending_requests_.end())
    {
        pending_requests_.erase(pending_it);
    }
}

}  // namespace firefly
