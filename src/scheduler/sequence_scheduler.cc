#include "firefly/scheduler/sequence_scheduler.h"

#include <algorithm>
#include <numeric>

#include "firefly/scheduler/prefix_cache.h"

namespace firefly::scheduler
{

SequenceScheduler::SequenceScheduler() {}

void SequenceScheduler::init(int max_context_blocks, int max_batch_size_limit, int max_prefill_chunk_size,
                             bool prefix_cache_enabled, int speculative_tokens)
{
    max_batch_size_limit_ = max_batch_size_limit;
    max_prefill_chunk_size_ = max_prefill_chunk_size;
    speculative_tokens_ = std::max(speculative_tokens, 0);
    prefix_cache_ = prefix_cache_enabled ? std::make_unique<PrefixCache>(&block_allocator_) : nullptr;
    block_allocator_.init(max_context_blocks, prefix_cache_.get());
    free_state_slots_.resize(max_batch_size_limit_);
    std::iota(free_state_slots_.rbegin(), free_state_slots_.rend(), 0);
}

void SequenceScheduler::add_sequence(SequencePtr req)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (prefix_cache_)
    {
        auto match = prefix_cache_->match(req->prompt_tokens);
        req->block_table = match.matched_blocks;
        req->context_len = match.matched_tokens;
        req->prefix_cache_nodes = match.matched_nodes;
    }
    pending_sequences_.push_back(req);
}

bool SequenceScheduler::has_unfinished_sequences()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !pending_sequences_.empty() || !active_sequences_.empty();
}

bool SequenceScheduler::has_pending_sequences()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !pending_sequences_.empty();
}

bool SequenceScheduler::has_active_decode_sequences()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(active_sequences_.begin(), active_sequences_.end(),
                       [](const SequencePtr& request) { return !request->generated_tokens.empty(); });
}

BatchPlan SequenceScheduler::step()
{
    std::lock_guard<std::mutex> lock(mutex_);
    BatchPlan                   batch;

    auto pending_it = pending_sequences_.begin();
    while (pending_it != pending_sequences_.end())
    {
        auto req = *pending_it;
        if (req->is_cancelled())
        {
            req->status = SequenceStatus::FAILED;
            req->error_message = "request cancelled";
            release_owned_blocks(req);
            pending_it = pending_sequences_.erase(pending_it);
            continue;
        }
        ++pending_it;
    }

    // 1. Prioritize allocating 1 block for ACTIVE requests that need it
    auto active_it = active_sequences_.begin();
    while (active_it != active_sequences_.end())
    {
        auto req = *active_it;
        if (req->is_cancelled())
        {
            req->status = SequenceStatus::FAILED;
            req->error_message = "request cancelled";
            release_owned_blocks(req);
            active_it = active_sequences_.erase(active_it);
            continue;
        }

        int  current_blocks = req->block_table.size();
        int  tokens_to_process = req->generated_tokens.empty() ? 1 : (1 + speculative_tokens_);

        if (req->generated_tokens.empty())
        {
            int unmatched_tokens = req->prompt_tokens.size() - req->context_len;
            if (unmatched_tokens <= 0)
            {
                req->status = SequenceStatus::FAILED;
                req->error_message = "active prefill request has no remaining prompt tokens";
                release_owned_blocks(req);
                batch.failed_sequences.push_back(req);
                active_it = active_sequences_.erase(active_it);
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
                req->owned_blocks.insert(req->owned_blocks.end(), new_block.begin(), new_block.end());
                batch.sequences.push_back(req);
            }
            else
            {
                req->status = SequenceStatus::FAILED;
                req->error_message = "KV cache exhausted while extending active request";
                release_owned_blocks(req);
                batch.failed_sequences.push_back(req);
                active_it = active_sequences_.erase(active_it);
                continue;
            }
        }
        else
        {
            batch.sequences.push_back(req);
        }

        ++active_it;
    }

    // 2. Schedule PENDING requests for prefix if block memory allows
    // Cache-aware scheduling: sort pending by longest matched cache prefix
    pending_sequences_.sort([](const SequencePtr& a, const SequencePtr& b) { return a->context_len > b->context_len; });

    auto it = pending_sequences_.begin();
    while (it != pending_sequences_.end())
    {
        auto req = *it;
        int  unmatched_tokens = req->prompt_tokens.size() - req->context_len;

        // Ensure at least 1 token is sent to the Engine to compute logits
        if (unmatched_tokens == 0 && !req->prefix_cache_nodes.empty())
        {
            int replay_pos = static_cast<int>(req->prompt_tokens.size()) - 1;
            int cow_block_index = replay_pos / block_size_;
            if (cow_block_index < 0 || cow_block_index >= static_cast<int>(req->block_table.size()))
            {
                req->status = SequenceStatus::FAILED;
                req->error_message = "prefix cache hit has invalid block table";
                release_owned_blocks(req);
                batch.failed_sequences.push_back(req);
                it = pending_sequences_.erase(it);
                continue;
            }

            if (req->prefix_cow_private_block < 0)
            {
                std::vector<int> cow_block;
                if (!block_allocator_.allocate(1, cow_block))
                {
                    break;
                }

                req->prefix_cow_block_index = cow_block_index;
                req->prefix_cow_source_block = req->block_table[cow_block_index];
                req->prefix_cow_private_block = cow_block[0];
                req->prefix_cow_copied = false;
                req->block_table[cow_block_index] = cow_block[0];
                req->owned_blocks.push_back(cow_block[0]);
            }

            req->context_len = replay_pos;
            unmatched_tokens = req->prompt_tokens.size() - req->context_len;
        }
        if (unmatched_tokens <= 0)
        {
            req->status = SequenceStatus::FAILED;
            req->error_message = "request has no prompt tokens to prefill";
            release_owned_blocks(req);
            batch.failed_sequences.push_back(req);
            it = pending_sequences_.erase(it);
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
        if (active_sequences_.size() < max_batch_size_limit_)
        {
            std::vector<int> blocks;
            if (!free_state_slots_.empty() && (required_blocks == 0 || block_allocator_.allocate(required_blocks, blocks)))
            {
                req->block_table.insert(req->block_table.end(), blocks.begin(), blocks.end());
                req->owned_blocks.insert(req->owned_blocks.end(), blocks.begin(), blocks.end());
                req->state_slot = free_state_slots_.back();
                free_state_slots_.pop_back();
                req->status = SequenceStatus::ACTIVE;
                active_sequences_.push_back(req);
                batch.sequences.push_back(req);
                it = pending_sequences_.erase(it);
                continue;
            }
        }

        // KV Cache is too full to allocate blocks for this chunk, stop admitting pending requests
        break;
    }

    return batch;
}

void SequenceScheduler::release_owned_blocks(SequencePtr req)
{
    if (!req->owned_blocks.empty())
    {
        block_allocator_.free(req->owned_blocks);
    }

    if (prefix_cache_)
    {
        prefix_cache_->release(req->prefix_cache_nodes);
    }
    req->prefix_cache_nodes.clear();
    req->block_table.clear();
    req->owned_blocks.clear();
    req->prefix_cow_block_index = -1;
    req->prefix_cow_source_block = -1;
    req->prefix_cow_private_block = -1;
    req->prefix_cow_copied = false;
    release_state_slot(req);
}

void SequenceScheduler::release_state_slot(SequencePtr req)
{
    if (req->state_slot < 0) return;
    free_state_slots_.push_back(req->state_slot);
    req->state_slot = -1;
}

void SequenceScheduler::finish_sequence(SequencePtr req)
{
    std::lock_guard<std::mutex> lock(mutex_);
    req->status = SequenceStatus::FINISHED;

    // Decrement tree ref counts
    if (prefix_cache_) prefix_cache_->release(req->prefix_cache_nodes);
    req->prefix_cache_nodes.clear();

    // Insert only KV-materialized tokens back into the prefix cache. The last sampled
    // token has not been appended to KV cache yet.
    std::vector<int> all_tokens = req->prompt_tokens;
    int materialized_generated = std::max(0, req->context_len - static_cast<int>(req->prompt_tokens.size()));
    materialized_generated = std::min<int>(materialized_generated, req->generated_tokens.size());
    all_tokens.insert(all_tokens.end(), req->generated_tokens.begin(),
                      req->generated_tokens.begin() + materialized_generated);

    std::vector<int> cache_block_table = req->block_table;
    if (req->prefix_cow_block_index >= 0 && req->prefix_cow_block_index < static_cast<int>(cache_block_table.size()) &&
        req->prefix_cow_source_block >= 0)
    {
        cache_block_table[req->prefix_cow_block_index] = req->prefix_cow_source_block;
    }

    PrefixCache::InsertResult insert_result;
    if (prefix_cache_) insert_result = prefix_cache_->insert(all_tokens, cache_block_table);

    int cached_blocks = std::min<int>(insert_result.cached_blocks, req->block_table.size());
    int adopted_block_start = std::min<int>(insert_result.adopted_block_start, cached_blocks);

    std::vector<int> blocks_to_free;
    for (size_t i = 0; i < req->block_table.size(); ++i)
    {
        int  block = req->block_table[i];
        bool owned = std::find(req->owned_blocks.begin(), req->owned_blocks.end(), block) != req->owned_blocks.end();
        if (!owned) continue;

        bool is_prefix_cow =
            req->prefix_cow_private_block == block && static_cast<int>(i) == req->prefix_cow_block_index;
        bool adopted_by_tree = static_cast<int>(i) >= adopted_block_start && static_cast<int>(i) < cached_blocks;
        if (is_prefix_cow || !adopted_by_tree)
        {
            blocks_to_free.push_back(block);
        }
    }
    if (!blocks_to_free.empty()) block_allocator_.free(blocks_to_free);

    req->block_table.clear();
    req->owned_blocks.clear();
    req->prefix_cow_block_index = -1;
    req->prefix_cow_source_block = -1;
    req->prefix_cow_private_block = -1;
    req->prefix_cow_copied = false;
    release_state_slot(req);

    auto it = std::find(active_sequences_.begin(), active_sequences_.end(), req);
    if (it != active_sequences_.end())
    {
        active_sequences_.erase(it);
    }
}

void SequenceScheduler::abort_sequence(SequencePtr req)
{
    std::lock_guard<std::mutex> lock(mutex_);
    req->status = SequenceStatus::FAILED;
    release_owned_blocks(req);

    auto active_it = std::find(active_sequences_.begin(), active_sequences_.end(), req);
    if (active_it != active_sequences_.end())
    {
        active_sequences_.erase(active_it);
    }

    auto pending_it = std::find(pending_sequences_.begin(), pending_sequences_.end(), req);
    if (pending_it != pending_sequences_.end())
    {
        pending_sequences_.erase(pending_it);
    }
}

}  // namespace firefly::scheduler
