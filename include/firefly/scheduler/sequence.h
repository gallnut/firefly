#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace firefly::scheduler
{
class PrefixCacheNode;

enum class SequenceStatus
{
    PENDING,
    ACTIVE,
    FINISHED,
    FAILED
};

struct Sequence
{
    std::string      id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int              max_tokens;
    bool             ignore_eos = false;
    SequenceStatus   status = SequenceStatus::PENDING;
    std::string      error_message;
    std::string      utf8_buffer;

    /** @brief Internal state for block allocation */
    std::vector<int>                       block_table;
    std::vector<int>                       owned_blocks;
    int                                    context_len = 0;
    std::vector<std::shared_ptr<PrefixCacheNode>> prefix_cache_nodes;
    int                                    prefix_cow_block_index = -1;
    int                                    prefix_cow_source_block = -1;
    int                                    prefix_cow_private_block = -1;
    bool                                   prefix_cow_copied = false;
    int                                    state_slot = -1;

    std::chrono::steady_clock::time_point arrival_time;
    std::shared_ptr<std::atomic_bool>     cancel_flag;

    Sequence(const std::string& id, const std::vector<int>& tokens, int max_tok,
             std::shared_ptr<std::atomic_bool> cancel = nullptr, bool ignore_end_token = false)
        : id(id),
          prompt_tokens(tokens),
          max_tokens(max_tok),
          ignore_eos(ignore_end_token),
          arrival_time(std::chrono::steady_clock::now()),
          cancel_flag(std::move(cancel))
    {
        context_len = 0;
    }

    int get_total_len() const { return prompt_tokens.size() + generated_tokens.size(); }
    bool is_cancelled() const { return cancel_flag && cancel_flag->load(std::memory_order_relaxed); }
};

using SequencePtr = std::shared_ptr<Sequence>;

}  // namespace firefly::scheduler
