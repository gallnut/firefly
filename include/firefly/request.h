#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace firefly
{
class TreeNode;

enum class RequestStatus
{
    PENDING,
    ACTIVE,
    FINISHED,
    FAILED
};

struct GenerationRequest
{
    std::string      id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int              max_tokens;
    RequestStatus    status = RequestStatus::PENDING;
    std::string      error_message;
    std::string      utf8_buffer;

    /** @brief Internal state for block allocation */
    std::vector<int>                       block_table;
    int                                    context_len = 0;
    std::vector<std::shared_ptr<TreeNode>> radix_nodes;

    std::chrono::steady_clock::time_point arrival_time;
    std::shared_ptr<std::atomic_bool>     cancel_flag;

    GenerationRequest(const std::string& id, const std::vector<int>& tokens, int max_tok,
                      std::shared_ptr<std::atomic_bool> cancel = nullptr)
        : id(id),
          prompt_tokens(tokens),
          max_tokens(max_tok),
          arrival_time(std::chrono::steady_clock::now()),
          cancel_flag(std::move(cancel))
    {
        context_len = 0;
    }

    int get_total_len() const { return prompt_tokens.size() + generated_tokens.size(); }
    bool is_cancelled() const { return cancel_flag && cancel_flag->load(std::memory_order_relaxed); }
};

using RequestPtr = std::shared_ptr<GenerationRequest>;

}  // namespace firefly
