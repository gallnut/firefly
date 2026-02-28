#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace firefly
{
class TreeNode;

enum class RequestStatus
{
    PENDING,
    ACTIVE,
    FINISHED
};

struct GenerationRequest
{
    std::string      id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int              max_tokens;
    RequestStatus    status = RequestStatus::PENDING;
    std::string      utf8_buffer;

    /** @brief Internal state for block allocation */
    std::vector<int>                       block_table;
    int                                    context_len = 0;
    std::vector<std::shared_ptr<TreeNode>> radix_nodes;

    std::chrono::steady_clock::time_point arrival_time;

    GenerationRequest(const std::string& id, const std::vector<int>& tokens, int max_tok)
        : id(id), prompt_tokens(tokens), max_tokens(max_tok), arrival_time(std::chrono::steady_clock::now())
    {
        context_len = 0;
    }

    int get_total_len() const { return prompt_tokens.size() + generated_tokens.size(); }
};

using RequestPtr = std::shared_ptr<GenerationRequest>;

}  // namespace firefly
