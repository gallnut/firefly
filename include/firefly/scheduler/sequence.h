#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace firefly::scheduler
{
/** @brief Forward declaration of prefix nodes retained by an active sequence. */
class PrefixCacheNode;

/** @brief Lifecycle state of one generation request inside the scheduler. */
enum class SequenceStatus
{
    PENDING,  ///< Awaiting admission and runtime resources.
    ACTIVE,   ///< Admitted and eligible for prefill or decode scheduling.
    FINISHED, ///< Successfully reached EOS or the output-token limit.
    FAILED    ///< Cancelled or unable to acquire required runtime resources.
};

/** @brief Mutable scheduler and generation state for one request. */
struct Sequence
{
    std::string      id; ///< Caller-provided request identifier.
    std::vector<int> prompt_tokens; ///< Immutable tokenized prompt.
    std::vector<int> generated_tokens; ///< Sampled completion tokens, including one not-yet-materialized tail token.
    int              max_tokens; ///< Maximum completion tokens requested by the caller.
    bool             ignore_eos = false; ///< Whether tokenizer stop tokens are treated as ordinary output.
    SequenceStatus   status = SequenceStatus::PENDING; ///< Current scheduler lifecycle state.
    std::string      error_message; ///< Terminal scheduler diagnostic for failed requests.
    std::string      utf8_buffer; ///< Incomplete UTF-8 bytes retained between token emissions.
    size_t           emitted_tokens = 0; ///< Number of generated tokens already sent to the result queue.

    /** @brief Internal state for block allocation */
    std::vector<int> block_table; ///< Logical-to-physical paged KV mapping.
    std::vector<int> owned_blocks; ///< Physical blocks released or adopted when the sequence retires.
    int context_len = 0; ///< Tokens already materialized in KV and recurrent state.
    std::vector<std::shared_ptr<PrefixCacheNode>> prefix_cache_nodes; ///< Referenced shared prefix path.
    int prefix_cow_block_index = -1; ///< Logical page replaced by a private copy-on-write block.
    int prefix_cow_source_block = -1; ///< Shared physical block copied before replay or extension.
    int prefix_cow_private_block = -1; ///< Private physical destination block.
    bool prefix_cow_copied = false; ///< Whether device KV contents have been copied to the private block.
    int state_slot = -1; ///< Recurrent-model runtime slot assigned while active.

    std::chrono::steady_clock::time_point arrival_time; ///< Monotonic request-arrival timestamp.
    std::shared_ptr<std::atomic_bool> cancel_flag; ///< Optional cross-thread cancellation signal.

    /**
     * @brief Constructs a pending request and records its arrival time.
     * @param id Caller-provided unique request identifier.
     * @param tokens Tokenized prompt copied into sequence ownership.
     * @param max_tok Maximum number of completion tokens to generate.
     * @param cancel Optional shared cross-thread cancellation signal.
     * @param ignore_end_token Whether tokenizer stop tokens are emitted as ordinary output.
     */
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

    /** @brief Returns prompt plus generated token count, including the unmaterialized tail token. */
    int get_total_len() const { return prompt_tokens.size() + generated_tokens.size(); }
    /** @brief Returns whether the optional cancellation flag has been set. */
    bool is_cancelled() const { return cancel_flag && cancel_flag->load(std::memory_order_relaxed); }
};

/** @brief Shared request handle used by scheduler batches and the engine thread. */
using SequencePtr = std::shared_ptr<Sequence>;

}  // namespace firefly::scheduler
