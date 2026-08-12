#pragma once

#include <list>
#include <mutex>
#include <vector>

#include "firefly/scheduler/block_allocator.h"
#include "firefly/scheduler/prefix_cache.h"
#include "firefly/scheduler/sequence.h"

namespace firefly::scheduler
{

/** @brief Requests selected for one engine iteration plus requests failed during planning. */
struct BatchPlan
{
    std::vector<SequencePtr> sequences; ///< Active prefill and decode requests with required resources reserved.
    std::vector<SequencePtr> failed_sequences; ///< Requests retired with a terminal scheduling error.
};

/**
 * @brief Thread-safe continuous-batching scheduler for paged KV and recurrent-state slots.
 *
 * Admission, block extension, prefix-cache references, and slot ownership are serialized
 * under the scheduler mutex. Returned sequence objects remain shared with the engine.
 */
class SequenceScheduler
{
public:
    /** @brief Constructs an uninitialized scheduler with no allocatable blocks or slots. */
    SequenceScheduler();
    /**
     * @brief Initializes block capacity, admission limits, prefix caching, and speculative lookahead.
     * @param max_context_blocks Total physical KV blocks managed by the scheduler.
     * @param max_batch_size_limit Maximum number of simultaneously active requests.
     * @param max_prefill_chunk_size Maximum prompt tokens scheduled per request and step.
     * @param prefix_cache_enabled Whether full prompt pages may be shared and evicted.
     * @param speculative_tokens Additional writable token slots reserved for verification.
     */
    void init(int max_context_blocks, int max_batch_size_limit, int max_prefill_chunk_size,
              bool prefix_cache_enabled = true, int speculative_tokens = 0);

    /**
     * @brief Adds a request to the pending queue after optional prefix lookup.
     * @param sequence Shared mutable request state; it must be non-null and pending.
     * @threadsafe Serialized with scheduling and retirement operations.
     */
    void add_sequence(SequencePtr sequence);

    /**
     * @brief Core function: builds the next batch to run.
     *
     * This function promotes pending requests to active status, and
     * allocates blocks that map to active inputs.
     *
     * @return BatchPlan Requests selected for the next execution step.
     */
    BatchPlan step();

    /** @brief Retires a successful sequence, updates prefix cache, and releases owned resources. */
    void finish_sequence(SequencePtr sequence);
    /** @brief Retires a cancelled or failed sequence and releases all owned resources. */
    void abort_sequence(SequencePtr sequence);

    /** @brief Returns whether pending or active sequences remain. */
    bool has_unfinished_sequences();
    /** @brief Returns whether at least one request awaits admission. */
    bool has_pending_sequences();
    /** @brief Returns whether at least one active request has entered decode. */
    bool has_active_decode_sequences();

private:
    std::mutex            mutex_; ///< Serializes request queues and all resource ownership transitions.
    std::list<SequencePtr> pending_sequences_; ///< Requests awaiting admission and runtime resources.
    std::list<SequencePtr> active_sequences_; ///< Admitted requests eligible for engine execution.

    BlockAllocator              block_allocator_; ///< Allocates physical paged-KV block identifiers.
    std::unique_ptr<PrefixCache> prefix_cache_; ///< Optional page-wise immutable prefix-sharing cache.
    int                        block_size_ = 16; ///< Tokens stored in one physical KV cache block.
    int                        max_batch_size_limit_ = 64; ///< Maximum number of admitted requests.
    int                        max_prefill_chunk_size_ = 256; ///< Per-request prompt-token scheduling cap.
    int                        speculative_tokens_ = 0; ///< Extra writable positions reserved for verification.
    std::vector<int>           free_state_slots_; ///< Stack of available recurrent-model state slots.

    /**
     * @brief Releases owned blocks, prefix references, copy-on-write metadata, and state slot.
     * @param sequence Retiring request whose resources are returned.
     * @pre The caller holds `mutex_`.
     */
    void release_owned_blocks(SequencePtr sequence);
    /**
     * @brief Returns a recurrent-state slot to the free-slot stack.
     * @param sequence Request whose nonnegative slot is released and reset to `-1`.
     * @pre The caller holds `mutex_`.
     */
    void release_state_slot(SequencePtr sequence);
};

}  // namespace firefly::scheduler
