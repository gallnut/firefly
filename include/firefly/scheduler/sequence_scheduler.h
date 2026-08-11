#pragma once

#include <list>
#include <mutex>
#include <vector>

#include "firefly/scheduler/block_allocator.h"
#include "firefly/scheduler/prefix_cache.h"
#include "firefly/scheduler/sequence.h"

namespace firefly::scheduler
{

struct BatchPlan
{
    std::vector<SequencePtr> sequences;
    std::vector<SequencePtr> failed_sequences;
};

class SequenceScheduler
{
public:
    SequenceScheduler();
    void init(int max_context_blocks, int max_batch_size_limit, int max_prefill_chunk_size,
              bool prefix_cache_enabled = true);

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

    void finish_sequence(SequencePtr sequence);
    void abort_sequence(SequencePtr sequence);

    bool has_unfinished_sequences();
    bool has_pending_sequences();
    bool has_active_decode_sequences();

private:
    std::mutex            mutex_;
    std::list<SequencePtr> pending_sequences_;
    std::list<SequencePtr> active_sequences_;

    BlockAllocator              block_allocator_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    int                        block_size_ = 16;
    int                        max_batch_size_limit_ = 64;
    int                        max_prefill_chunk_size_ = 256;
    std::vector<int>           free_state_slots_;

    void release_owned_blocks(SequencePtr sequence);
    void release_state_slot(SequencePtr sequence);
};

}  // namespace firefly::scheduler
