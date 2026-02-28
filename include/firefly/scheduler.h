#pragma once

#include <list>
#include <mutex>
#include <vector>

#include "firefly/radix_tree.h"
#include "firefly/request.h"

namespace firefly
{

class BlockAllocator
{
public:
    BlockAllocator();
    void init(int total_blocks, RadixTree* tree = nullptr);
    void set_radix_tree(RadixTree* tree);

    /**
     * @brief Allocates raw blocks and returns block IDs.
     *
     * @param num_blocks The number of blocks to allocate.
     * @param out_blocks Vector populated with resulting block indices.
     * @return bool True if successful, false otherwise.
     */
    bool allocate(int num_blocks, std::vector<int>& out_blocks);
    void free(const std::vector<int>& blocks);
    int  get_free_blocks() const;

private:
    std::vector<int> free_blocks_;
    int              total_blocks_;
    RadixTree*       radix_tree_{nullptr};
};

struct SchedulerBatch
{
    std::vector<RequestPtr> requests;
};

class Scheduler
{
public:
    Scheduler();
    void init(int max_context_blocks, int max_batch_size_limit, int max_prefill_chunk_size);

    void add_request(RequestPtr req);

    /**
     * @brief Core function: builds the next batch to run.
     *
     * This function promotes pending requests to active status, and
     * allocates blocks that map to active inputs.
     *
     * @return SchedulerBatch The batch containing requests to execute.
     */
    SchedulerBatch step();

    void finish_request(RequestPtr req);

    bool has_unfinished_requests();

private:
    std::mutex            mutex_;
    std::list<RequestPtr> pending_requests_;
    std::list<RequestPtr> active_requests_;

    BlockAllocator             block_allocator_;
    std::unique_ptr<RadixTree> radix_tree_;
    int                        block_size_ = 16;
    int                        max_batch_size_limit_ = 64;
    int                        max_prefill_chunk_size_ = 256;
};

}  // namespace firefly
