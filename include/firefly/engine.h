#pragma once

#include <atomic>
#include <thread>
#include <unordered_map>

#include "firefly/hal/graph.h"
#include "firefly/model/model.h"
#include "firefly/queue.h"
#include "firefly/scheduler.h"
#include "firefly/tokenizer.h"

namespace firefly
{

class Engine
{
public:
    Engine(model::Model* model, const model::ModelConfig& config, const Tokenizer& tokenizer,
           ResultQueue* result_queue = nullptr, int max_prefill_chunk_size = 256);
    ~Engine();

    /**
     * @brief Pushes a new generation request into the engine asynchronously.
     *
     * This method is thread-safe.
     *
     * @param id Unique identifier for the request.
     * @param input_ids Tokenized input sequence.
     * @param max_tokens Maximum number of tokens to generate.
     */
    void async_generate(const std::string& id, const std::vector<int>& input_ids, int max_tokens);

    void start();
    void stop();

private:
    void loop();

    model::Model*      model_;
    model::ModelConfig config_;
    Tokenizer          tokenizer_;
    ResultQueue*       result_queue_;

    Scheduler         scheduler_;
    std::thread       background_thread_;
    std::atomic<bool> running_{false};

    /** @brief Key caches for the model */
    std::vector<Tensor> k_caches_;
    std::vector<Tensor> v_caches_;

    int max_context_blocks_ = 1024;
    int max_prefill_chunk_size_ = 256;

    struct GraphData
    {
        Tensor     input_ids;
        Tensor     context_lens;
        Tensor     block_table;
        Tensor     next_tokens;
        hal::Graph graph;

        int* h_input_ids = nullptr;
        int* h_context_lens = nullptr;
        int* h_block_table = nullptr;
        int* h_next_tokens = nullptr;
    };

    /** @brief Pre-allocated CUDA graphs for decode phase */
    std::unordered_map<int, GraphData> dec_graphs_;
    std::vector<int>                   supported_batch_sizes_ = {1, 2, 4, 8, 16, 32, 64};
};

}  // namespace firefly
