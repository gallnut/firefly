#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "firefly/device/graph.h"
#include "firefly/execution/result_queue.h"
#include "firefly/model/model.h"
#include "firefly/model/tokenizer.h"
#include "firefly/scheduler/sequence_scheduler.h"

namespace firefly::execution
{

enum class KVCacheFormat
{
    Model,
    Int8
};

struct EngineOptions
{
    int           max_prefill_chunk_size = 256;
    KVCacheFormat kv_cache_format = KVCacheFormat::Model;
    double        gpu_memory_utilization = 0.90;
};

class Engine
{
public:
    Engine(model::Model* model, const model::ModelConfig& config, const model::Tokenizer& tokenizer,
           ResultQueue* result_queue = nullptr, EngineOptions options = {});
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
    void async_generate(const std::string& id, const std::vector<int>& input_ids, int max_tokens,
                        std::shared_ptr<std::atomic_bool> cancel_flag = nullptr, bool ignore_eos = false);

    void start();
    void stop();

private:
    void loop();
    void process_prefill(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context);
    void process_decode(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context);
    void process_mixed_batch(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context);
    void process_dynamic_decode(const std::vector<scheduler::SequencePtr>& requests, int max_context_length,
                                bool prefer_split_decode, const device::Context& context);
    bool process_static_decode_graph(const std::vector<scheduler::SequencePtr>& requests, int target_batch_size,
                                     const device::Context& context);
    void copy_prefix_cow_blocks(const std::vector<scheduler::SequencePtr>& sequences, const device::Context& context);
    void process_outputs(const std::vector<scheduler::SequencePtr>& sequences);

    model::Model*      model_;
    model::ModelConfig config_;
    model::Tokenizer   tokenizer_;
    ResultQueue*       result_queue_;
    EngineOptions      options_;
    model::ModelRuntimeRequirements runtime_requirements_;

    scheduler::SequenceScheduler scheduler_;
    std::thread                  background_thread_;
    std::atomic<bool>            running_{false};
    std::mutex                   lifecycle_mutex_;

    /** @brief Key caches for the model */
    std::vector<Tensor> k_caches_;
    std::vector<Tensor> v_caches_;
    std::vector<Tensor> kv_scale_caches_;

    int max_context_blocks_ = 1024;
    int decode_scratch_first_block_ = 0;
    int decode_scratch_blocks_ = 0;

    Tensor  decode_fb_input_storage_;
    Tensor  decode_fb_context_storage_;
    Tensor  decode_fb_block_table_storage_;
    Tensor  decode_fb_next_token_storage_;
    Tensor  decode_fb_state_slot_storage_;
    int64_t decode_fb_input_capacity_ = 0;
    int64_t decode_fb_context_capacity_ = 0;
    int64_t decode_fb_block_table_capacity_ = 0;
    int64_t decode_fb_next_token_capacity_ = 0;
    int64_t decode_fb_state_slot_capacity_ = 0;

    device::Graph    flashinfer_decode_graph_;
    std::vector<int> flashinfer_decode_pages_;
    int              flashinfer_decode_batch_size_ = 0;
    int              flashinfer_decode_max_blocks_ = 0;

    device::Graph quantized_decode_graph_;
    int           quantized_decode_batch_size_ = 0;
    int           quantized_decode_max_blocks_ = 0;
    int           quantized_decode_context_bucket_ = 0;

    struct GraphData
    {
        Tensor        input_ids;
        Tensor        context_lens;
        Tensor        block_table;
        Tensor        next_tokens;
        device::Graph graph;

        int* h_input_ids = nullptr;
        int* h_context_lens = nullptr;
        int* h_block_table = nullptr;
        int* h_next_tokens = nullptr;
    };

    /** @brief Pre-allocated CUDA graphs for decode phase */
    std::unordered_map<int, GraphData> dec_graphs_;
    std::vector<int>                   supported_batch_sizes_ = {1, 2, 4, 8, 16, 32, 64};
};

}  // namespace firefly::execution
