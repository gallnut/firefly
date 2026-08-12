#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "firefly/device/graph.h"
#include "firefly/execution/result_queue.h"
#include "firefly/execution/speculative_decoder.h"
#include "firefly/model/model.h"
#include "firefly/model/tokenizer.h"
#include "firefly/scheduler/sequence_scheduler.h"

namespace firefly::execution
{

/** @brief Storage format used for paged key/value cache tensors. */
enum class KVCacheFormat
{
    Model, ///< Store keys and values in the model activation dtype.
    Int8   ///< Store quantized keys and values with per-token floating-point scales.
};

/** @brief Configurable engine limits, cache policy, and optional speculative capability. */
struct EngineOptions
{
    int           max_prefill_chunk_size = 256; ///< Maximum prompt tokens processed per sequence and step.
    KVCacheFormat kv_cache_format = KVCacheFormat::Model; ///< Paged KV storage representation.
    double        gpu_memory_utilization = 0.90; ///< Fraction of post-profile free GPU memory reserved for KV blocks.
    SpeculativeDecoderOptions speculative; ///< Optional proposer; null leaves ordinary execution unchanged.
};

/**
 * @brief Asynchronous single-device inference engine with continuous request scheduling.
 *
 * The engine borrows the loaded model and optional result queue, owns runtime caches and
 * CUDA graphs, and executes scheduling/model work on one background thread. Public
 * submission and lifecycle methods are synchronized for service-side use.
 */
class Engine
{
public:
    /**
     * @brief Constructs an idle engine around an already loaded model.
     * @param model Borrowed target model that must outlive the engine.
     * @param config Target model dimensions.
     * @param tokenizer Tokenizer copied into the engine for streaming decode.
     * @param result_queue Borrowed output queue, or null to discard streamed results.
     * @param options Runtime and cache configuration.
     */
    static Result<std::unique_ptr<Engine>> create(model::Model* model, const model::ModelConfig& config,
                                                  const model::Tokenizer& tokenizer,
                                                  ResultQueue* result_queue = nullptr, EngineOptions options = {});
    /** @brief Stops the worker thread and releases runtime resources. */
    ~Engine();

    /**
     * @brief Pushes a new generation request into the engine asynchronously.
     *
     * This method is thread-safe.
     *
     * @param id Unique identifier for the request.
     * @param input_ids Tokenized input sequence.
     * @param max_tokens Maximum number of tokens to generate.
     * @param cancel_flag Optional shared cancellation flag observed by the scheduler.
     * @param ignore_eos Whether generation should continue through tokenizer stop tokens.
     */
    void async_generate(const std::string& id, const std::vector<int>& input_ids, int max_tokens,
                        std::shared_ptr<std::atomic_bool> cancel_flag = nullptr, bool ignore_eos = false);

    /** @brief Profiles memory, allocates caches, captures eligible graphs, and starts the worker thread. */
    Status start();
    /** @brief Idempotently requests worker termination and joins the background thread. */
    void stop();

private:
    Engine(model::Model* model, const model::ModelConfig& config, const model::Tokenizer& tokenizer,
           ResultQueue* result_queue, EngineOptions options);

    /**
     * @brief Runs the background scheduling and execution loop until `stop` clears `running_`.
     * @note This is the sole thread that mutates model runtime state and device caches.
     */
    void loop() noexcept;
    /**
     * @brief Runs grouped chunked-prefill forwards and samples first completion tokens.
     * @param requests Prefill requests selected by the scheduler for this step.
     * @param context CUDA stream and lifetime context for all enqueued operations.
     */
    Status process_prefill(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context);
    /**
     * @brief Selects speculative, dynamic, or static-graph decode for active sequences.
     * @param requests Decode requests selected by the scheduler for this step.
     * @param context CUDA stream and lifetime context for all enqueued operations.
     */
    Status process_decode(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context);
    /**
     * @brief Runs ragged prefill and decode rows in one model forward.
     * @param requests Mixed request batch in the row order consumed by the model.
     * @param context CUDA stream and lifetime context for all enqueued operations.
     */
    Status process_mixed_batch(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context);
    /**
     * @brief Executes shape-dynamic decode with optional split attention and runtime graph caching.
     * @param requests Decode requests in model batch order.
     * @param max_context_length Largest token context length in the batch.
     * @param prefer_split_decode Whether to prefer the split paged-decode attention backend.
     * @param context CUDA stream and lifetime context for all enqueued operations.
     */
    Status process_dynamic_decode(const std::vector<scheduler::SequencePtr>& requests, int max_context_length,
                                  bool prefer_split_decode, const device::Context& context);
    /**
     * @brief Replays a pre-captured decode graph for a supported padded batch size.
     * @param requests Decode requests copied into the graph's leading rows.
     * @param target_batch_size Captured physical batch size, including padding rows.
     * @param context CUDA stream on which inputs are copied and the graph is launched.
     * @return `true` when a compatible graph was available and launched.
     */
    Result<bool> process_static_decode_graph(const std::vector<scheduler::SequencePtr>& requests,
                                             int target_batch_size, const device::Context& context);
    /**
     * @brief Materializes private copies before writing into shared prefix-cache blocks.
     * @param sequences Requests whose next KV write may target a shared cached block.
     * @param context CUDA stream used for asynchronous key/value and scale copies.
     */
    Status copy_prefix_cow_blocks(const std::vector<scheduler::SequencePtr>& sequences,
                                  const device::Context& context);
    /**
     * @brief Converts newly generated tokens into UTF-8 result items and retires finished requests.
     * @param sequences Requests whose device-produced next-token IDs are ready on the host.
     */
    void process_outputs(const std::vector<scheduler::SequencePtr>& sequences);
    /** @brief Marks affected requests failed and publishes terminal errors. */
    void fail_requests(const std::vector<scheduler::SequencePtr>& requests, const Error& error);
    /** @brief Grants the generic speculative coordinator access to engine-owned caches and output handling. */
    friend class SpeculativeDecoder;

    model::Model*      model_; ///< Borrowed target model executed exclusively by the worker thread.
    model::ModelConfig config_; ///< Immutable target dimensions used for cache sizing and batching.
    model::Tokenizer   tokenizer_; ///< Owned tokenizer used to stream generated token text.
    ResultQueue*       result_queue_; ///< Borrowed service queue, or null when output streaming is disabled.
    EngineOptions      options_; ///< Runtime limits and cache/speculative configuration.
    model::ModelRuntimeRequirements runtime_requirements_; ///< Capabilities reported by the target model.
    std::unique_ptr<SpeculativeDecoder> speculative_decoder_; ///< Optional generic speculative coordinator.

    scheduler::SequenceScheduler scheduler_; ///< Owns request queues, cache blocks, and runtime slot assignments.
    std::thread                  background_thread_; ///< Worker executing `loop` while the engine is running.
    std::atomic<bool>            running_{false}; ///< Cooperative worker termination flag.
    std::mutex                   lifecycle_mutex_; ///< Serializes `start` and `stop` transitions.

    std::vector<Tensor> k_caches_; ///< Per-attention-layer paged key-cache storage.
    std::vector<Tensor> v_caches_; ///< Per-attention-layer paged value-cache storage.
    std::vector<Tensor> kv_scale_caches_; ///< Optional per-layer INT8 key/value quantization scales.

    int max_context_blocks_ = 1024; ///< Maximum logical KV pages represented in one request block-table row.
    int decode_scratch_first_block_ = 0; ///< First physical KV block reserved for speculative verification scratch.
    int decode_scratch_blocks_ = 0; ///< Number of physical KV blocks reserved outside scheduler ownership.

    Tensor  decode_fb_input_storage_; ///< Reusable token-ID input for shape-dynamic decode.
    Tensor  decode_fb_context_storage_; ///< Reusable context-length input for shape-dynamic decode.
    Tensor  decode_fb_block_table_storage_; ///< Reusable paged-cache block table for shape-dynamic decode.
    Tensor  decode_fb_next_token_storage_; ///< Reusable sampled-token output for shape-dynamic decode.
    Tensor  decode_fb_state_slot_storage_; ///< Reusable recurrent-state slot mapping for dynamic decode.
    int64_t decode_fb_input_capacity_ = 0; ///< Allocated element capacity of `decode_fb_input_storage_`.
    int64_t decode_fb_context_capacity_ = 0; ///< Allocated element capacity of `decode_fb_context_storage_`.
    int64_t decode_fb_block_table_capacity_ = 0; ///< Allocated element capacity of the dynamic block table.
    int64_t decode_fb_next_token_capacity_ = 0; ///< Allocated element capacity of sampled-token storage.
    int64_t decode_fb_state_slot_capacity_ = 0; ///< Allocated element capacity of recurrent slot storage.

    device::Graph    flashinfer_decode_graph_; ///< Runtime-captured dynamic decode graph for model-dtype KV cache.
    Tensor           flashinfer_decode_layered_hidden_states_; ///< Captured hidden context output for speculation.
    std::vector<int> flashinfer_decode_pages_; ///< Physical KV pages fixed into the current runtime graph plan.
    int              flashinfer_decode_batch_size_ = 0; ///< Batch shape captured by `flashinfer_decode_graph_`.
    int              flashinfer_decode_max_blocks_ = 0; ///< Block-table row stride captured by the runtime graph.

    device::Graph quantized_decode_graph_; ///< Runtime-captured dynamic decode graph for INT8 KV cache.
    Tensor        quantized_decode_layered_hidden_states_; ///< Captured hidden context output for speculation.
    int           quantized_decode_batch_size_ = 0; ///< Batch shape captured by the quantized graph.
    int           quantized_decode_max_blocks_ = 0; ///< Block-table row stride captured by the quantized graph.
    int           quantized_decode_context_bucket_ = 0; ///< Context-length planning bucket captured by the graph.

    /** @brief Persistent device and pinned-host buffers captured by one static decode graph. */
    struct GraphData
    {
        Tensor        input_ids; ///< Persistent device token-ID input captured by the graph.
        Tensor        context_lens; ///< Persistent device context lengths captured by the graph.
        Tensor        block_table; ///< Persistent device paged-cache mapping captured by the graph.
        Tensor        next_tokens; ///< Persistent device sampling output captured by the graph.
        Tensor        layered_hidden_states; ///< Optional hidden context output captured for speculation.
        device::Graph graph; ///< Executable static decode graph for this padded batch size.

        int* h_input_ids = nullptr; ///< Page-locked host staging buffer for token IDs.
        int* h_context_lens = nullptr; ///< Page-locked host staging buffer for context lengths.
        int* h_block_table = nullptr; ///< Page-locked host staging buffer for block-table rows.
        int* h_next_tokens = nullptr; ///< Page-locked host destination for sampled token IDs.
    };

    std::unordered_map<int, GraphData> dec_graphs_; ///< Static decode graph and buffers keyed by padded batch size.
    std::vector<int> supported_batch_sizes_ = {1, 2, 4, 8, 16, 32, 64}; ///< Batch sizes eligible for static capture.
};

}  // namespace firefly::execution
