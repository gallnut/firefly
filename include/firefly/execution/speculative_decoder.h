#pragma once

#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"
#include "firefly/model/model.h"
#include "firefly/model/speculative.h"
#include "firefly/scheduler/sequence.h"

namespace firefly::execution
{

/** @brief Forward declaration of the execution engine whose caches are used during verification. */
class Engine;

/** @brief Configuration and non-owning proposer dependency for speculative decoding. */
struct SpeculativeDecoderOptions
{
    model::SpeculativeProposer* proposer = nullptr; ///< Borrowed proposer that must outlive the engine.
    int max_draft_tokens = 0; ///< Proposal limit; zero selects the proposer's trained block size.
    float confidence_threshold = 0.0f; ///< Prefix-pruning threshold in `[0, 1]`; zero disables pruning.
};

/**
 * @brief Model-agnostic speculative proposal, target verification, and commit coordinator.
 *
 * The decoder owns only generic execution state and compact per-request target context.
 * Draft architecture details remain behind `SpeculativeProposer`, while mutable target
 * state is handled through the optional `SpeculativeTargetRuntime` capability.
 */
class SpeculativeDecoder final
{
public:
    /**
     * @brief Constructs and validates a speculative target/proposer pairing.
     * @param target Borrowed target model used to verify draft tokens; it must outlive this decoder.
     * @param target_config Static target dimensions used to validate proposal tensors and hidden states.
     * @param target_runtime Target cache and batching requirements used to determine supported requests.
     * @param options Borrowed proposer, proposal limit, and optional confidence-pruning threshold.
     * @return Validated coordinator or a structured dependency or compatibility error.
     */
    static Result<std::unique_ptr<SpeculativeDecoder>> create(model::Model* target, model::ModelConfig target_config,
                                                              model::ModelRuntimeRequirements target_runtime,
                                                              SpeculativeDecoderOptions options);

    /**
     * @brief Reports whether the current implementation can speculatively process a request batch.
     * @param requests Active decode requests considered for one speculative cycle.
     * @return `true` when all batch and runtime constraints permit speculative decoding.
     */
    [[nodiscard]] bool can_decode(const std::vector<scheduler::SequencePtr>& requests) const;
    /**
     * @brief Returns scheduler lookahead required for the largest proposal block.
     * @return Maximum number of uncommitted draft-token slots needed per request.
     */
    [[nodiscard]] int max_draft_tokens() const { return max_draft_tokens_; }
    /**
     * @brief Returns sorted target layers captured as proposer context.
     * @return Non-owning view valid for the lifetime of this decoder.
     */
    [[nodiscard]] std::span<const int> target_hidden_layers() const { return target_hidden_layers_; }

    /**
     * @brief Allocates proposer-specific runtime state.
     * @param max_sequence_slots Maximum number of scheduler state slots used concurrently.
     * @param context Device and CUDA stream on which initialization work is enqueued.
     */
    Status initialize_runtime(int max_sequence_slots, const device::Context& context);
    /**
     * @brief Resets proposer-specific runtime state and discards captured target contexts.
     * @param context CUDA context used for any asynchronous device-side reset work.
     */
    Status reset_runtime(const device::Context& context);
    /**
     * @brief Allocates dense layered output and configures a target forward to populate it.
     * @param options Target forward options updated with the requested hidden layers and output buffer.
     * @param layered_hidden_states Receives owned storage shaped `[layers, batch, sequence, hidden]`.
     * @param batch_size Number of dense batch rows to capture.
     * @param sequence_length Number of padded token positions in each batch row.
     */
    Status configure_target_forward(model::ForwardOptions& options, Tensor& layered_hidden_states,
                                    int batch_size, int sequence_length) const;
    /**
     * @brief Allocates flattened ragged layered output and configures a target forward to populate it.
     * @param options Target forward options updated with the requested hidden layers and output buffer.
     * @param layered_hidden_states Receives owned storage shaped `[layers, total_tokens, hidden]`.
     * @param total_tokens Sum of the unpadded token counts in the ragged batch.
     */
    Status configure_target_ragged_forward(model::ForwardOptions& options, Tensor& layered_hidden_states,
                                           int total_tokens) const;
    /**
     * @brief Extracts one token's requested layer states per dense row into persistent request context.
     * @param layered_hidden_states Dense layered output produced by the configured target forward.
     * @param requests Requests corresponding to the leading batch rows.
     * @param sequence_length Padded sequence stride in `layered_hidden_states`.
     * @param context CUDA context used for asynchronous device-to-device copies.
     * @param token_index Token position to capture; negative values select the last position.
     * @param model_batch_size Physical model batch size, or zero to use `requests.size()`.
     * @note Captured tensors remain owned by the decoder until replaced, erased, or reset.
     */
    Status capture_target_context(const Tensor& layered_hidden_states,
                                  const std::vector<scheduler::SequencePtr>& requests,
                                  int sequence_length, const device::Context& context,
                                  int token_index = -1, int model_batch_size = 0);
    /**
     * @brief Extracts each ragged row's final token states into persistent request context.
     * @param layered_hidden_states Flattened layered output produced by a ragged target forward.
     * @param requests Requests corresponding one-to-one with the supplied offsets and lengths.
     * @param sequence_offsets Starting token offset of every request in the flattened tensor.
     * @param sequence_lengths Unpadded token count of every request.
     * @param context CUDA context used for asynchronous device-to-device copies.
     */
    Status capture_target_context_ragged(const Tensor& layered_hidden_states,
                                         const std::vector<scheduler::SequencePtr>& requests,
                                         std::span<const int> sequence_offsets,
                                         std::span<const int> sequence_lengths,
                                         const device::Context& context);
    /**
     * @brief Removes compact target context for a finished, cancelled, or failed request.
     * @param request_id Stable scheduler request identifier whose stored context is discarded.
     */
    void erase(const std::string& request_id);
    /**
     * @brief Executes one proposal, target verification, and accepted-prefix commit cycle.
     * @param engine Engine that owns target KV caches, scheduler state, and output processing.
     * @param requests Decode requests processed in the current batch.
     * @param context Device and CUDA stream on which proposal and verification are enqueued.
     * @return Success or a structured proposal, verification, allocation, or CUDA error.
     */
    Status decode(Engine& engine, const std::vector<scheduler::SequencePtr>& requests,
                  const device::Context& context);

private:
    SpeculativeDecoder(model::Model* target, model::ModelConfig target_config,
                       model::ModelRuntimeRequirements target_runtime, SpeculativeDecoderOptions options);

    model::Model* target_ = nullptr; ///< Borrowed target model used for verification.
    model::SpeculativeProposer* proposer_ = nullptr; ///< Borrowed architecture-specific draft proposer.
    model::SpeculativeTargetRuntime* transactional_runtime_ = nullptr; ///< Optional borrowed state snapshot capability.
    model::ModelConfig target_config_{}; ///< Cached target dimensions used by generic verification.
    model::ModelRuntimeRequirements target_runtime_{}; ///< Cached target batching and cache requirements.
    int max_draft_tokens_ = 0; ///< Effective proposal length after applying the configured limit.
    float confidence_threshold_ = 0.0f; ///< Minimum confidence retained during optional prefix pruning.
    std::vector<int> target_hidden_layers_; ///< Sorted target layers captured for the proposer.
    std::unordered_map<std::string, Tensor> target_contexts_; ///< Device context tensor indexed by request ID.
};

}  // namespace firefly::execution
