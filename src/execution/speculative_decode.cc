#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "firefly/core/logging.h"
#include "firefly/device/error.h"
#include "firefly/execution/engine.h"
#include "firefly/kernels/attention/attention.h"
#include "firefly/kernels/sampling/argmax.h"

namespace firefly::execution
{
namespace
{
Result<std::vector<int>> greedy_rows(const Tensor& logits, int rows, int vocab_size,
                                     const device::Context& context)
{
    Tensor logits_rows = Tensor::from_external(logits.data(), {rows, vocab_size}, logits.dtype(), Device::CUDA);
    Tensor tokens = FIREFLY_TRY(Tensor::create({rows}, DType::I32, Device::CUDA, context));
    FIREFLY_TRY(kernels::argmax(logits_rows, tokens, context));
    std::vector<int> result(rows);
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(result.data(), tokens.data(), result.size() * sizeof(int), cudaMemcpyDeviceToHost,
                        context.stream()),
        "copy speculative greedy tokens to host"));
    FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(context.stream()),
                                   "synchronize speculative greedy tokens"));
    return result;
}

Result<std::vector<float>> copy_confidence(const Tensor& confidence, int count,
                                           const device::Context& context)
{
    std::vector<float> result(count);
    if (confidence.dtype() == DType::F32)
    {
        FIREFLY_TRY(device::check_cuda(
            cudaMemcpyAsync(result.data(), confidence.data(), count * sizeof(float), cudaMemcpyDeviceToHost,
                            context.stream()),
            "copy speculative float confidence to host"));
    }
    else if (confidence.dtype() == DType::BF16)
    {
        std::vector<__nv_bfloat16> values(count);
        FIREFLY_TRY(device::check_cuda(
            cudaMemcpyAsync(values.data(), confidence.data(), count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost,
                            context.stream()),
            "copy speculative bfloat16 confidence to host"));
        FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(context.stream()),
                                       "synchronize speculative bfloat16 confidence"));
        std::transform(values.begin(), values.end(), result.begin(), __bfloat162float);
        return result;
    }
    else if (confidence.dtype() == DType::F16)
    {
        std::vector<half> values(count);
        FIREFLY_TRY(device::check_cuda(
            cudaMemcpyAsync(values.data(), confidence.data(), count * sizeof(half), cudaMemcpyDeviceToHost,
                            context.stream()),
            "copy speculative float16 confidence to host"));
        FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(context.stream()),
                                       "synchronize speculative float16 confidence"));
        std::transform(values.begin(), values.end(), result.begin(), __half2float);
        return result;
    }
    else
    {
        return unexpected(Error{ErrorCode::InvalidArgument, "unsupported speculative confidence dtype"});
    }
    FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(context.stream()),
                                   "synchronize speculative float confidence"));
    return result;
}

}  // namespace

SpeculativeDecoder::SpeculativeDecoder(model::Model* target, model::ModelConfig target_config,
                                       model::ModelRuntimeRequirements target_runtime,
                                       SpeculativeDecoderOptions options)
    : target_(target),
      proposer_(options.proposer),
      transactional_runtime_(dynamic_cast<model::SpeculativeTargetRuntime*>(target)),
      target_config_(target_config),
      target_runtime_(target_runtime),
      confidence_threshold_(std::clamp(options.confidence_threshold, 0.0f, 1.0f))
{
}

Result<std::unique_ptr<SpeculativeDecoder>> SpeculativeDecoder::create(
    model::Model* target, model::ModelConfig target_config, model::ModelRuntimeRequirements target_runtime,
    SpeculativeDecoderOptions options)
{
    if (target == nullptr || options.proposer == nullptr)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "speculative decoder requires target and proposer capabilities"});
    FIREFLY_TRY_CONTEXT(options.proposer->validate_target({.hidden_size = target_config.hidden_size,
                                                          .vocab_size = target_config.vocab_size,
                                                          .num_hidden_layers = target_config.num_hidden_layers}),
                        "validate speculative target compatibility");
    auto decoder = std::unique_ptr<SpeculativeDecoder>(
        new SpeculativeDecoder(target, target_config, target_runtime, options));
    decoder->target_hidden_layers_.assign(options.proposer->target_hidden_layers().begin(),
                                          options.proposer->target_hidden_layers().end());
    if (decoder->target_hidden_layers_.empty() ||
        !std::is_sorted(decoder->target_hidden_layers_.begin(), decoder->target_hidden_layers_.end()) ||
        decoder->target_hidden_layers_.front() < 0 ||
        decoder->target_hidden_layers_.back() >= target_config.num_hidden_layers)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "speculative proposer target hidden layers are invalid"});
    if (std::adjacent_find(decoder->target_hidden_layers_.begin(), decoder->target_hidden_layers_.end()) !=
        decoder->target_hidden_layers_.end())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "speculative proposer target hidden layers contain duplicates"});
    decoder->max_draft_tokens_ = options.max_draft_tokens > 0 ? options.max_draft_tokens
                                                              : options.proposer->block_size();
    if (decoder->max_draft_tokens_ <= 0 || decoder->max_draft_tokens_ > options.proposer->block_size())
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "speculative draft token count exceeds proposer block size"});
    if (target_runtime.sequence_state && decoder->transactional_runtime_ == nullptr)
        return unexpected(Error{ErrorCode::Model,
                                "stateful speculative target requires transactional runtime capability"});
    return decoder;
}

bool SpeculativeDecoder::can_decode(const std::vector<scheduler::SequencePtr>& requests) const
{
    return requests.size() == 1;
}

Status SpeculativeDecoder::initialize_runtime(int max_sequence_slots, const device::Context& context)
{
    return proposer_->initialize_speculative_runtime(max_sequence_slots, context);
}

Status SpeculativeDecoder::reset_runtime(const device::Context& context)
{
    target_contexts_.clear();
    return proposer_->reset_speculative_runtime(context);
}

Status SpeculativeDecoder::configure_target_forward(model::ForwardOptions& options,
                                                     Tensor& layered_hidden_states,
                                                     int batch_size, int sequence_length) const
{
    layered_hidden_states = FIREFLY_TRY(Tensor::create(
        {static_cast<int64_t>(target_hidden_layers_.size()), batch_size, sequence_length,
         target_config_.hidden_size}, target_config_.dtype, Device::CUDA, options.context));
    options.hidden_state_layers = target_hidden_layers_;
    options.layered_hidden_state_output = &layered_hidden_states;
    return {};
}

Status SpeculativeDecoder::configure_target_ragged_forward(model::ForwardOptions& options,
                                                            Tensor& layered_hidden_states,
                                                            int total_tokens) const
{
    layered_hidden_states = FIREFLY_TRY(Tensor::create(
        {static_cast<int64_t>(target_hidden_layers_.size()), total_tokens, target_config_.hidden_size},
        target_config_.dtype, Device::CUDA, options.context));
    options.hidden_state_layers = target_hidden_layers_;
    options.layered_hidden_state_output = &layered_hidden_states;
    return {};
}

Status SpeculativeDecoder::capture_target_context(const Tensor& layered_hidden_states,
                                                   const std::vector<scheduler::SequencePtr>& requests,
                                                   int sequence_length, const device::Context& context,
                                                   int token_index, int model_batch_size)
{
    if (layered_hidden_states.data() == nullptr) return {};
    const int batch_size = static_cast<int>(requests.size());
    if (model_batch_size <= 0) model_batch_size = batch_size;
    if (model_batch_size < batch_size)
        return unexpected(Error{ErrorCode::InvalidArgument,
                                "speculative context model batch is smaller than request batch"});
    const int hidden_size = target_config_.hidden_size;
    const size_t row_bytes = static_cast<size_t>(hidden_size) * dtype_size(layered_hidden_states.dtype());
    const auto* source = static_cast<const std::byte*>(layered_hidden_states.data());
    const int selected_token = token_index >= 0 ? token_index : sequence_length - 1;
    if (selected_token < 0 || selected_token >= sequence_length)
        return unexpected(Error{ErrorCode::InvalidArgument, "speculative context token index is invalid"});
    for (int batch = 0; batch < batch_size; ++batch)
    {
        Tensor compact = FIREFLY_TRY(Tensor::create(
            {1, static_cast<int64_t>(target_hidden_layers_.size()), hidden_size}, layered_hidden_states.dtype(),
            Device::CUDA, context));
        auto* destination = static_cast<std::byte*>(compact.data());
        for (size_t layer = 0; layer < target_hidden_layers_.size(); ++layer)
        {
            const size_t source_row =
                ((layer * model_batch_size + batch) * sequence_length + selected_token) * row_bytes;
            FIREFLY_TRY(device::check_cuda(
                cudaMemcpyAsync(destination + layer * row_bytes, source + source_row, row_bytes,
                                cudaMemcpyDeviceToDevice, context.stream()),
                "capture speculative dense target context"));
        }
        target_contexts_[requests[batch]->id] = std::move(compact);
    }
    return {};
}

Status SpeculativeDecoder::capture_target_context_ragged(
    const Tensor& layered_hidden_states, const std::vector<scheduler::SequencePtr>& requests,
    std::span<const int> sequence_offsets, std::span<const int> sequence_lengths,
    const device::Context& context)
{
    if (layered_hidden_states.data() == nullptr) return {};
    if (requests.size() != sequence_offsets.size() || requests.size() != sequence_lengths.size())
        return unexpected(Error{ErrorCode::InvalidArgument, "speculative ragged context metadata is invalid"});
    const int hidden_size = target_config_.hidden_size;
    const int total_tokens = layered_hidden_states.shape()[1];
    const size_t row_bytes = static_cast<size_t>(hidden_size) * dtype_size(layered_hidden_states.dtype());
    const auto* source = static_cast<const std::byte*>(layered_hidden_states.data());
    for (size_t batch = 0; batch < requests.size(); ++batch)
    {
        const int token_index = sequence_offsets[batch] + sequence_lengths[batch] - 1;
        if (token_index < 0 || token_index >= total_tokens)
            return unexpected(Error{ErrorCode::InvalidArgument,
                                    "speculative ragged context token index is invalid"});
        Tensor compact = FIREFLY_TRY(Tensor::create(
            {1, static_cast<int64_t>(target_hidden_layers_.size()), hidden_size}, layered_hidden_states.dtype(),
            Device::CUDA, context));
        auto* destination = static_cast<std::byte*>(compact.data());
        for (size_t layer = 0; layer < target_hidden_layers_.size(); ++layer)
        {
            const size_t source_row = (layer * total_tokens + token_index) * row_bytes;
            FIREFLY_TRY(device::check_cuda(
                cudaMemcpyAsync(destination + layer * row_bytes, source + source_row, row_bytes,
                                cudaMemcpyDeviceToDevice, context.stream()),
                "capture speculative ragged target context"));
        }
        target_contexts_[requests[batch]->id] = std::move(compact);
    }
    return {};
}

void SpeculativeDecoder::erase(const std::string& request_id) { target_contexts_.erase(request_id); }

Status SpeculativeDecoder::decode(Engine& engine, const std::vector<scheduler::SequencePtr>& requests,
                                  const device::Context& context)
{
    if (!can_decode(requests))
        return unexpected(Error{ErrorCode::InvalidArgument, "speculative decode batch is unsupported"});

    auto request = requests.front();
    const int remaining = request->max_tokens - static_cast<int>(request->generated_tokens.size());
    const int draft_length = std::min(max_draft_tokens_, std::max(0, remaining - 1));
    if (draft_length <= 0)
    {
        return engine.process_dynamic_decode(requests, request->context_len, false, context);
    }

    const int anchor = request->generated_tokens.back();
    std::vector<int> proposals(draft_length);
    auto context_found = target_contexts_.find(request->id);
    const Tensor* target_context = context_found == target_contexts_.end() ? nullptr : &context_found->second;
    model::SpeculativeProposal proposal = FIREFLY_TRY_CONTEXT(
        proposer_->propose({.anchor_token = anchor, .token_count = draft_length,
                            .target_context = target_context, .context = context}),
        "generate speculative proposal");
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(proposals.data(), proposal.token_ids.data(), proposals.size() * sizeof(int),
                        cudaMemcpyDeviceToHost, context.stream()),
        "copy speculative proposal tokens to host"));
    FIREFLY_TRY(device::check_cuda(cudaStreamSynchronize(context.stream()),
                                   "synchronize speculative proposal tokens"));
    if (confidence_threshold_ > 0.0f && proposal.confidence.data() != nullptr)
    {
        const std::vector<float> confidence = FIREFLY_TRY(copy_confidence(proposal.confidence, draft_length,
                                                                          context));
        size_t retained = 0;
        while (retained < proposals.size())
        {
            if (confidence[retained] < confidence_threshold_) break;
            ++retained;
        }
        proposals.resize(retained);
    }
    if (proposals.empty())
    {
        return engine.process_dynamic_decode(requests, request->context_len, false, context);
    }

    const int verify_length = static_cast<int>(proposals.size()) + 1;
    FIREFLY_LOG_DEBUG("speculative", "target verify begin proposals={} context={}", proposals.size(),
                      request->context_len);
    std::vector<int> verify_input_host;
    verify_input_host.reserve(verify_length);
    verify_input_host.push_back(anchor);
    verify_input_host.insert(verify_input_host.end(), proposals.begin(), proposals.end());
    Tensor verify_input = FIREFLY_TRY(Tensor::create({1, verify_length}, DType::I32, Device::CUDA, context));
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(verify_input.data(), verify_input_host.data(), verify_input_host.size() * sizeof(int),
                        cudaMemcpyHostToDevice, context.stream()),
        "copy speculative verification tokens"));
    Tensor context_lens = FIREFLY_TRY(Tensor::create({1}, DType::I32, Device::CUDA, context));
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(context_lens.data(), &request->context_len, sizeof(int), cudaMemcpyHostToDevice,
                        context.stream()),
        "copy speculative verification context length"));
    Tensor block_table = FIREFLY_TRY(Tensor::create(
        {static_cast<int64_t>(request->block_table.size())}, DType::I32, Device::CUDA, context));
    FIREFLY_TRY(device::check_cuda(
        cudaMemcpyAsync(block_table.data(), request->block_table.data(), request->block_table.size() * sizeof(int),
                        cudaMemcpyHostToDevice, context.stream()),
        "copy speculative verification block table"));
    Tensor state_slot;
    const int* state_slot_ptr = nullptr;
    if (target_runtime_.sequence_state)
    {
        state_slot = FIREFLY_TRY(Tensor::create({1}, DType::I32, Device::CUDA, context));
        FIREFLY_TRY(device::check_cuda(
            cudaMemcpyAsync(state_slot.data(), &request->state_slot, sizeof(int), cudaMemcpyHostToDevice,
                            context.stream()),
            "copy speculative verification state slot"));
        state_slot_ptr = static_cast<int*>(state_slot.data());
    }

    std::unique_ptr<model::SpeculativeRuntimeState> runtime_snapshot;
    if (transactional_runtime_ != nullptr)
        runtime_snapshot = FIREFLY_TRY_CONTEXT(
            transactional_runtime_->snapshot_speculative_runtime(request->state_slot, context),
            "snapshot speculative target runtime");
    model::ModelInput verify_model_input{verify_input,
                                         context_lens,
                                         {engine.k_caches_, engine.v_caches_, engine.kv_scale_caches_,
                                          static_cast<int*>(block_table.data()),
                                          static_cast<int>(request->block_table.size())},
                                         state_slot_ptr};
    FIREFLY_TRY(kernels::prepare_attention_prefill(
        static_cast<const int*>(block_table.data()), 1, verify_length, request->context_len,
        static_cast<int>(request->block_table.size()), target_config_.num_attention_heads,
        target_runtime_.kv_cache_head_count, target_runtime_.kv_cache_head_dim, context));
    Tensor target_hidden_layers;
    model::ForwardOptions verify_options{.context = context,
                                         .return_all_logits = true,
                                         .hidden_state_layers = target_hidden_layers_,
                                         .layered_hidden_state_output = &target_hidden_layers,
                                         .max_decode_context_len = request->context_len + verify_length,
                                         .min_context_len = request->context_len};
    FIREFLY_TRY(configure_target_forward(verify_options, target_hidden_layers, 1, verify_length));
    Tensor target_logits = FIREFLY_TRY_CONTEXT(target_->forward(verify_model_input, verify_options),
                                               "execute speculative target verification");
    const std::vector<int> target_tokens = FIREFLY_TRY(
        greedy_rows(target_logits, verify_length, target_config_.vocab_size, context));

    int verified_prefix = 0;
    while (verified_prefix < static_cast<int>(proposals.size()) &&
           proposals[verified_prefix] == target_tokens[verified_prefix])
        ++verified_prefix;
    if (runtime_snapshot == nullptr && target_runtime_.sequence_state)
        return unexpected(Error{ErrorCode::InvalidState,
                                "stateful speculative target returned an empty runtime snapshot"});
    if (runtime_snapshot == nullptr)
    {
        FIREFLY_TRY(capture_target_context(target_hidden_layers, requests, verify_length, context,
                                           verified_prefix));
        request->generated_tokens.insert(request->generated_tokens.end(), proposals.begin(),
                                         proposals.begin() + verified_prefix);
        request->generated_tokens.push_back(target_tokens[verified_prefix]);
        request->context_len += 1 + verified_prefix;
        return {};
    }
    FIREFLY_TRY_CONTEXT(transactional_runtime_->restore_speculative_runtime(std::move(runtime_snapshot), context),
                        "restore speculative target runtime");

    int accepted = 0;
    int bonus = 0;
    Tensor committed_hidden_layers;
    for (int replay_index = 0; replay_index <= verified_prefix; ++replay_index)
    {
        const int replay_token = replay_index == 0 ? anchor : proposals[replay_index - 1];
        const int replay_context = request->context_len + replay_index;
        Tensor replay_input = FIREFLY_TRY(Tensor::create({1, 1}, DType::I32, Device::CUDA, context));
        FIREFLY_TRY(device::check_cuda(
            cudaMemcpyAsync(replay_input.data(), &replay_token, sizeof(int), cudaMemcpyHostToDevice,
                            context.stream()),
            "copy speculative replay token"));
        FIREFLY_TRY(device::check_cuda(
            cudaMemcpyAsync(context_lens.data(), &replay_context, sizeof(int), cudaMemcpyHostToDevice,
                            context.stream()),
            "copy speculative replay context length"));
        model::ModelInput replay_model_input{replay_input,
                                              context_lens,
                                              {engine.k_caches_, engine.v_caches_, engine.kv_scale_caches_,
                                               static_cast<int*>(block_table.data()),
                                               static_cast<int>(request->block_table.size())},
                                              state_slot_ptr};
        FIREFLY_TRY(kernels::prepare_attention_decode(
            &replay_context, request->block_table.data(), 1, static_cast<int>(request->block_table.size()),
            target_config_.num_attention_heads, target_runtime_.kv_cache_head_count,
            target_runtime_.kv_cache_head_dim, context));
        Tensor replay_hidden_layers;
        model::ForwardOptions replay_options{.context = context, .max_decode_context_len = replay_context};
        FIREFLY_TRY(configure_target_forward(replay_options, replay_hidden_layers, 1, 1));
        Tensor replay_logits = FIREFLY_TRY_CONTEXT(target_->forward(replay_model_input, replay_options),
                                                   "execute speculative target replay");
        const int replay_next = FIREFLY_TRY(greedy_rows(replay_logits, 1, target_config_.vocab_size, context)).front();
        committed_hidden_layers = std::move(replay_hidden_layers);
        if (replay_index < verified_prefix && replay_next == proposals[replay_index])
        {
            ++accepted;
            continue;
        }
        bonus = replay_next;
        break;
    }
    FIREFLY_TRY(capture_target_context(committed_hidden_layers, requests, 1, context));
    request->generated_tokens.insert(request->generated_tokens.end(), proposals.begin(), proposals.begin() + accepted);
    request->generated_tokens.push_back(bonus);
    request->context_len += 1 + accepted;
    FIREFLY_LOG_DEBUG("speculative", "request_id={} proposed={} accepted={} bonus={} context={}", request->id,
                      proposals.size(), accepted, bonus, request->context_len);
    return {};
}

}  // namespace firefly::execution
