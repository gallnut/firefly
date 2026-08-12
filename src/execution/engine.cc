#include "firefly/execution/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

#include "firefly/core/logging.h"
#include "firefly/device/stream.h"
#include "firefly/execution/trace.h"

namespace firefly::execution
{
Engine::Engine(model::Model* model, const model::ModelConfig& config, const model::Tokenizer& tokenizer,
               ResultQueue* result_queue, EngineOptions options)
    : model_(model),
      config_(config),
      tokenizer_(tokenizer),
      result_queue_(result_queue),
      options_(options),
      runtime_requirements_(model->runtime_requirements()),
      scheduler_()
{
    if (runtime_requirements_.prefill_chunk_limit > 0)
        options_.max_prefill_chunk_size =
            std::min(options_.max_prefill_chunk_size, runtime_requirements_.prefill_chunk_limit);
}

Result<std::unique_ptr<Engine>> Engine::create(model::Model* model, const model::ModelConfig& config,
                                               const model::Tokenizer& tokenizer, ResultQueue* result_queue,
                                               EngineOptions options)
{
    if (model == nullptr)
        return unexpected(Error{ErrorCode::InvalidArgument, "engine requires a loaded target model"});
    auto engine = std::unique_ptr<Engine>(new Engine(model, config, tokenizer, result_queue, options));
    if (options.speculative.proposer != nullptr)
        engine->speculative_decoder_ = FIREFLY_TRY(SpeculativeDecoder::create(
            model, config, engine->runtime_requirements_, options.speculative));
    return engine;
}

Engine::~Engine() { stop(); }

void Engine::async_generate(const std::string& id, const std::vector<int>& input_ids, int max_tokens,
                            std::shared_ptr<std::atomic_bool> cancel_flag, bool ignore_eos)
{
    auto sequence =
        std::make_shared<scheduler::Sequence>(id, input_ids, max_tokens, std::move(cancel_flag), ignore_eos);
    scheduler_.add_sequence(sequence);
}

void Engine::loop() noexcept
{
    if (const cudaError_t error = cudaSetDevice(0); error != cudaSuccess)
    {
        Error failure = device::cuda_error(error, "select engine CUDA device");
        FIREFLY_LOG_ERROR("engine", "{}", failure.describe());
        running_ = false;
        return;
    }
    auto stream_result = device::Stream::create();
    if (!stream_result)
    {
        FIREFLY_LOG_ERROR("engine", "{}", stream_result.error().describe());
        running_ = false;
        return;
    }
    device::Stream  stream_owner = std::move(stream_result.value());
    device::Context context = stream_owner.context();
    uint64_t         step_index = 0;

    while (running_)
    {
        if (!scheduler_.has_unfinished_sequences())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto schedule_start = std::chrono::high_resolution_clock::now();
#endif
        FIREFLY_NVTX_PUSH("Engine_Schedule");
        if (scheduler_.has_pending_sequences() && !scheduler_.has_active_decode_sequences())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        auto batch = scheduler_.step();
        FIREFLY_NVTX_POP();
        ++step_index;
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto schedule_end = std::chrono::high_resolution_clock::now();
#endif
        for (const auto& request : batch.failed_sequences)
        {
            if (speculative_decoder_ != nullptr) speculative_decoder_->erase(request->id);
            if (result_queue_)
            {
                result_queue_->push(request->id,
                                    request->error_message.empty() ? "request failed" : request->error_message, true,
                                    static_cast<int>(request->prompt_tokens.size()),
                                    static_cast<int>(request->generated_tokens.size()));
            }
        }
        if (batch.sequences.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        std::vector<scheduler::SequencePtr> prefill_requests;
        std::vector<scheduler::SequencePtr> decode_requests;
        for (const auto& request : batch.sequences)
        {
            if (request->is_cancelled())
            {
                if (speculative_decoder_ != nullptr) speculative_decoder_->erase(request->id);
                scheduler_.abort_sequence(request);
                if (result_queue_)
                {
                    result_queue_->push(request->id, "", true, static_cast<int>(request->prompt_tokens.size()),
                                        static_cast<int>(request->generated_tokens.size()));
                }
            }
            else if (request->generated_tokens.empty())
            {
                prefill_requests.push_back(request);
            }
            else
            {
                decode_requests.push_back(request);
            }
        }

        FIREFLY_LOG_DEBUG("engine", "step={} batch={} prefill={} decode={} failed={}", step_index,
                          batch.sequences.size(), prefill_requests.size(), decode_requests.size(),
                          batch.failed_sequences.size());

        const bool mixed_batch = !runtime_requirements_.sequence_state &&
                                 std::getenv("FIREFLY_MIXED_BATCH") != nullptr;
        bool       decode_handled = false;
        Status step_status;
        if (mixed_batch && !prefill_requests.empty())
        {
            FIREFLY_NVTX_PUSH("Engine_Mixed");
            step_status = process_mixed_batch(batch.sequences, context);
            FIREFLY_NVTX_POP();
            decode_handled = decode_requests.empty() == false;
        }
        else if (!prefill_requests.empty())
        {
            FIREFLY_NVTX_PUSH("Engine_Prefill");
            step_status = process_prefill(prefill_requests, context);
            FIREFLY_NVTX_POP();
        }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto prefill_end = std::chrono::high_resolution_clock::now();
#endif

        if (!decode_handled && !decode_requests.empty())
        {
            FIREFLY_NVTX_PUSH("Engine_Decode");
            step_status = process_decode(decode_requests, context);
            FIREFLY_NVTX_POP();
        }
        if (!step_status)
        {
            FIREFLY_LOG_ERROR("engine", "request batch failed error={}", step_status.error().describe());
            fail_requests(batch.sequences, step_status.error());
            continue;
        }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto decode_end = std::chrono::high_resolution_clock::now();
#endif

        FIREFLY_NVTX_PUSH("Engine_Process_Results");
        process_outputs(batch.sequences);
        FIREFLY_NVTX_POP();
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
        auto          process_end = std::chrono::high_resolution_clock::now();
        static int    steps = 0;
        static double schedule_ms = 0;
        static double prefill_ms = 0;
        static double decode_ms = 0;
        static double process_ms = 0;
        schedule_ms += std::chrono::duration<double, std::milli>(schedule_end - schedule_start).count();
        prefill_ms += std::chrono::duration<double, std::milli>(prefill_end - schedule_end).count();
        decode_ms += std::chrono::duration<double, std::milli>(decode_end - prefill_end).count();
        process_ms += std::chrono::duration<double, std::milli>(process_end - decode_end).count();
        if (++steps % 50 == 0)
        {
            FIREFLY_LOG_INFO("performance",
                             "engine profile steps=50 schedule_ms={:.3f} prefill_ms={:.3f} decode_ms={:.3f} "
                             "process_ms={:.3f}",
                             schedule_ms, prefill_ms, decode_ms, process_ms);
            schedule_ms = prefill_ms = decode_ms = process_ms = 0;
        }
#endif
    }

}

void Engine::fail_requests(const std::vector<scheduler::SequencePtr>& requests, const Error& error)
{
    for (const auto& request : requests)
    {
        if (speculative_decoder_ != nullptr) speculative_decoder_->erase(request->id);
        request->error_message = error.describe();
        scheduler_.abort_sequence(request);
        if (result_queue_)
            result_queue_->push(request->id, request->error_message, true,
                                static_cast<int>(request->prompt_tokens.size()),
                                static_cast<int>(request->generated_tokens.size()));
    }
}
}  // namespace firefly::execution
