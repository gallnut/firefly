#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace firefly::service
{

/** @brief Thread-safe response queue and cancellation state for one active gRPC call. */
struct Session
{
    std::mutex              mtx;  ///< Protects response and usage fields.
    std::condition_variable cv;   ///< Wakes response writers when output or terminal state changes.

    std::queue<std::pair<std::string, bool>> output_queue;  ///< UTF-8 fragments paired with terminal flags.

    bool is_finished = false;    ///< Whether no further output will be produced.
    int  prompt_tokens = 0;      ///< Prompt-token usage for the terminal response.
    int  completion_tokens = 0;  ///< Completion-token usage for the terminal response.
    int  total_tokens = 0;       ///< Sum of prompt and completion tokens.
    bool has_usage = false;      ///< Whether usage fields are valid.

    std::shared_ptr<std::atomic_bool> cancel_flag =
        std::make_shared<std::atomic_bool>(false);  ///< Engine cancellation signal.

    /**
     * @brief Enqueues one response fragment, records optional usage, and wakes a transport thread.
     * @param text Newly decoded UTF-8 fragment.
     * @param finished Whether this fragment terminates the response stream.
     * @param prompt_count Prompt-token usage, or `-1` when unavailable.
     * @param completion_count Completion-token usage, or `-1` when unavailable.
     * @threadsafe Serialized by `mtx` and safe against transport-side consumers.
     */
    void push(const std::string& text, bool finished, int prompt_count = -1, int completion_count = -1)
    {
        std::lock_guard<std::mutex> lock(mtx);
        output_queue.push({text, finished});
        if (prompt_count >= 0 && completion_count >= 0)
        {
            prompt_tokens = prompt_count;
            completion_tokens = completion_count;
            total_tokens = prompt_count + completion_count;
            has_usage = true;
        }
        if (finished) is_finished = true;
        cv.notify_one();
    }

    /** @brief Atomically requests engine cancellation and marks the transport session finished. */
    void cancel()
    {
        cancel_flag->store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mtx);
        is_finished = true;
        cv.notify_one();
    }
};

}  // namespace firefly::service
