#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>

namespace firefly::execution
{

/** @brief One streaming generation update transferred from the engine to the service layer. */
struct ResultItem
{
    std::string req_id; ///< Request identifier supplied to `Engine::async_generate`.
    std::string text; ///< Newly decoded UTF-8 text for this update.
    bool        is_finished; ///< Whether this is the terminal update for the request.
    int         prompt_tokens = 0; ///< Prompt-token count reported on terminal updates.
    int         completion_tokens = 0; ///< Generated-token count reported on terminal updates.
    int         total_tokens = 0; ///< Sum of prompt and completion tokens.
    bool        has_usage = false; ///< Whether token usage fields contain valid values.
};

/** @brief Thread-safe blocking FIFO connecting asynchronous execution and response streaming. */
class ResultQueue
{
public:
    /** @brief Constructs an empty queue. */
    ResultQueue() = default;
    /** @brief Destroys the queue; callers must ensure no thread is blocked in `pop`. */
    ~ResultQueue() = default;

    /**
     * @brief Enqueues one response update and wakes a waiting consumer.
     * @param req_id Request identifier.
     * @param text Newly decoded response fragment.
     * @param is_finished Whether this update terminates the request.
     * @param prompt_tokens Prompt-token usage, or `-1` when unavailable.
     * @param completion_tokens Completion-token usage, or `-1` when unavailable.
     * @threadsafe May be called concurrently with `push`, `pop`, and `empty`.
     */
    void push(const std::string& req_id, const std::string& text, bool is_finished, int prompt_tokens = -1,
              int completion_tokens = -1)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ResultItem                  item;
        item.req_id = req_id;
        item.text = text;
        item.is_finished = is_finished;
        if (prompt_tokens >= 0 && completion_tokens >= 0)
        {
            item.prompt_tokens = prompt_tokens;
            item.completion_tokens = completion_tokens;
            item.total_tokens = prompt_tokens + completion_tokens;
            item.has_usage = true;
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    /**
     * @brief Blocking pop operation.
     *
     * Waits until the queue is not empty, then removes and returns the front item.
     *
     * @return Oldest queued result item, removed from the FIFO.
     * @threadsafe Supports one or more synchronized consumers and concurrent producers.
     */
    ResultItem pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty(); });
        ResultItem item = queue_.front();
        queue_.pop();
        return item;
    }

    /**
     * @brief Returns a synchronized snapshot indicating whether the queue is empty.
     * @return `true` when no result item was queued at the instant the mutex was held.
     * @threadsafe May be called concurrently with producers and consumers.
     */
    bool empty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<ResultItem>  queue_; ///< FIFO of pending streaming updates guarded by `mutex_`.
    std::mutex              mutex_; ///< Serializes every queue operation.
    std::condition_variable cv_; ///< Wakes blocking consumers after a producer enqueues data.
};

}  // namespace firefly::execution
