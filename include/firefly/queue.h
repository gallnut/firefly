#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>

namespace firefly
{

struct ResultItem
{
    std::string req_id;
    std::string text;
    bool        is_finished;
    int         prompt_tokens = 0;
    int         completion_tokens = 0;
    int         total_tokens = 0;
    bool        has_usage = false;
};

class ResultQueue
{
public:
    ResultQueue() = default;
    ~ResultQueue() = default;

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
     * @return ResultItem The item popped from the queue.
     */
    ResultItem pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty(); });
        ResultItem item = queue_.front();
        queue_.pop();
        return item;
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<ResultItem>  queue_;
    std::mutex              mutex_;
    std::condition_variable cv_;
};

}  // namespace firefly
