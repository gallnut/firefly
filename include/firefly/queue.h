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
};

class ResultQueue
{
public:
    ResultQueue() = default;
    ~ResultQueue() = default;

    void push(const std::string& req_id, const std::string& text, bool is_finished)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push({req_id, text, is_finished});
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
