#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace firefly
{

// We define a Session struct to hold the target gRPC connection queue
struct Session
{
    std::mutex                               mtx;
    std::condition_variable                  cv;
    std::queue<std::pair<std::string, bool>> output_queue;
    bool                                     is_finished = false;
    int                                      prompt_tokens = 0;
    int                                      completion_tokens = 0;
    int                                      total_tokens = 0;
    bool                                     has_usage = false;
    std::shared_ptr<std::atomic_bool>        cancel_flag = std::make_shared<std::atomic_bool>(false);

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

    void cancel()
    {
        cancel_flag->store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mtx);
        is_finished = true;
        cv.notify_one();
    }
};

}  // namespace firefly
