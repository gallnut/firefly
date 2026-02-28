#pragma once

#include <condition_variable>
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

    void push(const std::string& text, bool finished)
    {
        std::lock_guard<std::mutex> lock(mtx);
        output_queue.push({text, finished});
        if (finished) is_finished = true;
        cv.notify_one();
    }
};

}  // namespace firefly
