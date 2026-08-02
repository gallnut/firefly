#pragma once

#include <cuda_runtime_api.h>

#include <memory>

namespace firefly::device
{

namespace detail
{
struct StreamState
{
    explicit StreamState(cudaStream_t handle) : handle(handle) {}
    ~StreamState()
    {
        if (handle) cudaStreamDestroy(handle);
    }

    cudaStream_t handle = nullptr;
};
}  // namespace detail

class Context
{
public:
    Context() = default;
    [[nodiscard]] cudaStream_t stream() const { return stream_ ? stream_->handle : nullptr; }

private:
    explicit Context(std::shared_ptr<detail::StreamState> stream) : stream_(std::move(stream)) {}

    std::shared_ptr<detail::StreamState> stream_;

    friend class Stream;
};

}  // namespace firefly::device
