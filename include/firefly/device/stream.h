#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "firefly/device/context.h"
#include "firefly/device/error.h"

namespace firefly::device
{

/**
 * @brief RAII wrapper for cudaStream_t with Result-based error handling.
 */
class Stream
{
public:
    enum class Priority
    {
        Default,
        High,
        Low
    };

    // Factory method: The preferred way to create objects when construction can fail.
    static Result<Stream> create(Priority priority = Priority::Default, bool non_blocking = true)
    {
        cudaStream_t handle = nullptr;
        unsigned int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;

        if (priority == Priority::Default)
        {
            auto err = cudaStreamCreateWithFlags(&handle, flags);
            if (err != cudaSuccess) return unexpected(Error(err));
        }
        else
        {
            int  least, greatest;
            auto err = cudaDeviceGetStreamPriorityRange(&least, &greatest);
            if (err != cudaSuccess) return unexpected(Error(err));

            int p = (priority == Priority::High) ? greatest : least;
            err = cudaStreamCreateWithPriority(&handle, flags, p);
            if (err != cudaSuccess) return unexpected(Error(err));
        }

        return Stream(std::make_shared<detail::StreamState>(handle));
    }

    // Default constructor creates a null/invalid stream wrapper (optional)
    // or we can delete default constructor to force usage of create()
    Stream() = default;

    ~Stream() = default;

    // Move-only semantics
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&&) noexcept = default;

    Stream& operator=(Stream&&) noexcept = default;

    [[nodiscard]] cudaStream_t get() const { return stream_ ? stream_->handle : nullptr; }
                               operator cudaStream_t() const { return get(); }
    [[nodiscard]] Context context() const { return Context(stream_); }

    [[nodiscard]] Status synchronize() const { return check_cuda(cudaStreamSynchronize(get())); }

    [[nodiscard]] Status wait(cudaEvent_t event) const { return check_cuda(cudaStreamWaitEvent(get(), event, 0)); }

private:
    // Private constructor used by create()
    explicit Stream(std::shared_ptr<detail::StreamState> stream) : stream_(std::move(stream)) {}

    std::shared_ptr<detail::StreamState> stream_;
};

}  // namespace firefly::device
