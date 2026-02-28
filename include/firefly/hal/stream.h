#pragma once

#include <cuda_runtime.h>

#include "firefly/hal/error.h"

namespace firefly::hal
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

        return Stream(handle);
    }

    // Default constructor creates a null/invalid stream wrapper (optional)
    // or we can delete default constructor to force usage of create()
    Stream() = default;

    ~Stream()
    {
        if (stream_)
        {
            // Destructors should not throw or return errors, so we abort on fatal error
            // or just log. Since destroying a stream usually only fails if context is dead,
            // generic cleanup is safe.
            cudaStreamDestroy(stream_);
        }
    }

    // Move-only semantics
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }

    Stream& operator=(Stream&& other) noexcept
    {
        if (this != &other)
        {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const { return stream_; }
                               operator cudaStream_t() const { return stream_; }

    [[nodiscard]] Status synchronize() const { return check_cuda(cudaStreamSynchronize(stream_)); }

    [[nodiscard]] Status wait(cudaEvent_t event) const { return check_cuda(cudaStreamWaitEvent(stream_, event, 0)); }

private:
    // Private constructor used by create()
    explicit Stream(cudaStream_t s) : stream_(s) {}

    cudaStream_t stream_{nullptr};
};

}  // namespace firefly::hal
