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
    /** @brief Relative scheduling priorities supported by the stream factory. */
    enum class Priority
    {
        Default, ///< CUDA's default stream priority.
        High,    ///< Greatest priority supported by the active device.
        Low      ///< Least priority supported by the active device.
    };

    /**
     * @brief Creates an owned CUDA stream.
     * @param priority Requested device scheduling priority.
     * @param non_blocking Whether the stream may execute independently of the legacy default stream.
     * @return A stream owner or a CUDA error.
     */
    static Result<Stream> create(Priority priority = Priority::Default, bool non_blocking = true)
    {
        cudaStream_t handle = nullptr;
        unsigned int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;

        if (priority == Priority::Default)
        {
            auto err = cudaStreamCreateWithFlags(&handle, flags);
            if (err != cudaSuccess) return unexpected(cuda_error(err, "create CUDA stream"));
        }
        else
        {
            int  least, greatest;
            auto err = cudaDeviceGetStreamPriorityRange(&least, &greatest);
            if (err != cudaSuccess) return unexpected(cuda_error(err, "query CUDA stream priority range"));

            int p = (priority == Priority::High) ? greatest : least;
            err = cudaStreamCreateWithPriority(&handle, flags, p);
            if (err != cudaSuccess) return unexpected(cuda_error(err, "create prioritized CUDA stream"));
        }

        return Stream(std::make_shared<detail::StreamState>(handle));
    }

    /** @brief Constructs a wrapper representing CUDA's default stream. */
    Stream() = default;

    /** @brief Releases shared ownership; the stream is destroyed with the last owner. */
    ~Stream() = default;

    /** @brief Stream ownership wrappers cannot be copied. */
    Stream(const Stream&) = delete;
    /** @brief Stream ownership wrappers cannot be copy-assigned. */
    Stream& operator=(const Stream&) = delete;

    /** @brief Transfers shared stream state from another wrapper. */
    Stream(Stream&&) noexcept = default;

    /** @brief Transfers shared stream state from another wrapper. */
    Stream& operator=(Stream&&) noexcept = default;

    /** @brief Returns the borrowed native CUDA stream handle. */
    [[nodiscard]] cudaStream_t get() const { return stream_ ? stream_->handle : nullptr; }
    /** @brief Converts to the borrowed native CUDA stream handle. */
    operator cudaStream_t() const { return get(); }
    /** @brief Creates a copyable execution context sharing this stream's lifetime. */
    [[nodiscard]] Context context() const { return Context(stream_); }

    /** @brief Blocks the host until all previously enqueued stream work completes. */
    [[nodiscard]] Status synchronize() const { return check_cuda(cudaStreamSynchronize(get())); }

    /** @brief Makes this stream wait for a CUDA event before executing later work. */
    [[nodiscard]] Status wait(cudaEvent_t event) const { return check_cuda(cudaStreamWaitEvent(get(), event, 0)); }

private:
    /**
     * @brief Constructs a stream wrapper from shared ownership state.
     * @param stream Shared owner of a successfully created native CUDA stream.
     */
    explicit Stream(std::shared_ptr<detail::StreamState> stream) : stream_(std::move(stream)) {}

    std::shared_ptr<detail::StreamState> stream_; ///< Optional shared owner of the native CUDA stream.
};

}  // namespace firefly::device
