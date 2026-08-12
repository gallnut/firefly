#pragma once

#include <cuda_runtime_api.h>

#include <memory>

namespace firefly::device
{

namespace detail
{
/** @brief Shared ownership state that destroys a CUDA stream when its last owner disappears. */
struct StreamState
{
    /** @brief Takes ownership of an existing CUDA stream handle. */
    explicit StreamState(cudaStream_t handle) : handle(handle) {}
    /** @brief Destroys the owned CUDA stream when it is non-null. */
    ~StreamState()
    {
        if (handle) cudaStreamDestroy(handle);
    }

    cudaStream_t handle = nullptr;  ///< Owned CUDA stream handle.
};
}  // namespace detail

/**
 * @brief Lightweight copyable execution context sharing a CUDA stream lifetime.
 *
 * A default context represents CUDA's legacy default stream. Context copies share
 * stream ownership with the originating `Stream` and remain valid after it is moved.
 */
class Context
{
public:
    /** @brief Constructs a context for CUDA's default stream. */
    Context() = default;
    /** @brief Returns the associated CUDA stream, or `nullptr` for the default stream. */
    [[nodiscard]] cudaStream_t stream() const { return stream_ ? stream_->handle : nullptr; }

private:
    /**
     * @brief Constructs a context sharing ownership of a stream state.
     * @param stream Shared stream state, or null to represent CUDA's default stream.
     */
    explicit Context(std::shared_ptr<detail::StreamState> stream) : stream_(std::move(stream)) {}

    std::shared_ptr<detail::StreamState> stream_; ///< Optional shared owner of the native CUDA stream.

    /** @brief Allows `Stream::context` to invoke the private sharing constructor. */
    friend class Stream;
};

}  // namespace firefly::device
