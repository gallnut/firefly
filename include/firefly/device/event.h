#pragma once

#include <cuda_runtime.h>

#include "firefly/device/error.h"

namespace firefly::device
{

/**
 * @brief RAII wrapper for cudaEvent_t using std::expected for errors.
 */
class Event
{
public:
    /** @brief CUDA event creation modes exposed as a strongly typed flag value. */
    enum class Mode
    {
        Default = cudaEventDefault,              ///< Default timing-capable event behavior.
        BlockingSync = cudaEventBlockingSync,    ///< Host synchronization blocks instead of busy-waiting.
        DisableTiming = cudaEventDisableTiming,  ///< Omits timestamp support for lower overhead.
        Interprocess = cudaEventInterprocess     ///< Creates an event suitable for IPC handles.
    };

    /**
     * @brief Creates and owns a CUDA event.
     * @param mode CUDA creation mode.
     * @return An event on success or a CUDA error on failure.
     */
    static Result<Event> create(Mode mode = Mode::DisableTiming)
    {
        cudaEvent_t handle = nullptr;
        auto        err = cudaEventCreateWithFlags(&handle, static_cast<unsigned int>(mode));
        if (err != cudaSuccess) return unexpected(cuda_error(err, "create CUDA event"));
        return Event(handle);
    }

    /** @brief Constructs an empty event wrapper. */
    Event() = default;

    /** @brief Destroys the owned CUDA event, if any. */
    ~Event()
    {
        if (event_)
        {
            cudaEventDestroy(event_);
        }
    }

    /** @brief CUDA event ownership cannot be copied. */
    Event(const Event&) = delete;
    /** @brief CUDA event ownership cannot be copy-assigned. */
    Event& operator=(const Event&) = delete;

    /** @brief Transfers ownership from another event wrapper. */
    Event(Event&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }

    /** @brief Releases the current event and transfers ownership from another wrapper. */
    Event& operator=(Event&& other) noexcept
    {
        if (this != &other)
        {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    /** @brief Returns the borrowed native CUDA event handle. */
    [[nodiscard]] cudaEvent_t get() const { return event_; }
    /** @brief Converts to the borrowed native CUDA event handle. */
    operator cudaEvent_t() const { return event_; }

    /** @brief Records the event after prior work in `stream`. */
    [[nodiscard]] Status record(cudaStream_t stream = 0) const { return check_cuda(cudaEventRecord(event_, stream)); }

    /** @brief Blocks the host until the recorded event completes. */
    [[nodiscard]] Status synchronize() const { return check_cuda(cudaEventSynchronize(event_)); }

    /** @brief Returns true when the event has completed; CUDA query failures also return false. */
    [[nodiscard]] bool query() const
    {
        cudaError_t err = cudaEventQuery(event_);
        // NotReady is usually not an "Error" in the fatal sense, just status
        if (err == cudaSuccess) return true;
        return false;
    }

    /**
     * @brief Measures elapsed device time between two timing-capable events.
     * @param start Earlier recorded event.
     * @param end Later recorded event.
     * @return Elapsed milliseconds or a CUDA error.
     */
    static Result<float> elapsed_time(const Event& start, const Event& end)
    {
        float ms = 0.0f;
        auto  err = cudaEventElapsedTime(&ms, start.event_, end.event_);
        if (err != cudaSuccess) return unexpected(cuda_error(err, "measure CUDA event elapsed time"));
        return ms;
    }

private:
    /**
     * @brief Takes ownership of an already-created CUDA event.
     * @param e Native event handle returned by the CUDA Runtime API.
     */
    explicit Event(cudaEvent_t e) : event_(e) {}
    cudaEvent_t event_{nullptr}; ///< Owned native event handle, or null for an empty wrapper.
};

}  // namespace firefly::device
