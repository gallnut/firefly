#pragma once

#include <cuda_runtime.h>

#include "firefly/hal/error.h"

namespace firefly::hal
{

/**
 * @brief RAII wrapper for cudaEvent_t using std::expected for errors.
 */
class Event
{
public:
    enum class Mode
    {
        Default = cudaEventDefault,
        BlockingSync = cudaEventBlockingSync,
        DisableTiming = cudaEventDisableTiming,
        Interprocess = cudaEventInterprocess
    };

    static Result<Event> create(Mode mode = Mode::DisableTiming)
    {
        cudaEvent_t handle = nullptr;
        auto        err = cudaEventCreateWithFlags(&handle, static_cast<unsigned int>(mode));
        if (err != cudaSuccess) return unexpected(Error(err));
        return Event(handle);
    }

    Event() = default;

    ~Event()
    {
        if (event_)
        {
            cudaEventDestroy(event_);
        }
    }

    // Move-only semantics
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    Event(Event&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }

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

    [[nodiscard]] cudaEvent_t get() const { return event_; }
                              operator cudaEvent_t() const { return event_; }

    [[nodiscard]] Status record(cudaStream_t stream = 0) const { return check_cuda(cudaEventRecord(event_, stream)); }

    [[nodiscard]] Status synchronize() const { return check_cuda(cudaEventSynchronize(event_)); }

    [[nodiscard]] bool query() const
    {
        cudaError_t err = cudaEventQuery(event_);
        // NotReady is usually not an "Error" in the fatal sense, just status
        if (err == cudaSuccess) return true;
        return false;
    }

    static Result<float> elapsed_time(const Event& start, const Event& end)
    {
        float ms = 0.0f;
        auto  err = cudaEventElapsedTime(&ms, start.event_, end.event_);
        if (err != cudaSuccess) return std::unexpected(Error(err));
        return ms;
    }

private:
    explicit Event(cudaEvent_t e) : event_(e) {}
    cudaEvent_t event_{nullptr};
};

}  // namespace firefly::hal
