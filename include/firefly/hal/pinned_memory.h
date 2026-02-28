#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "firefly/hal/error.h"

namespace firefly::hal
{

/**
 * @brief RAII wrapper for Pinned Host Memory (Page-locked memory).
 * Pinned memory allows for faster host <-> device transfers and asynchronous operations.
 *
 * Updated to use Result<T> for error handling instead of exceptions.
 */
template <typename T = std::byte>
class PinnedMemory
{
public:
    // Default constructor creates an empty/invalid instance
    PinnedMemory() = default;

    /**
     * @brief Factory method to create allocated pinned memory.
     * @param count Number of elements of type T to allocate.
     * @return Result<PinnedMemory<T>> containing the instance or an error.
     */
    static Result<PinnedMemory<T>> create(size_t count)
    {
        PinnedMemory<T> memory;
        memory.count_ = count;
        memory.size_bytes_ = count * sizeof(T);

        // cudaMallocHost is standard for pinned memory
        cudaError_t err = cudaMallocHost(&memory.ptr_, memory.size_bytes_);
        if (err != cudaSuccess)
        {
            return unexpected(Error{err, "Failed to allocate pinned memory: " + std::string(cudaGetErrorString(err))});
        }

        return memory;
    }

    ~PinnedMemory() { free(); }

    // Move-only semantics
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    PinnedMemory(PinnedMemory&& other) noexcept : ptr_(other.ptr_), count_(other.count_), size_bytes_(other.size_bytes_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.size_bytes_ = 0;
    }

    PinnedMemory& operator=(PinnedMemory&& other) noexcept
    {
        if (this != &other)
        {
            free();
            ptr_ = other.ptr_;
            count_ = other.count_;
            size_bytes_ = other.size_bytes_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.size_bytes_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T*       data() { return ptr_; }
    [[nodiscard]] const T* data() const { return ptr_; }

    [[nodiscard]] size_t size() const { return count_; }
    [[nodiscard]] size_t size_bytes() const { return size_bytes_; }
    [[nodiscard]] bool   valid() const { return ptr_ != nullptr; }

    // Support typical container operations for range-based loops
    T*       begin() { return ptr_; }
    const T* begin() const { return ptr_; }
    T*       end() { return ptr_ + count_; }
    const T* end() const { return ptr_ + count_; }

    T&       operator[](size_t index) { return ptr_[index]; }
    const T& operator[](size_t index) const { return ptr_[index]; }

private:
    T*     ptr_{nullptr};
    size_t count_{0};
    size_t size_bytes_{0};

    // Private constructor used by create() is not needed since we use default and friend or just manual setup
    // But since we use default, we just populate members in create()

    void free()
    {
        if (ptr_)
        {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }
};

/**
 * @brief RAII wrapper for Registering HOST Memory as Pinned Memory.
 * Instead of allocating, this takes an existing pointer (e.g. from mmap) and pins it.
 */
class PinnedRegistration
{
public:
    PinnedRegistration() = default;

    /**
     * @brief Attempt to pin memory at ptr for size bytes.
     */
    static Result<PinnedRegistration> register_memory(void* ptr, size_t size,
                                                      unsigned int flags = cudaHostRegisterDefault)
    {
        PinnedRegistration reg;
        cudaError_t        err = cudaHostRegister(ptr, size, flags);
        if (err != cudaSuccess)
        {
            return unexpected(Error{err, "Failed to register host memory: " + std::string(cudaGetErrorString(err))});
        }
        reg.ptr_ = ptr;
        return reg;
    }

    ~PinnedRegistration() { unregister(); }

    PinnedRegistration(const PinnedRegistration&) = delete;
    PinnedRegistration& operator=(const PinnedRegistration&) = delete;

    PinnedRegistration(PinnedRegistration&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    PinnedRegistration& operator=(PinnedRegistration&& other) noexcept
    {
        if (this != &other)
        {
            unregister();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }

private:
    void* ptr_{nullptr};

    void unregister()
    {
        if (ptr_)
        {
            cudaHostUnregister(ptr_);
            ptr_ = nullptr;
        }
    }
};

}  // namespace firefly::hal
