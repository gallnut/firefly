#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "firefly/device/error.h"

namespace firefly::device
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
    /** @brief Constructs an empty pinned-memory owner. */
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
            return unexpected(cuda_error(err, "allocate page-locked host memory"));
        }

        return memory;
    }

    /** @brief Releases the owned page-locked allocation. */
    ~PinnedMemory() { free(); }

    /** @brief Pinned allocation ownership cannot be copied. */
    PinnedMemory(const PinnedMemory&) = delete;
    /** @brief Pinned allocation ownership cannot be copy-assigned. */
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    /** @brief Transfers ownership from another pinned allocation. */
    PinnedMemory(PinnedMemory&& other) noexcept : ptr_(other.ptr_), count_(other.count_), size_bytes_(other.size_bytes_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.size_bytes_ = 0;
    }

    /** @brief Releases current storage and transfers ownership from another allocation. */
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

    /** @brief Returns the mutable allocation pointer. */
    [[nodiscard]] T*       data() { return ptr_; }
    /** @brief Returns the immutable allocation pointer. */
    [[nodiscard]] const T* data() const { return ptr_; }

    /** @brief Returns the number of `T` elements in the allocation. */
    [[nodiscard]] size_t size() const { return count_; }
    /** @brief Returns the allocation size in bytes. */
    [[nodiscard]] size_t size_bytes() const { return size_bytes_; }
    /** @brief Returns whether this object owns a non-null allocation. */
    [[nodiscard]] bool   valid() const { return ptr_ != nullptr; }

    /** @brief Returns an iterator to the first element. */
    T*       begin() { return ptr_; }
    /** @brief Returns a const iterator to the first element. */
    const T* begin() const { return ptr_; }
    /** @brief Returns an iterator one past the last element. */
    T*       end() { return ptr_ + count_; }
    /** @brief Returns a const iterator one past the last element. */
    const T* end() const { return ptr_ + count_; }

    /** @brief Returns mutable unchecked access to an element. */
    T&       operator[](size_t index) { return ptr_[index]; }
    /** @brief Returns immutable unchecked access to an element. */
    const T& operator[](size_t index) const { return ptr_[index]; }

private:
    T*     ptr_{nullptr}; ///< Owned page-locked host allocation.
    size_t count_{0}; ///< Number of addressable `T` elements.
    size_t size_bytes_{0}; ///< Allocation size in bytes.

    // Private constructor used by create() is not needed since we use default and friend or just manual setup
    // But since we use default, we just populate members in create()

    /** @brief Releases owned pinned storage and restores the empty state. */
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
    /** @brief Constructs an empty host-memory registration owner. */
    PinnedRegistration() = default;

    /**
     * @brief Registers an existing host region for asynchronous CUDA transfers.
     * @param ptr Start address of the externally owned host region.
     * @param size Number of bytes to register.
     * @param flags CUDA host-registration flags.
     * @return Registration owner on success, or a CUDA error.
     * @warning The region must remain allocated and unregistered elsewhere for this object's lifetime.
     */
    static Result<PinnedRegistration> register_memory(void* ptr, size_t size,
                                                      unsigned int flags = cudaHostRegisterDefault)
    {
        PinnedRegistration reg;
        cudaError_t        err = cudaHostRegister(ptr, size, flags);
        if (err != cudaSuccess)
        {
            return unexpected(cuda_error(err, "register page-locked host memory"));
        }
        reg.ptr_ = ptr;
        return reg;
    }

    /** @brief Unregisters the owned host-memory region. */
    ~PinnedRegistration() { unregister(); }

    /** @brief Host registration ownership cannot be copied. */
    PinnedRegistration(const PinnedRegistration&) = delete;
    /** @brief Host registration ownership cannot be copy-assigned. */
    PinnedRegistration& operator=(const PinnedRegistration&) = delete;

    /** @brief Transfers registration ownership from another wrapper. */
    PinnedRegistration(PinnedRegistration&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    /** @brief Unregisters current storage and transfers ownership from another wrapper. */
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

    /** @brief Returns whether a host region is currently registered. */
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }

private:
    void* ptr_{nullptr}; ///< Start address of the registered, externally owned host region.

    /** @brief Unregisters the owned host region and restores the empty state. */
    void unregister()
    {
        if (ptr_)
        {
            cudaHostUnregister(ptr_);
            ptr_ = nullptr;
        }
    }
};

}  // namespace firefly::device
