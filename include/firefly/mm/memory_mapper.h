// memory_mapper.h
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "firefly/hal/pinned_memory.h"

namespace firefly::mm
{

class MappedFile
{
public:
    explicit MappedFile(const std::string &path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            throw std::runtime_error("Failed to open file: " + path);
        }

        struct stat sb;
        if (::fstat(fd_, &sb) < 0)
        {
            ::close(fd_);
            throw std::runtime_error("Failed to stat file: " + path);
        }
        size_ = sb.st_size;

        data_ = static_cast<const std::byte *>(::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
        if (data_ == MAP_FAILED)
        {
            ::close(fd_);
            throw std::runtime_error("Failed to mmap file: " + path);
        }
    }

    ~MappedFile()
    {
        if (data_ && data_ != MAP_FAILED)
        {
            ::munmap(const_cast<std::byte *>(data_), size_);
        }

        if (fd_ >= 0)
        {
            ::close(fd_);
        }
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    MappedFile(MappedFile &&other) noexcept : fd_(other.fd_), size_(other.size_), data_(other.data_)
    {
        other.fd_ = -1;
        other.size_ = 0;
        other.data_ = nullptr;
    }

    MappedFile &operator=(MappedFile &&other) noexcept
    {
        if (this != &other)
        {
            if (data_)
            {
                ::munmap(const_cast<std::byte *>(data_), size_);
            }
            if (fd_ >= 0)
            {
                ::close(fd_);
            }
            fd_ = other.fd_;
            size_ = other.size_;
            data_ = other.data_;

            other.fd_ = -1;
            other.size_ = 0;
            other.data_ = nullptr;
        }

        return *this;
    }

    [[nodiscard]]
    const std::byte *data() const
    {
        return data_;
    }

    [[nodiscard]]
    std::size_t size() const
    {
        return size_;
    }

private:
    int              fd_{-1};
    std::size_t      size_{0};
    const std::byte *data_{nullptr};
};

/**
 * @brief Zero-copy GPU access via Locked Pages (Pinned Memory).
 * Composes MappedFile (layout) and PinnedRegistration (DMA access).
 * Falls back to regular mapped memory if pinning is not supported.
 */
class PinnedMappedFile
{
public:
    explicit PinnedMappedFile(const std::string &path) : file_(path)
    {
        // Register memory as Pinned Memory, allowing GPU DMA access
        auto res = hal::PinnedRegistration::register_memory(const_cast<std::byte *>(file_.data()), file_.size(),
                                                            cudaHostRegisterReadOnly);

        if (res)
        {
            registration_ = std::move(res.value());
            is_pinned_ = true;
        }
        else
        {
            // Clear the CUDA error state since we are deliberately ignoring this failure
            cudaGetLastError();
        }
        // If registration fails (e.g. not supported), we fall back to standard pageable memory
        // No exception is thrown, we just don't get the perf boost.
    }

    // Default move semantics work because member moves are correct
    PinnedMappedFile(PinnedMappedFile &&) noexcept = default;
    PinnedMappedFile &operator=(PinnedMappedFile &&) noexcept = default;

    // No copy
    PinnedMappedFile(const PinnedMappedFile &) = delete;
    PinnedMappedFile &operator=(const PinnedMappedFile &) = delete;

    ~PinnedMappedFile() = default;

    [[nodiscard]] const std::byte *data() const { return file_.data(); }
    [[nodiscard]] size_t           size() const { return file_.size(); }
    [[nodiscard]] bool             is_pinned() const { return is_pinned_; }

private:
    MappedFile              file_;
    hal::PinnedRegistration registration_;
    bool                    is_pinned_{false};
};

}  // namespace firefly::mm