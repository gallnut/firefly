// memory_mapper.h
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cerrno>
#include <cstring>
#include <string>

#include "firefly/device/pinned_memory.h"

namespace firefly::storage
{

/** @brief Move-only owner of a read-only POSIX memory mapping and its file descriptor. */
class MappedFile
{
public:
    /**
     * @brief Opens and maps an entire file read-only with shared mapping semantics.
     * @param path File to map.
     * @return Mapping owner or a structured POSIX I/O error.
     */
    [[nodiscard]] static Result<MappedFile> create(const std::string &path)
    {
        MappedFile file;
        file.fd_ = ::open(path.c_str(), O_RDONLY);
        if (file.fd_ < 0)
            return unexpected(Error{ErrorCode::Io, "failed to open file " + path + ": " + std::strerror(errno),
                                    errno});

        struct stat sb;
        if (::fstat(file.fd_, &sb) < 0)
        {
            const int native_error = errno;
            ::close(file.fd_);
            file.fd_ = -1;
            return unexpected(Error{ErrorCode::Io, "failed to stat file " + path + ": " +
                                                           std::strerror(native_error), native_error});
        }
        if (sb.st_size <= 0)
        {
            ::close(file.fd_);
            file.fd_ = -1;
            return unexpected(Error{ErrorCode::Parse, "cannot map empty file: " + path});
        }
        file.size_ = static_cast<std::size_t>(sb.st_size);

        file.data_ = static_cast<const std::byte *>(
            ::mmap(nullptr, file.size_, PROT_READ, MAP_SHARED, file.fd_, 0));
        if (file.data_ == MAP_FAILED)
        {
            const int native_error = errno;
            ::close(file.fd_);
            file.fd_ = -1;
            file.data_ = nullptr;
            return unexpected(Error{ErrorCode::Io, "failed to map file " + path + ": " +
                                                           std::strerror(native_error), native_error});
        }
        return file;
    }

    /** @brief Unmaps the file contents and closes the owned descriptor. */
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

    /** @brief Mapping ownership cannot be copied. */
    MappedFile(const MappedFile &) = delete;
    /** @brief Mapping ownership cannot be copy-assigned. */
    MappedFile &operator=(const MappedFile &) = delete;

    /** @brief Transfers mapping and descriptor ownership from another object. */
    MappedFile(MappedFile &&other) noexcept : fd_(other.fd_), size_(other.size_), data_(other.data_)
    {
        other.fd_ = -1;
        other.size_ = 0;
        other.data_ = nullptr;
    }

    /** @brief Releases the current mapping and transfers ownership from another object. */
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

    /** @brief Returns the first mapped byte; the pointer is valid until move, assignment, or destruction. */
    [[nodiscard]]
    const std::byte *data() const
    {
        return data_;
    }

    /** @brief Returns the mapped file size in bytes. */
    [[nodiscard]]
    std::size_t size() const
    {
        return size_;
    }

private:
    /** @brief Constructs an empty mapping populated only by `create`. */
    MappedFile() = default;
    int              fd_{-1}; ///< Owned read-only file descriptor.
    std::size_t      size_{0}; ///< Mapped file length in bytes.
    const std::byte *data_{nullptr}; ///< Start address of the read-only mapping.
};

/**
 * @brief Zero-copy GPU access via Locked Pages (Pinned Memory).
 * Composes MappedFile (layout) and PinnedRegistration (DMA access).
 * Falls back to regular mapped memory if pinning is not supported.
 */
class PinnedMappedFile
{
public:
    /**
     * @brief Maps a file and attempts to register the mapping as read-only pinned host memory.
     * @param path File whose entire contents are mapped.
     * @return Mapped file with optional registration, or a structured mapping error.
     */
    [[nodiscard]] static Result<PinnedMappedFile> create(const std::string &path)
    {
        PinnedMappedFile pinned(FIREFLY_TRY(MappedFile::create(path)));
        // Register memory as Pinned Memory, allowing GPU DMA access
        auto res = device::PinnedRegistration::register_memory(const_cast<std::byte *>(pinned.file_.data()), pinned.file_.size(),
                                                            cudaHostRegisterReadOnly);

        if (res)
        {
            pinned.registration_ = std::move(res.value());
            pinned.is_pinned_ = true;
        }
        else
        {
            // Clear the CUDA error state since we are deliberately ignoring this failure
            cudaGetLastError();
        }
        return pinned;
    }

    /** @brief Transfers mapping and optional registration ownership. */
    PinnedMappedFile(PinnedMappedFile &&) noexcept = default;
    /** @brief Releases current resources and transfers mapping ownership. */
    PinnedMappedFile &operator=(PinnedMappedFile &&) noexcept = default;

    /** @brief Mapping and registration ownership cannot be copied. */
    PinnedMappedFile(const PinnedMappedFile &) = delete;
    /** @brief Mapping and registration ownership cannot be copy-assigned. */
    PinnedMappedFile &operator=(const PinnedMappedFile &) = delete;

    /** @brief Releases registration before unmapping the file. */
    ~PinnedMappedFile() = default;

    /** @brief Returns the first mapped byte. */
    [[nodiscard]] const std::byte *data() const { return file_.data(); }
    /** @brief Returns the mapped file size in bytes. */
    [[nodiscard]] size_t           size() const { return file_.size(); }
    /** @brief Returns whether CUDA host registration succeeded. */
    [[nodiscard]] bool             is_pinned() const { return is_pinned_; }

private:
    /** @brief Wraps an already validated mapping before optional CUDA registration. */
    explicit PinnedMappedFile(MappedFile file) : file_(std::move(file)) {}
    MappedFile                 file_; ///< Owned read-only mapping that must outlive registration.
    device::PinnedRegistration registration_; ///< Optional CUDA registration released before unmapping.
    bool                       is_pinned_{false}; ///< Whether `registration_` was created successfully.
};

}  // namespace firefly::storage
