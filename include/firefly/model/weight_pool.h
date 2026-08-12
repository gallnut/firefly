#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "firefly/device/allocator.h"

namespace firefly::model
{

/**
 * @brief Monotonic chunk allocator that owns model weights for the model lifetime.
 * @tparam D Device domain used for backing chunks.
 *
 * Individual allocations cannot be freed. `reset` releases every chunk at once, which
 * keeps checkpoint loading fast and avoids allocator fragmentation.
 */
template <Device D>
class ModelWeightPool
{
public:
    /**
     * @brief Default backing-chunk size used for ordinary weight allocations.
     * @note Large requests exceeding half this size receive dedicated chunks instead.
     */
    static constexpr size_t DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024;

    /** @brief Constructs an empty pool with the requested ordinary chunk size. */
    explicit ModelWeightPool(size_t chunk_size = DEFAULT_CHUNK_SIZE)
        : chunk_size_(chunk_size), active_chunk_(nullptr), total_allocated_(0)
    {
    }

    /** @brief Releases every backing allocation. */
    ~ModelWeightPool() { reset(); }

    /** @brief Pool ownership cannot be copied. */
    ModelWeightPool(const ModelWeightPool&) = delete;
    /** @brief Pool ownership cannot be copy-assigned. */
    ModelWeightPool& operator=(const ModelWeightPool&) = delete;

    /** @brief Transfers all backing chunks from another pool. */
    ModelWeightPool(ModelWeightPool&& other) noexcept
        : chunk_size_(other.chunk_size_),
          chunks_(std::move(other.chunks_)),
          active_chunk_(other.active_chunk_),
          total_allocated_(other.total_allocated_)
    {
        other.active_chunk_ = nullptr;
        other.total_allocated_ = 0;
    }

    /** @brief Releases current chunks and transfers ownership from another pool. */
    ModelWeightPool& operator=(ModelWeightPool&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            chunk_size_ = other.chunk_size_;
            chunks_ = std::move(other.chunks_);
            active_chunk_ = other.active_chunk_;
            total_allocated_ = other.total_allocated_;

            other.active_chunk_ = nullptr;
            other.total_allocated_ = 0;
        }
        return *this;
    }

    /**
     * @brief Adds an initial backing chunk large enough for `bytes`.
     * @param bytes Minimum capacity of the newly allocated chunk.
     */
    [[nodiscard]] Status reserve(size_t bytes)
    {
        size_t initial_size = std::max(bytes, chunk_size_);

        auto new_chunk = FIREFLY_TRY_CONTEXT(Chunk::create(initial_size), "reserve model weight pool");
        active_chunk_ = new_chunk.get();
        chunks_.push_back(std::move(new_chunk));

        total_allocated_ += initial_size;
        return {};
    }

    /**
     * @brief Allocates aligned storage that remains valid until pool reset or destruction.
     * @param bytes Requested payload size.
     * @param alignment Power-of-two byte alignment.
     * @return Aligned pointer in device domain `D`.
     * @return Aligned pointer or a structured validation or device-allocation error.
     */
    [[nodiscard]] Result<void*> allocate(size_t bytes, size_t alignment = 256)
    {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            return unexpected(Error{ErrorCode::InvalidArgument, "weight alignment must be a nonzero power of two"});

        if (active_chunk_)
        {
            void* ptr = active_chunk_->try_allocate(bytes, alignment);
            if (ptr)
            {
                return ptr;
            }
        }

        if (bytes > chunk_size_ / 2)
        {
            size_t alloc_size = bytes + alignment;
            auto large_chunk = FIREFLY_TRY_CONTEXT(Chunk::create(alloc_size), "allocate large model weight chunk");

            void* ptr = large_chunk->try_allocate(bytes, alignment);
            if (!ptr)
                return unexpected(Error{ErrorCode::Internal, "new large weight chunk cannot satisfy allocation"});

            chunks_.push_back(std::move(large_chunk));
            total_allocated_ += alloc_size;

            return ptr;
        }

        auto new_chunk = FIREFLY_TRY_CONTEXT(Chunk::create(chunk_size_), "allocate model weight chunk");
        active_chunk_ = new_chunk.get();

        void* ptr = active_chunk_->try_allocate(bytes, alignment);
        if (!ptr)
        {
            return unexpected(Error{ErrorCode::Internal, "new weight chunk cannot satisfy allocation"});
        }

        chunks_.push_back(std::move(new_chunk));
        total_allocated_ += chunk_size_;

        return ptr;
    }

    /** @brief Releases all backing chunks and invalidates every pointer returned by the pool. */
    void reset()
    {
        chunks_.clear();
        active_chunk_ = nullptr;
        total_allocated_ = 0;
    }

    /** @brief Returns total backing capacity acquired from the device allocator. */
    [[nodiscard]] size_t total_memory() const { return total_allocated_; }

private:
    /** @brief One owned backing allocation with a monotonic byte offset. */
    struct Chunk
    {
        void*  ptr{nullptr}; ///< Owned backing allocation in device domain `D`.
        size_t capacity; ///< Total backing allocation size in bytes.
        size_t offset; ///< First unconsumed byte before alignment padding.

        /** @brief Allocates one backing region of `cap` bytes. */
        static Result<std::unique_ptr<Chunk>> create(size_t cap)
        {
            auto chunk = std::unique_ptr<Chunk>(new Chunk(cap));
            chunk->ptr = FIREFLY_TRY_CONTEXT(DeviceAllocator<D>::allocate(cap), "allocate weight chunk backing");
            return chunk;
        }

        /** @brief Releases the backing region. */
        ~Chunk()
        {
            if (ptr)
            {
                DeviceAllocator<D>::free(ptr);
            }
        }

        /** @brief Chunk ownership cannot be copied. */
        Chunk(const Chunk&) = delete;
        /** @brief Chunk ownership cannot be copy-assigned. */
        Chunk& operator=(const Chunk&) = delete;

        /**
         * @brief Attempts an aligned monotonic allocation.
         * @param bytes Requested payload size.
         * @param alignment Required power-of-two byte alignment.
         * @return Aligned device-domain pointer, or null when this chunk lacks capacity.
         */
        [[nodiscard]] void* try_allocate(size_t bytes, size_t alignment)
        {
            uintptr_t current_addr = reinterpret_cast<uintptr_t>(static_cast<std::byte*>(ptr) + offset);

            uintptr_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
            size_t    padding = aligned_addr - current_addr;

            if (offset + padding + bytes > capacity)
            {
                return nullptr;
            }

            std::byte* aligned_ptr = static_cast<std::byte*>(ptr) + offset + padding;
            offset += padding + bytes;

            return aligned_ptr;
        }

    private:
        explicit Chunk(size_t cap) : capacity(cap), offset(0) {}
    };

    size_t                              chunk_size_; ///< Capacity used for ordinary newly allocated chunks.
    std::vector<std::unique_ptr<Chunk>> chunks_; ///< All owned chunks, including dedicated large allocations.
    Chunk*                              active_chunk_{nullptr}; ///< Non-owning pointer to the ordinary allocation chunk.
    size_t                              total_allocated_{0}; ///< Sum of backing capacities acquired from the allocator.
};

}  // namespace firefly::model
