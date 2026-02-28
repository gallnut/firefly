#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "firefly/mm/allocator.h"

namespace firefly::mm
{

template <Device D>
class ModelWeightPool
{
public:
    static constexpr size_t DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024;

    explicit ModelWeightPool(size_t chunk_size = DEFAULT_CHUNK_SIZE)
        : chunk_size_(chunk_size), active_chunk_(nullptr), total_allocated_(0)
    {
    }

    ~ModelWeightPool() { reset(); }

    ModelWeightPool(const ModelWeightPool&) = delete;
    ModelWeightPool& operator=(const ModelWeightPool&) = delete;

    ModelWeightPool(ModelWeightPool&& other) noexcept
        : chunk_size_(other.chunk_size_),
          chunks_(std::move(other.chunks_)),
          active_chunk_(other.active_chunk_),
          total_allocated_(other.total_allocated_)
    {
        other.active_chunk_ = nullptr;
        other.total_allocated_ = 0;
    }

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

    void reserve(size_t bytes)
    {
        size_t initial_size = std::max(bytes, chunk_size_);

        auto new_chunk = std::make_unique<Chunk>(initial_size);
        active_chunk_ = new_chunk.get();
        chunks_.push_back(std::move(new_chunk));

        total_allocated_ += initial_size;
    }

    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = 256)
    {
        assert((alignment & (alignment - 1)) == 0 && "Alignment must be a power of 2");

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
            auto   large_chunk = std::make_unique<Chunk>(alloc_size);

            void* ptr = large_chunk->try_allocate(bytes, alignment);
            if (!ptr) throw std::runtime_error("ModelWeightPool: Large memory allocation failed unexpectedly.");

            chunks_.push_back(std::move(large_chunk));
            total_allocated_ += alloc_size;

            return ptr;
        }

        auto new_chunk = std::make_unique<Chunk>(chunk_size_);
        active_chunk_ = new_chunk.get();

        void* ptr = active_chunk_->try_allocate(bytes, alignment);
        if (!ptr)
        {
            throw std::runtime_error("ModelWeightPool: Allocation failed even in a new chunk.");
        }

        chunks_.push_back(std::move(new_chunk));
        total_allocated_ += chunk_size_;

        return ptr;
    }

    void reset()
    {
        chunks_.clear();
        active_chunk_ = nullptr;
        total_allocated_ = 0;
    }

    [[nodiscard]] size_t total_memory() const { return total_allocated_; }

private:
    struct Chunk
    {
        void*  ptr{nullptr};
        size_t capacity;
        size_t offset;

        explicit Chunk(size_t cap) : capacity(cap), offset(0) { ptr = DeviceAllocator<D>::allocate(capacity); }

        ~Chunk()
        {
            if (ptr)
            {
                DeviceAllocator<D>::free(ptr);
            }
        }

        Chunk(const Chunk&) = delete;
        Chunk& operator=(const Chunk&) = delete;

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
    };

    size_t                              chunk_size_;
    std::vector<std::unique_ptr<Chunk>> chunks_;
    Chunk*                              active_chunk_{nullptr};
    size_t                              total_allocated_{0};
};

}  // namespace firefly::mm