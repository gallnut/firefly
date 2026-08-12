#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "firefly/core/error.h"
#include "firefly/core/tensor_view.h"
#include "firefly/core/types.h"
#include "firefly/device/context.h"

namespace firefly
{

/**
 * @brief Move-only owner or non-owning view of a dense multidimensional allocation.
 *
 * A tensor records shape, element strides, scalar type, and device placement. Tensors
 * created by the allocating constructor own their storage, while tensors returned by
 * `from_external` borrow storage whose lifetime must outlive the tensor view.
 */
class Tensor
{
public:
    /** @brief Constructs an empty tensor with no storage and unknown scalar type. */
    Tensor() = default;

    /**
     * @brief Allocates contiguous storage for a tensor.
     * @param shape Logical dimensions in row-major order.
     * @param dtype Scalar storage format.
     * @param device Allocation domain.
     * @param context CUDA allocation context; ignored for CPU allocations.
     * @return Owned tensor or a structured shape, dtype, or allocation error.
     */
    [[nodiscard]] static Result<Tensor> create(std::vector<int64_t> shape, DType dtype, Device device,
                                               const device::Context& context = {});

    /** @brief Releases owned storage; borrowed external storage is never freed. */
    ~Tensor();

    /** @brief Tensor storage cannot be copied implicitly. */
    Tensor(const Tensor&) = delete;
    /** @brief Tensor storage cannot be copy-assigned implicitly. */
    Tensor& operator=(const Tensor&) = delete;

    /** @brief Transfers storage ownership and metadata from another tensor. */
    Tensor(Tensor&&) noexcept;
    /** @brief Releases current storage and transfers ownership from another tensor. */
    Tensor& operator=(Tensor&&) noexcept;

    /**
     * @brief Creates an owning copy of the tensor on the same device.
     * @return A contiguous tensor containing the same element values.
     */
    [[nodiscard]] Result<Tensor> clone() const;

    /**
     * @brief Changes the logical shape without moving or reallocating storage.
     * @param new_shape Replacement dimensions whose product must equal `numel()`.
     * @return Success or an invalid-shape error when the element count changes.
     */
    [[nodiscard]] Status reshape(std::vector<int64_t> new_shape);

    /** @brief Prints tensor metadata and values to standard output for diagnostics. */
    void print() const;

    /**
     * @brief Writes a diagnostic representation of a tensor to an output stream.
     * @param os Destination stream.
     * @param tensor Tensor to format.
     * @return The destination stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    /**
     * @brief Creates a contiguous non-owning tensor view over external storage.
     * @param data_ptr Pointer in the address space identified by `device`.
     * @param shape Logical tensor dimensions.
     * @param dtype Scalar storage format.
     * @param device Storage domain of `data_ptr`.
     * @return A tensor view that never frees `data_ptr`.
     */
    static Tensor from_external(void* data_ptr, std::vector<int64_t> shape, DType dtype, Device device);
    /**
     * @brief Creates a strided non-owning tensor view over external storage.
     * @param data_ptr Pointer in the address space identified by `device`.
     * @param shape Logical tensor dimensions.
     * @param strides Element strides for each dimension.
     * @param dtype Scalar storage format.
     * @param device Storage domain of `data_ptr`.
     * @return A tensor view that never frees `data_ptr`.
     */
    static Result<Tensor> from_external(void* data_ptr, std::vector<int64_t> shape, std::vector<int64_t> strides,
                                        DType dtype, Device device);

public:
    /**
     * @brief Returns a typed, fixed-rank multidimensional view of the tensor storage.
     * @tparam T Element type expected by the caller; it must match the tensor dtype.
     * @tparam Rank Required logical rank.
     * @return A non-owning mdspan using the tensor's recorded element strides.
     * @return Typed view or an invalid-shape error when the tensor rank differs from `Rank`.
     */
    template <typename T, size_t Rank>
    auto view() const -> Result<firefly::mdspan<T, firefly::dextents<int64_t, Rank>, firefly::layout_stride>>
    {
        using extents_type = firefly::dextents<int64_t, Rank>;

        if (shape_.size() != Rank)
        {
            return unexpected(Error{ErrorCode::InvalidArgument, "tensor rank does not match requested view rank"});
        }

        std::array<int64_t, Rank> extents_arr;
        std::ranges::copy_n(shape_.begin(), Rank, extents_arr.begin());
        auto extents = extents_type(extents_arr);

        std::array<int64_t, Rank> strides_arr;
        std::ranges::copy_n(strides_.begin(), Rank, strides_arr.begin());
        auto mapping = firefly::layout_stride::mapping<extents_type>(extents, strides_arr);

        return firefly::mdspan<T, extents_type, firefly::layout_stride>(static_cast<T*>(data_ptr_), mapping);
    }

    /**
     * @brief Casts the untyped storage pointer to an element pointer.
     * @tparam T Expected element type.
     * @return The underlying pointer, or `nullptr` for an empty tensor.
     * @warning No runtime dtype validation is performed.
     */
    template <typename T>
    T* data_as() const
    {
        return static_cast<T*>(data_ptr_);
    }

public:
    /** @brief Returns the untyped storage pointer. */
    void*                       data() const { return data_ptr_; }
    /** @brief Returns the logical dimensions. */
    const std::vector<int64_t>& shape() const { return shape_; }
    /** @brief Returns element strides for each logical dimension. */
    const std::vector<int64_t>& strides() const { return strides_; }
    /** @brief Returns the scalar storage format. */
    DType                       dtype() const { return dtype_; }
    /** @brief Returns the allocation domain. */
    Device                      device() const { return device_; }
    /** @brief Returns the logical number of scalar elements. */
    int64_t                     numel() const { return numel_; }
    /** @brief Returns the logical storage size in bytes. */
    size_t                      nbytes() const { return numel_ * dtype_size(dtype_); }
    /** @brief Returns whether the tensor borrows externally owned storage. */
    bool                        is_view() const { return is_view_; }

private:
    /** @brief Constructs metadata for an allocating factory before storage acquisition. */
    Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t numel, DType dtype, Device device,
           const device::Context& context);

    void*                data_ptr_ = nullptr; ///< Owned or borrowed address of the first logical element.
    std::vector<int64_t> shape_; ///< Logical extent of every dimension.
    std::vector<int64_t> strides_; ///< Row-major or externally supplied element strides.
    int64_t              numel_ = 0; ///< Product of all logical dimensions.
    DType                dtype_ = DType::UNKNOWN; ///< Scalar storage format.
    Device               device_ = Device::CPU; ///< Address space containing `data_ptr_`.
    bool                 is_view_ = false; ///< Whether storage is externally owned and must not be freed.
    device::Context      allocation_context_; ///< CUDA stream lifetime used for stream-ordered deallocation.
};

}  // namespace firefly
