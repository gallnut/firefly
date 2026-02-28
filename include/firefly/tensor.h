#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tensor_view.h"
#include "types.h"

namespace firefly
{

class Tensor
{
public:
    Tensor() = default;

    Tensor(std::vector<int64_t> shape, DType dtype, Device device);

    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    Tensor clone() const;

    void reshape(std::vector<int64_t> new_shape);

    void print() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    static Tensor from_external(void* data_ptr, std::vector<int64_t> shape, DType dtype, Device device);

public:
    template <typename T, size_t Rank>
    auto view() const
    {
        if (shape_.size() != Rank)
        {
            throw std::runtime_error("Tensor rank mismatch in view()");
        }

        std::array<int64_t, Rank> extents_arr;
        std::ranges::copy_n(shape_.begin(), Rank, extents_arr.begin());
        auto extents = firefly::dextents<int64_t, Rank>(extents_arr);

        std::array<int64_t, Rank> strides_arr;
        std::ranges::copy_n(strides_.begin(), Rank, strides_arr.begin());
        auto mapping = firefly::layout_stride::mapping(extents, strides_arr);

        return firefly::mdspan<T, firefly::dextents<int64_t, Rank>, firefly::layout_stride>(static_cast<T*>(data_ptr_),
                                                                                            mapping);
    }

    template <typename T>
    T* data_as() const
    {
        return static_cast<T*>(data_ptr_);
    }

public:
    void*                       data() const { return data_ptr_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType                       dtype() const { return dtype_; }
    Device                      device() const { return device_; }
    int64_t                     numel() const { return numel_; }
    size_t                      nbytes() const { return numel_ * dtype_size(dtype_); }
    bool                        is_view() const { return is_view_; }

private:
    void*                data_ptr_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t              numel_ = 0;
    DType                dtype_;
    Device               device_;
    bool                 is_view_ = false;
};

}  // namespace firefly