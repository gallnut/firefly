#include "firefly/tensor.h"

#include <cuda_runtime.h>

#include <cstring>  // for std::memcpy
#include <format>
#include <numeric>
#include <print>
#include <string>
#include <type_traits>
#include <vector>

#include "firefly/mm/allocator.h"
#include "firefly/types.h"

namespace firefly
{

namespace
{
template <typename F>
auto dispatch_device(Device d, F&& func)
{
    switch (d)
    {
        case Device::CPU:
            return func(std::integral_constant<Device, Device::CPU>{});
        case Device::CUDA:
            return func(std::integral_constant<Device, Device::CUDA>{});
        default:
            throw std::runtime_error("Unknown device type");
    }
}

std::string format_shape(const std::vector<int64_t>& shape)
{
    if (shape.empty()) return "[]";
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

template <typename T>
std::string format_flat_data(const void* ptr, int64_t numel)
{
    const T*    data = static_cast<const T*>(ptr);
    std::string s = "[";

    const int64_t limit = 20;
    int64_t       count = std::min(numel, limit);

    for (int64_t i = 0; i < count; ++i)
    {
        if (i > 0) s += ", ";

        if constexpr (std::is_floating_point_v<T>)
        {
            s += std::format("{:.4f}", data[i]);
        }
        else
        {
            s += std::format("{}", data[i]);
        }
    }

    if (numel > limit)
    {
        s += ", ...";
    }
    s += "]";
    return s;
}

std::string tensor_to_string(const Tensor& t)
{
    std::string s = "Tensor(";

    s += std::format("shape={}, device={}, dtype={})\n", format_shape(t.shape()),
                     (t.device() == Device::CPU ? "CPU" : "CUDA"), dtype_to_string(t.dtype()));

    if (t.numel() > 0 && t.data())
    {
        const void*          print_ptr = t.data();
        std::vector<uint8_t> host_buffer;

        if (t.device() == Device::CUDA)
        {
            size_t bytes = t.numel() * element_size(t.dtype());
            host_buffer.resize(bytes);
            cudaError_t err = cudaMemcpy(host_buffer.data(), t.data(), bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                return s + "[Error copying CUDA data]";
            }
            print_ptr = host_buffer.data();
        }

        switch (t.dtype())
        {
            case DType::F32:
                s += format_flat_data<float>(print_ptr, t.numel());
                break;
            case DType::I32:
                s += format_flat_data<int32_t>(print_ptr, t.numel());
                break;
            case DType::I64:
                s += format_flat_data<int64_t>(print_ptr, t.numel());
                break;
            case DType::I8:
                s += format_flat_data<int8_t>(print_ptr, t.numel());
                break;
            default:
                s += "[Data print not implemented]";
        }
    }

    return s;
}

}  // namespace

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device)
{
    if (!shape_.empty())
    {
        strides_.resize(shape_.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i)
        {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    numel_ = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
    size_t bytes = numel_ * element_size(dtype);

    if (bytes > 0)
    {
        data_ptr_ =
            dispatch_device(device, [=](auto dev_const) { return DeviceAllocator<dev_const.value>::allocate(bytes); });
    }
}

Tensor::~Tensor()
{
    if (data_ptr_ != nullptr && !is_view_)
    {
        dispatch_device(device_, [ptr = data_ptr_](auto dev_const) { DeviceAllocator<dev_const.value>::free(ptr); });
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(other.data_ptr_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      numel_(other.numel_),
      dtype_(other.dtype_),
      device_(other.device_),
      is_view_(other.is_view_)
{
    other.data_ptr_ = nullptr;
    other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        if (data_ptr_ != nullptr && !is_view_)
        {
            dispatch_device(device_,
                            [ptr = data_ptr_](auto dev_const) { DeviceAllocator<dev_const.value>::free(ptr); });
        }

        data_ptr_ = other.data_ptr_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        numel_ = other.numel_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        is_view_ = other.is_view_;

        other.data_ptr_ = nullptr;
        other.numel_ = 0;
    }
    return *this;
}

Tensor Tensor::from_external(void* data_ptr, std::vector<int64_t> shape, DType dtype, Device device)
{
    Tensor t;
    t.data_ptr_ = data_ptr;
    t.shape_ = std::move(shape);
    t.dtype_ = dtype;
    t.device_ = device;
    t.is_view_ = true;

    if (t.shape_.empty())
    {
        t.numel_ = 0;
    }
    else
    {
        t.numel_ = 1;
        t.strides_.resize(t.shape_.size());
        int64_t stride = 1;

        for (int i = static_cast<int>(t.shape_.size()) - 1; i >= 0; --i)
        {
            t.strides_[i] = stride;
            stride *= t.shape_[i];
            t.numel_ *= t.shape_[i];
        }
    }
    return t;
}

Tensor Tensor::clone() const
{
    Tensor new_tensor(shape_, dtype_, device_);
    size_t bytes = numel_ * element_size(dtype_);

    if (bytes > 0 && data_ptr_)
    {
        if (device_ == Device::CPU)
        {
            std::memcpy(new_tensor.data(), data_ptr_, bytes);
        }
        else
        {
            cudaError_t err = cudaMemcpy(new_tensor.data(), data_ptr_, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Tensor clone failed: CUDA copy error");
            }
        }
    }
    return new_tensor;
}

void Tensor::reshape(std::vector<int64_t> new_shape)
{
    int64_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;

    if (new_numel != numel_)
    {
        throw std::runtime_error("Reshape: numel mismatch");
    }

    shape_ = std::move(new_shape);

    if (!shape_.empty())
    {
        strides_.resize(shape_.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i)
        {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }
}

void Tensor::print() const { std::println("{}", tensor_to_string(*this)); }

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    os << tensor_to_string(t);
    return os;
}

}  // namespace firefly