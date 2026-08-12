#include "firefly/core/tensor.h"

#include <cuda_runtime.h>

#include <cstring>  // for std::memcpy
#include <format>
#include <limits>
#include <numeric>
#include <print>
#include <string>
#include <type_traits>
#include <vector>

#include "firefly/core/types.h"
#include "firefly/device/allocator.h"
#include "firefly/device/error.h"

namespace firefly
{

namespace
{
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

Result<std::pair<std::vector<int64_t>, int64_t>> make_contiguous_layout(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size());
    int64_t              numel = shape.empty() ? 0 : 1;
    int64_t              stride = 1;
    for (int index = static_cast<int>(shape.size()) - 1; index >= 0; --index)
    {
        if (shape[index] < 0)
            return unexpected(Error{ErrorCode::InvalidArgument, "tensor dimensions must be nonnegative"});
        strides[index] = stride;
        if (shape[index] != 0 && stride > std::numeric_limits<int64_t>::max() / shape[index])
            return unexpected(Error{ErrorCode::InvalidArgument, "tensor element count overflows int64"});
        stride *= shape[index];
        numel *= shape[index];
    }
    return std::pair{std::move(strides), numel};
}
}  // namespace

Tensor::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t numel, DType dtype, Device device,
               const device::Context& context)
    : shape_(std::move(shape)), strides_(std::move(strides)), numel_(numel), dtype_(dtype), device_(device),
      allocation_context_(context)
{
}

Result<Tensor> Tensor::create(std::vector<int64_t> shape, DType dtype, Device device, const device::Context& context)
{
    if (dtype == DType::UNKNOWN || element_size(dtype) == 0)
        return unexpected(Error{ErrorCode::InvalidArgument, "cannot allocate tensor with unknown dtype"});
    auto layout = FIREFLY_TRY_CONTEXT(make_contiguous_layout(shape), "compute contiguous tensor layout");
    Tensor tensor(std::move(shape), std::move(layout.first), layout.second, dtype, device, context);
    const size_t bytes = tensor.nbytes();
    if (bytes == 0) return tensor;
    if (device == Device::CPU)
        tensor.data_ptr_ = FIREFLY_TRY_CONTEXT(DeviceAllocator<Device::CPU>::allocate(bytes), "allocate CPU tensor");
    else if (device == Device::CUDA)
        tensor.data_ptr_ = FIREFLY_TRY_CONTEXT(
            DeviceAllocator<Device::CUDA>::allocate(bytes, context.stream()), "allocate CUDA tensor");
    else
        return unexpected(Error{ErrorCode::InvalidArgument, "unknown tensor device"});
    return tensor;
}

Tensor::~Tensor()
{
    if (data_ptr_ != nullptr && !is_view_)
    {
        if (device_ == Device::CUDA) DeviceAllocator<Device::CUDA>::free(data_ptr_, allocation_context_.stream());
        else DeviceAllocator<Device::CPU>::free(data_ptr_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(other.data_ptr_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      numel_(other.numel_),
      dtype_(other.dtype_),
      device_(other.device_),
      is_view_(other.is_view_),
      allocation_context_(std::move(other.allocation_context_))
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
            if (device_ == Device::CUDA) DeviceAllocator<Device::CUDA>::free(data_ptr_, allocation_context_.stream());
            else DeviceAllocator<Device::CPU>::free(data_ptr_);
        }

        data_ptr_ = other.data_ptr_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        numel_ = other.numel_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        is_view_ = other.is_view_;
        allocation_context_ = std::move(other.allocation_context_);

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

Result<Tensor> Tensor::from_external(void* data_ptr, std::vector<int64_t> shape, std::vector<int64_t> strides,
                                     DType dtype, Device device)
{
    if (shape.size() != strides.size())
        return unexpected(Error{ErrorCode::InvalidArgument, "tensor view shape and stride ranks differ"});
    Tensor tensor;
    tensor.data_ptr_ = data_ptr;
    tensor.shape_ = std::move(shape);
    tensor.strides_ = std::move(strides);
    tensor.dtype_ = dtype;
    tensor.device_ = device;
    tensor.is_view_ = true;
    tensor.numel_ = std::accumulate(tensor.shape_.begin(), tensor.shape_.end(), int64_t{1}, std::multiplies<>());
    return tensor;
}

Result<Tensor> Tensor::clone() const
{
    Tensor new_tensor = FIREFLY_TRY_CONTEXT(Tensor::create(shape_, dtype_, device_, allocation_context_),
                                            "allocate cloned tensor");
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
                return unexpected(device::cuda_error(err, "copy cloned CUDA tensor"));
            }
        }
    }
    return new_tensor;
}

Status Tensor::reshape(std::vector<int64_t> new_shape)
{
    auto layout = FIREFLY_TRY_CONTEXT(make_contiguous_layout(new_shape), "compute reshaped tensor layout");
    if (layout.second != numel_)
        return unexpected(Error{ErrorCode::InvalidArgument, "reshape changes the tensor element count"});

    shape_ = std::move(new_shape);
    strides_ = std::move(layout.first);
    return {};
}

void Tensor::print() const { std::println("{}", tensor_to_string(*this)); }

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    os << tensor_to_string(t);
    return os;
}

}  // namespace firefly
