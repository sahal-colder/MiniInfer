#include "mini_infer/core/mini_infer_tensor.h"

#include <limits>
#include <utility>

namespace mini_infer {

Tensor::Tensor(Tensor&& other) noexcept
    : dtype_(std::exchange(other.dtype_, DataType::UNKNOWN)),
      dims_(std::move(other.dims_)), size_(std::exchange(other.size_, 0)),
      buffer_(std::move(other.buffer_)) {
    other.dims_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        dtype_ = std::exchange(other.dtype_, DataType::UNKNOWN);
        dims_ = std::move(other.dims_);
        size_ = std::exchange(other.size_, 0);
        buffer_ = std::move(other.buffer_);
        other.dims_.clear();
    }
    return *this;
}

std::size_t data_type_size(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
        return sizeof(float);
    case DataType::INT32:
        return sizeof(int32_t);
    case DataType::INT8:
        return sizeof(int8_t);
    default:
        throw std::invalid_argument("Unknown dtype");
    }
}

std::size_t Tensor::checked_size(const std::vector<int32_t>& dims) {
    if (dims.empty()) {
        throw std::invalid_argument("Shape must not be empty");
    }

    std::size_t count = 1;

    for (int32_t d : dims) {
        if (d <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }

        const auto dim = static_cast<std::size_t>(d);

        if (count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("Tensor element count overflow");
        }

        count *= dim;
    }

    return count;
}

Tensor::Tensor(
    DataType dtype,
    std::vector<int32_t> dims,
    std::shared_ptr<DeviceAllocator> allocator
)
    : dtype_(dtype),
      dims_(std::move(dims)),
      size_(checked_size(dims_)) {

    const std::size_t element_size = data_type_size(dtype_);

    if (size_ > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("Tensor byte size overflow");
    }

    if (allocator) {
        allocate(std::move(allocator));
    }
}

Tensor Tensor::borrow(
    DataType dtype,
    std::vector<int32_t> dims,
    void* ptr,
    DeviceType device
) {
    Tensor result(dtype, std::move(dims));

    if (!ptr ||
        (device != DeviceType::CPU && device != DeviceType::CUDA)) {
        throw std::invalid_argument("Invalid external storage");
    }

    if (reinterpret_cast<std::uintptr_t>(ptr) % data_type_size(dtype) != 0) {
        throw std::invalid_argument("Unaligned external storage");
    }

    result.buffer_ = std::make_shared<Buffer>(
        result.byte_size(),
        ptr,
        device
    );

    return result;
}

Tensor Tensor::view(std::size_t offset, std::vector<int32_t> dims) const {
    require_storage();
    Tensor result(dtype_, std::move(dims));
    // 用减法检查范围，避免 offset + result.size() 自身溢出。
    if (offset > size_ || result.size() > size_ - offset) {
        throw std::out_of_range("Tensor view exceeds source storage");
    }
    auto* start = static_cast<std::byte*>(buffer_->ptr())
                  + offset * data_type_size(dtype_);
    // 新 Buffer 只描述子区间，不释放子指针。自定义 deleter 持有
    // 原 Buffer 的 shared_ptr；最后一个 view 析构后，原存储才可
    // 释放。view 的 view 也通过这条所有权链保活根 Buffer。
    result.buffer_ = std::shared_ptr<Buffer>(
        new Buffer(result.byte_size(), start, device_type()),
        [owner = buffer_](Buffer* slice) {
            delete slice;
            (void)owner;  // owner 随 deleter 销毁，而不是随局部变量销毁。
        });
    return result;
}

void Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator) {
    if (!allocator ||
        size_ == 0 || dims_.empty() ||
        (allocator->device_type() != DeviceType::CPU &&
         allocator->device_type() != DeviceType::CUDA)) {
        throw std::invalid_argument(
            "Allocation needs a shape and valid allocator"
        );
    }

    if (buffer_) {
        throw std::logic_error("Tensor already has storage");
    }

    buffer_ = std::make_shared<Buffer>(
        byte_size(),
        std::move(allocator)
    );
}

std::size_t Tensor::byte_size() const {
    if (size_ == 0) {
        return 0;
    }

    return size_ * data_type_size(dtype_);
}

DeviceType Tensor::device_type() const {
    return buffer_ ? buffer_->device_type() : DeviceType::UNKNOWN;
}

void Tensor::require_storage() const {
    if (!buffer_) {
        throw std::logic_error("Tensor has no storage");
    }
}

void Tensor::check_index(std::size_t offset) const {
    require_storage();

    if (device_type() != DeviceType::CPU) {
        throw std::logic_error(
            "Host cannot index a CUDA tensor; copy to CPU first"
        );
    }

    if (offset >= size_) {
        throw std::out_of_range("Tensor index out of range");
    }
}

std::vector<std::size_t> Tensor::strides() const {
    std::vector<std::size_t> result(dims_.size());

    std::size_t stride = 1;

    for (std::size_t i = dims_.size(); i > 0; --i) {
        result[i - 1] = stride;
        stride *= static_cast<std::size_t>(dims_[i - 1]);
    }

    return result;
}

void Tensor::reshape(std::vector<int32_t> dims) {
    if (checked_size(dims) != size_) {
        throw std::invalid_argument(
            "reshape must preserve element count"
        );
    }

    dims_ = std::move(dims);
}

bool Tensor::overlaps(const Tensor& other) const {
    if (empty() ||
        other.empty() ||
        device_type() != other.device_type()) {
        return false;
    }

    const auto a = reinterpret_cast<std::uintptr_t>(buffer_->ptr());
    const auto b = reinterpret_cast<std::uintptr_t>(other.buffer_->ptr());

    if (a <= b) {
        return b - a < byte_size();
    }

    return a - b < other.byte_size();
}

void Tensor::copy_from(
    const Tensor& source,
    cudaStream_t stream
) {
    require_storage();
    source.require_storage();

    if (dtype_ != source.dtype_ || dims_ != source.dims_) {
        throw std::invalid_argument(
            "copy_from requires equal dtype and shape"
        );
    }

    const DeviceType src_device = source.device_type();
    const DeviceType dst_device = device_type();

    const bool src_valid =
        src_device == DeviceType::CPU || src_device == DeviceType::CUDA;
    const bool dst_valid =
        dst_device == DeviceType::CPU || dst_device == DeviceType::CUDA;

    if (!src_valid || !dst_valid) {
        throw std::logic_error("Unsupported tensor device");
    }

    if (src_device == dst_device &&
        buffer_->ptr() == source.buffer_->ptr()) {
        return;
    }

    if (overlaps(source)) {
        throw std::invalid_argument(
            "Overlapping copy is unsupported"
        );
    }

    MemcpyKind kind;

    if (src_device == DeviceType::CPU &&
        dst_device == DeviceType::CPU) {
        kind = MemcpyKind::CPU2CPU;
    } else if (src_device == DeviceType::CPU &&
               dst_device == DeviceType::CUDA) {
        kind = MemcpyKind::CPU2CUDA;
    } else if (src_device == DeviceType::CUDA &&
               dst_device == DeviceType::CPU) {
        kind = MemcpyKind::CUDA2CPU;
    } else {
        kind = MemcpyKind::CUDA2CUDA;
    }

    CPUDeviceAllocator copier;

    copier.copy(
        source.buffer_->ptr(),
        buffer_->ptr(),
        byte_size(),
        kind,
        stream
    );
}

Tensor Tensor::clone(
    std::shared_ptr<DeviceAllocator> allocator
) const {
    require_storage();

    if (!allocator) {
        throw std::invalid_argument("clone requires an allocator");
    }

    Tensor result(
        dtype_,
        dims_,
        std::move(allocator)
    );

    result.copy_from(*this);

    return result;
}

Tensor Tensor::to_cuda(
    std::shared_ptr<DeviceAllocator> cuda_allocator,
    cudaStream_t stream
) const {
    require_storage();

    if (!cuda_allocator ||
        cuda_allocator->device_type() != DeviceType::CUDA) {
        throw std::invalid_argument(
            "to_cuda requires a CUDA allocator"
        );
    }

    // 已经在 CUDA 上时不做第二份显存拷贝。Tensor 的普通复制会共享
    // shared_ptr<Buffer>，因此返回值和当前 Tensor 指向同一块显存。
    if (device_type() == DeviceType::CUDA) {
        return *this;
    }

    Tensor result(dtype_, dims_, std::move(cuda_allocator));
    result.copy_from(*this, stream);
    return result;
}

Tensor Tensor::to_cpu(cudaStream_t stream) const {
    require_storage();

    if (device_type() == DeviceType::CPU) {
        return *this;
    }

    auto cpu_allocator = std::make_shared<CPUDeviceAllocator>();
    Tensor result(dtype_, dims_, std::move(cpu_allocator));
    result.copy_from(*this, stream);
    return result;
}

}
