#include "mini_infer/base/mini_infer_memory.h"
#include <cstdio>
#include <new>
#include <stdexcept>
#include <utility>

namespace mini_infer {
Buffer::Buffer(std::size_t byte_size, std::shared_ptr<DeviceAllocator> allocator)
    : byte_size_(byte_size), allocator_(std::move(allocator)) {

    if (!allocator_ || (allocator_->device_type() != DeviceType::CPU &&
                        allocator_->device_type() != DeviceType::CUDA)) {
        throw std::invalid_argument("Buffer requires a CPU/CUDA allocator");
    }

    device_type_ = allocator_->device_type();
    ptr_ = allocator_->allocate(byte_size_);
    if (byte_size_ > 0 && ptr_ == nullptr) {
        throw std::bad_alloc();
    }
}

Buffer::Buffer(std::size_t byte_size, void* external_ptr, DeviceType device_type)
    : byte_size_(byte_size), ptr_(external_ptr), device_type_(device_type) {

    if ((byte_size_ > 0 && ptr_ == nullptr) ||
        (device_type_ != DeviceType::CPU && device_type_ != DeviceType::CUDA)) {
        throw std::invalid_argument("invalid borrowed Buffer storage");
    }
}

Buffer::~Buffer() {
    release_owned_memory();
}

Buffer::Buffer(Buffer&& other) noexcept {
    move_from(std::move(other));
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release_owned_memory();
        move_from(std::move(other));
    }

    return *this;
}

void Buffer::release_owned_memory() noexcept {
    if (ptr_ != nullptr && allocator_) {
        // 析构和 noexcept 移动赋值期间，释放失败不能终止宿主进程。
        try {
            allocator_->release(ptr_);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "MiniInfer cleanup: %s\n", e.what());
        } catch (...) {
            std::fputs("MiniInfer cleanup: allocator release failed\n", stderr);
        }
        ptr_ = nullptr;
    }
}

void Buffer::move_from(Buffer&& other) {
    byte_size_ = other.byte_size_;
    ptr_ = other.ptr_;
    device_type_ = other.device_type_;
    allocator_ = std::move(other.allocator_);

    // 所有权已经移动给当前对象。
    // 原对象必须失去 ptr，否则它析构时还会再次释放。
    other.byte_size_ = 0;
    other.ptr_ = nullptr;
    other.device_type_ = DeviceType::UNKNOWN;
}

} // namespace mini_infer
