#pragma once

#include "infer/core/allocator.h"
#include "mini_infer/base/cuda_error.h"
#include <cstddef>
#include <memory>

namespace mini_infer {

using DeviceType = infer::DeviceType;
using MemcpyKind = infer::MemcpyKind;

// Native allocations are ready to use on any stream when allocate returns.
// Non-default-stream copies remain asynchronous unless need_sync is requested.
class DeviceAllocator {
  public:
    explicit DeviceAllocator(std::shared_ptr<infer::DeviceAllocator> allocator);
    virtual ~DeviceAllocator() = default;
    DeviceType device_type() const { return device_type_; }
    virtual void* allocate(std::size_t byte_size) const;
    virtual void release(void* ptr) const;
    virtual void copy(const void* src, void* dst, std::size_t bytes, MemcpyKind kind,
                      cudaStream_t stream = nullptr, bool need_sync = false) const;
    virtual void memset_zero(void* ptr, std::size_t bytes, cudaStream_t stream = nullptr) const;

  protected:
    explicit DeviceAllocator(DeviceType device) : device_type_(device) {}

  private:
    DeviceType device_type_;
    std::shared_ptr<infer::DeviceAllocator> allocator_;
};

class CPUDeviceAllocator final : public DeviceAllocator {
  public:
    CPUDeviceAllocator();
};

class CUDADeviceAllocator final : public DeviceAllocator {
  public:
    CUDADeviceAllocator();
};

// Owns an allocation or borrows external storage. Tensor views share this owner.
class Buffer {
  public:
    Buffer(std::size_t bytes, std::shared_ptr<DeviceAllocator> allocator);
    Buffer(std::size_t bytes, void* ptr, DeviceType device);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    void* ptr() const { return ptr_; }
    std::size_t size() const { return byte_size_; }
    DeviceType device_type() const { return device_type_; }
    bool is_external() const { return !allocator_; }

  private:
    void release_owned_memory() noexcept;
    void move_from(Buffer&& other);
    std::size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    DeviceType device_type_ = DeviceType::UNKNOWN;
    std::shared_ptr<DeviceAllocator> allocator_;
};

} // namespace mini_infer

#include "mini_infer/base/mini_infer_memory_pool.h"
