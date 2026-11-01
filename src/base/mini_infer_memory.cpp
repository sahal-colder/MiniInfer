#include "mini_infer/base/mini_infer_memory.h"
#include "infer/cpu/cpu_memory_pool.h"
#include <cstring>
#include <stdexcept>
#include <utility>

namespace mini_infer {

DeviceAllocator::DeviceAllocator(std::shared_ptr<infer::DeviceAllocator> allocator)
    : device_type_(allocator ? allocator->device_type() : DeviceType::UNKNOWN),
      allocator_(std::move(allocator)) {
    if (!allocator_ || (device_type_ != DeviceType::CPU && device_type_ != DeviceType::CUDA))
        throw std::invalid_argument("native allocator requires a CPU/CUDA allocator");
}

void* DeviceAllocator::allocate(std::size_t bytes) const {
    if (!bytes)
        return nullptr;
    void* ptr = allocator_->allocate(bytes);
    // Shared CUDA pools allocate on the default stream; native callers may
    // immediately submit work on another stream without an allocation event.
    if (device_type_ == DeviceType::CUDA)
        check_cuda(cudaStreamSynchronize(nullptr), "complete native CUDA allocation");
    return ptr;
}

void DeviceAllocator::release(void* ptr) const {
    if (ptr)
        allocator_->release(ptr);
}

void DeviceAllocator::copy(const void* src, void* dst, std::size_t bytes, MemcpyKind kind,
                           cudaStream_t stream, bool need_sync) const {
    if (!bytes)
        return;
    infer::CPUDeviceAllocator copier;
    copier.copy(src, dst, bytes, kind, stream, stream != nullptr);
    if (need_sync && kind != MemcpyKind::CPU2CPU)
        check_cuda(cudaDeviceSynchronize(), "wait for native CUDA copy");
}

void DeviceAllocator::memset_zero(void* ptr, std::size_t bytes, cudaStream_t stream) const {
    if (!bytes)
        return;
    if (!ptr)
        throw std::invalid_argument("memset requires non-null storage");
    if (device_type_ == DeviceType::CPU) {
        std::memset(ptr, 0, bytes);
        return;
    }
    if (device_type_ != DeviceType::CUDA)
        throw std::invalid_argument("unknown memory device");
    check_cuda(cudaMemsetAsync(ptr, 0, bytes, stream), "native CUDA memset");
    if (!stream)
        check_cuda(cudaStreamSynchronize(nullptr), "wait for default-stream memset");
}

CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(std::make_shared<infer::CachingCPUDeviceAllocator>()) {}

CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(std::make_shared<infer::CUDADeviceAllocator>()) {}

} // namespace mini_infer
