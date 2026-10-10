#include "mini_infer/core/mini_infer_memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_infer {

void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(message ? message : "CUDA operation") + ": " +
                                 cudaGetErrorString(error));
}

DeviceAllocator::DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}
DeviceType DeviceAllocator::device_type() const { return device_type_; }

void DeviceAllocator::copy(const void* src, void* dst, std::size_t bytes,
                           MemcpyKind kind, cudaStream_t stream, bool need_sync) const {
    if (bytes == 0) return;
    if (!src || !dst) throw std::invalid_argument("copy requires non-null storage");
    if (kind == MemcpyKind::CPU2CPU) {
        std::memmove(dst, src, bytes);
        return;  // A host copy never creates or synchronizes a CUDA context.
    }
    cudaMemcpyKind direction;
    switch (kind) {
        case MemcpyKind::CPU2CUDA: direction = cudaMemcpyHostToDevice; break;
        case MemcpyKind::CUDA2CPU: direction = cudaMemcpyDeviceToHost; break;
        case MemcpyKind::CUDA2CUDA: direction = cudaMemcpyDeviceToDevice; break;
        default: throw std::invalid_argument("unknown memory copy direction");
    }
    if (stream) check_cuda(cudaMemcpyAsync(dst, src, bytes, direction, stream), "cudaMemcpyAsync");
    else check_cuda(cudaMemcpy(dst, src, bytes, direction), "cudaMemcpy");
    if (need_sync) check_cuda(cudaDeviceSynchronize(), "wait for CUDA copy");
    else if (!stream) check_cuda(cudaStreamSynchronize(nullptr), "wait for default-stream copy");
}

void DeviceAllocator::memset_zero(void* ptr, std::size_t bytes, cudaStream_t stream) const {
    if (bytes == 0) return;
    if (!ptr) throw std::invalid_argument("memset requires non-null storage");
    if (device_type_ == DeviceType::CPU) { std::memset(ptr, 0, bytes); return; }
    if (device_type_ != DeviceType::CUDA) throw std::invalid_argument("unknown memory device");
    if (stream) check_cuda(cudaMemsetAsync(ptr, 0, bytes, stream), "cudaMemsetAsync");
    else {
        check_cuda(cudaMemset(ptr, 0, bytes), "cudaMemset");
        check_cuda(cudaStreamSynchronize(nullptr), "wait for default-stream memset");
    }
}

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::CPU) {}
void* CPUDeviceAllocator::allocate(std::size_t bytes) const {
    if (bytes == 0) return nullptr;
    if (void* ptr = std::malloc(bytes)) return ptr;
    throw std::bad_alloc();
}
void CPUDeviceAllocator::release(void* ptr) const { std::free(ptr); }

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::CUDA) {}
void* CUDADeviceAllocator::allocate(std::size_t bytes) const {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}
void CUDADeviceAllocator::release(void* ptr) const {
    if (ptr) check_cuda(cudaFree(ptr), "cudaFree");
}

Buffer::Buffer(std::size_t bytes, std::shared_ptr<DeviceAllocator> allocator)
    : byte_size_(bytes), use_external_(false), allocator_(std::move(allocator)) {
    if (!allocator_ || (allocator_->device_type() != DeviceType::CPU &&
                       allocator_->device_type() != DeviceType::CUDA))
        throw std::invalid_argument("Buffer requires a CPU/CUDA allocator");
    device_type_ = allocator_->device_type();
    ptr_ = allocator_->allocate(bytes);
    if (bytes && !ptr_) throw std::bad_alloc();
}
Buffer::Buffer(std::size_t bytes, void* ptr, DeviceType device)
    : byte_size_(bytes), ptr_(ptr), use_external_(true), device_type_(device) {
    if ((bytes && !ptr) || (device != DeviceType::CPU && device != DeviceType::CUDA))
        throw std::invalid_argument("invalid borrowed Buffer storage");
}
Buffer::~Buffer() { release_owned_memory(); }
Buffer::Buffer(Buffer&& other) noexcept { move_from(std::move(other)); }
Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) { release_owned_memory(); move_from(std::move(other)); }
    return *this;
}
void* Buffer::ptr() const { return ptr_; }
std::size_t Buffer::size() const { return byte_size_; }
DeviceType Buffer::device_type() const { return device_type_; }
bool Buffer::is_external() const { return use_external_; }

void Buffer::release_owned_memory() noexcept {
    if (!use_external_ && ptr_ && allocator_) {
        // Explicit release can report errors to callers; RAII cleanup must not
        // terminate a host process while another exception is unwinding.
        try { allocator_->release(ptr_); }
        catch (const std::exception& e) { std::fprintf(stderr, "MiniInfer cleanup: %s\n", e.what()); }
        catch (...) { std::fputs("MiniInfer cleanup: allocator release failed\n", stderr); }
        ptr_ = nullptr;
    }
}
void Buffer::move_from(Buffer&& other) {
    byte_size_ = std::exchange(other.byte_size_, 0);
    ptr_ = std::exchange(other.ptr_, nullptr);
    use_external_ = std::exchange(other.use_external_, true);
    device_type_ = std::exchange(other.device_type_, DeviceType::UNKNOWN);
    allocator_ = std::move(other.allocator_);
}
}  // namespace mini_infer
