// 保留基础分配器、copy/memset 与 Buffer 的实现。
#include "infer/core/memory.h"
#if INFER_ENABLE_CUDA && INFER_MEMORY_MODE == 2
#include "infer/cuda/cuda_memory_pool.h"
#endif

#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace infer {

std::size_t io_allocation_size(std::size_t bytes) {
    if (!bytes)
        throw std::invalid_argument("Cannot allocate zero bytes.");
    if (bytes > std::numeric_limits<std::size_t>::max() - (io_alignment - 1))
        throw std::bad_alloc();
    return (bytes + io_alignment - 1) & ~(io_alignment - 1);
}

DeviceAllocator::DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

void DeviceAllocator::copy(const void* src, void* dst, std::size_t size, MemcpyKind kind,
                           cudaStream_t stream, bool need_async) const {
    if (src == nullptr || dst == nullptr || size == 0) {
        throw std::invalid_argument("Invalid arguments for copy: src, dst, or size is invalid.");
    }

    if (kind == MemcpyKind::CPU2CPU) {
        std::memmove(dst, src, size);
        return;
    }

#if INFER_ENABLE_CUDA
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case MemcpyKind::CPU2CUDA:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case MemcpyKind::CUDA2CPU:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case MemcpyKind::CUDA2CUDA:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        throw std::invalid_argument("Invalid MemcpyKind specified.");
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, size, cuda_kind, stream);
    check_cuda(err, "Failed to launch cudaMemcpyAsync");

    if (!need_async) {
        err = cudaStreamSynchronize(stream);
        check_cuda(err, "Failed to synchronize stream after copy");
    }
#else
    (void)stream;
    (void)need_async;
    if (kind != MemcpyKind::CPU2CUDA && kind != MemcpyKind::CUDA2CPU &&
        kind != MemcpyKind::CUDA2CUDA)
        throw std::invalid_argument("Invalid MemcpyKind specified.");
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "CUDA support is disabled");
#endif
}

void DeviceAllocator::memset_zero(void* ptr, std::size_t bytes, cudaStream_t stream) const {
    if (!ptr || bytes == 0) {
        throw std::invalid_argument(
            "Invalid arguments for memset_zero: ptr is null or bytes is zero.");
    }

    switch (device_type_) {
    case DeviceType::CPU:
        std::memset(ptr, 0, bytes);
        return;
    case DeviceType::CUDA:
#if INFER_ENABLE_CUDA
        check_cuda(cudaMemsetAsync(ptr, 0, bytes, stream), "Failed to launch cudaMemsetAsync");
        return;
#else
        (void)stream;
        throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                                "CUDA support is disabled");
#endif
    default:
        throw std::invalid_argument("Unknown device type for memset_zero.");
    }
}

// CPUDeviceAllocator
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::CPU) {}

void* CPUDeviceAllocator::allocate(std::size_t size, cudaStream_t /*stream*/) const {
    if (size == 0) {
        throw std::invalid_argument("Cannot allocate zero bytes.");
    }

    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, size >= io_alignment ? io_alignment : 64, size);
    if (ret != 0) {
        throw std::bad_alloc();
    }
    return ptr;
}

void* CPUDeviceAllocator::allocate_io(std::size_t size) const {
    return allocate(io_allocation_size(size));
}

void CPUDeviceAllocator::release(void* ptr, cudaStream_t /*stream*/) const {
    std::free(ptr);
}

// PinnedHostAllocator
PinnedHostAllocator::PinnedHostAllocator() : DeviceAllocator(DeviceType::CPU) {
#if !INFER_ENABLE_CUDA
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "Pinned memory requires CUDA support");
#endif
}

namespace {
void* allocate_pinned(std::size_t size, std::size_t alignment) {
    if (!size)
        throw std::invalid_argument("Cannot allocate zero bytes.");

#if INFER_ENABLE_CUDA
    const auto overhead = sizeof(void*) + alignment - 1;
    if (size > std::numeric_limits<std::size_t>::max() - overhead)
        throw std::bad_alloc();
    void* base = nullptr;
    const auto err = cudaMallocHost(&base, size + overhead);
    if (err == cudaErrorMemoryAllocation)
        throw std::bad_alloc();
    check_cuda(err, "Failed to allocate pinned host memory via cudaMallocHost");
    const auto address = reinterpret_cast<std::uintptr_t>(base) + overhead;
    auto* data = reinterpret_cast<void*>(address & ~(alignment - 1));
    // CUDA frees the original allocation, including any leading padding.
    std::memcpy(static_cast<char*>(data) - sizeof(base), &base, sizeof(base));
    return data;
#else
    (void)alignment;
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "Pinned memory requires CUDA support");
#endif
}
} // namespace

void* PinnedHostAllocator::allocate(std::size_t size, cudaStream_t /*stream*/) const {
    return allocate_pinned(size, size >= io_alignment ? io_alignment : 64);
}

void* PinnedHostAllocator::allocate_io(std::size_t size) const {
    return allocate_pinned(io_allocation_size(size), io_alignment);
}

void PinnedHostAllocator::release(void* ptr, cudaStream_t /*stream*/) const {
    if (!ptr)
        return;
#if INFER_ENABLE_CUDA
    // 与 cudaMallocHost 成对；调用者须已完成所有异步使用。
    void* base;
    std::memcpy(&base, static_cast<char*>(ptr) - sizeof(base), sizeof(base));
    check_cuda(cudaFreeHost(base), "Failed to free pinned host memory via cudaFreeHost");
#else
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "Pinned memory requires CUDA support");
#endif
}

// CUDADeviceAllocator
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::CUDA) {
#if !INFER_ENABLE_CUDA
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "CUDA support is disabled");
#endif
#if INFER_ENABLE_CUDA && INFER_MEMORY_MODE == 2
    pool_ = std::make_shared<CachingCUDADeviceAllocator>();
#endif
}
CUDADeviceAllocator::~CUDADeviceAllocator() = default;

void* CUDADeviceAllocator::allocate(std::size_t size, cudaStream_t stream) const {
    if (size == 0) {
        throw std::invalid_argument("Cannot allocate zero bytes.");
    }

#if INFER_ENABLE_CUDA && INFER_MEMORY_MODE == 2
    return pool_->allocate(size, stream);
#elif !INFER_ENABLE_CUDA
    (void)stream;
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "CUDA support is disabled");
#else
    void* ptr = nullptr;
#if INFER_MEMORY_MODE == 0
    // cudaMalloc 是传统分配 API，没有 stream 参数；这不改变后续拷贝/清零的流。
    (void)stream;
    cudaError_t err = cudaMalloc(&ptr, size);
    const char* msg = "Failed to allocate CUDA memory via cudaMalloc";
#elif INFER_MEMORY_MODE == 1
    cudaError_t err = cudaMallocAsync(&ptr, size, stream);
    const char* msg = "Failed to allocate CUDA memory via cudaMallocAsync";
#endif

    if (err == cudaErrorMemoryAllocation) {
        throw std::bad_alloc();
    }
    check_cuda(err, msg);

    return ptr;
#endif
}

void CUDADeviceAllocator::release(void* ptr, cudaStream_t stream) const {
    if (!ptr)
        return;
#if !INFER_ENABLE_CUDA
    (void)stream;
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "CUDA support is disabled");
#elif INFER_MEMORY_MODE == 0
    // cudaFree 也不接受 stream；调用者仍须保证所有使用已按 CUDA 规则完成。
    (void)stream;
    check_cuda(cudaFree(ptr), "Failed to free CUDA memory via cudaFree");
#elif INFER_MEMORY_MODE == 1
    check_cuda(cudaFreeAsync(ptr, stream), "Failed to free CUDA memory via cudaFreeAsync");
#elif INFER_MEMORY_MODE == 2
    pool_->release(ptr, stream);
#endif
}

// Buffer
Buffer::Buffer(std::size_t bytes, std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream)
    : size_(bytes), is_external_(false), allocator_(std::move(allocator)), stream_(stream) {
    if (!allocator_ || (allocator_->device_type() != DeviceType::CPU &&
                        allocator_->device_type() != DeviceType::CUDA))
        throw std::invalid_argument("Buffer requires a CPU/CUDA allocator");
    device_type_ = allocator_->device_type();
    ptr_ = allocator_->allocate(bytes, stream_);
    if (bytes && !ptr_)
        throw std::bad_alloc();
}

Buffer::Buffer(std::size_t size, void* ptr, DeviceType device_type)
    : size_(size), ptr_(ptr), is_external_(true), device_type_(device_type) {

    if (size > 0 && !ptr) {
        throw std::invalid_argument("Null external pointer");
    }
    if (device_type != DeviceType::CPU && device_type != DeviceType::CUDA) {
        throw std::invalid_argument("Unknown device");
    }
}

Buffer::~Buffer() {
    release_owned_buffer();
}

Buffer::Buffer(Buffer&& other) noexcept {
    move_from(std::move(other));
}
Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release_owned_buffer();
        move_from(std::move(other));
    }
    return *this;
}

void Buffer::release_owned_buffer() noexcept {
    if (is_external_ || !ptr_)
        return;

    try {
        allocator_->release(ptr_, stream_);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Buffer release failed: %s\n", e.what());
    } catch (...) {
        std::fputs("Buffer release failed\n", stderr);
    }

    ptr_ = nullptr;
}

void Buffer::move_from(Buffer&& other) noexcept {
    size_ = std::exchange(other.size_, 0);
    ptr_ = std::exchange(other.ptr_, nullptr);

    is_external_ = std::exchange(other.is_external_, false);

    device_type_ = std::exchange(other.device_type_, DeviceType::UNKNOWN);

    allocator_ = std::move(other.allocator_);
    stream_ = std::exchange(other.stream_, nullptr);
}

} // namespace infer
