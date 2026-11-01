#include "infer/cpu/cpu_block.h"
#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/pinned_memory_pool.h"
#include "infer/storage/storage_backend.h"
#include "slot_table.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace infer {
namespace storage_detail {
void validate_slot_bytes(std::size_t bytes) {
    if (bytes < 4096 || bytes % 4096 ||
        bytes != cpu_detail::capacities[cpu_detail::size_class(bytes)]) {
        throw std::invalid_argument("Storage slot must be a page-aligned CPU pool size class");
    }
}
StorageHandle next_storage_handle() {
    static std::atomic<StorageHandle> next{1};
    auto id = next.load(std::memory_order_relaxed);
    do {
        if (!id)
            throw std::overflow_error("Storage backend handle exhausted");
    } while (!next.compare_exchange_weak(id, id + 1, std::memory_order_relaxed));
    return id;
}
void validate_io(std::size_t capacity, std::size_t offset, const void* host, std::size_t bytes) {
    if (offset > capacity || bytes > capacity - offset)
        throw std::out_of_range("Storage I/O range");
    if (bytes && !host)
        throw std::invalid_argument("Storage I/O requires a host buffer");
}
} // namespace storage_detail

struct CPUStorageBackend::Impl {
    struct Allocation {
        void *data = nullptr;
        std::size_t capacity = 0;
    };
    std::size_t slot_bytes;
    bool pinned = false, variable = false;
    std::shared_ptr<DeviceAllocator> allocator;
    mutable std::mutex mutex;
    storage_detail::SlotTable<Allocation> allocations;
    ~Impl() {
        allocations.for_each([&](const auto& allocation) {
            try {
                allocator->release(allocation.data);
            } catch (...) {
                std::fputs("CPU storage cleanup failed\n", stderr);
            }
        });
    }
};
CPUStorageBackend::CPUStorageBackend(std::size_t bytes, std::shared_ptr<DeviceAllocator> allocator,
                                     HostMemoryKind kind, bool variable_slots) {
    storage_detail::validate_slot_bytes(bytes);
    if (kind != HostMemoryKind::PAGEABLE && kind != HostMemoryKind::PINNED)
        throw std::invalid_argument("Invalid host memory kind");
    if (!allocator) {
        CachingCPUAllocatorConfig config;
        config.max_reuse_waste = 0;
        if (kind == HostMemoryKind::PINNED)
            allocator = std::make_shared<CachingPinnedHostAllocator>(config);
        else
            allocator = std::make_shared<CachingCPUDeviceAllocator>(config);
    }
    const bool pinned = dynamic_cast<PinnedHostAllocator*>(allocator.get()) ||
                        dynamic_cast<CachingPinnedHostAllocator*>(allocator.get());
    if (allocator->device_type() != DeviceType::CPU || pinned != (kind == HostMemoryKind::PINNED)) {
        throw std::invalid_argument("CPU allocator does not match host memory kind");
    }
    impl_ = std::make_unique<Impl>();
    impl_->slot_bytes = bytes;
    impl_->pinned = pinned;
    impl_->variable = variable_slots;
    impl_->allocator = std::move(allocator);
}
CPUStorageBackend::~CPUStorageBackend() = default;
std::size_t CPUStorageBackend::slot_bytes() const noexcept {
    return impl_->slot_bytes;
}
bool CPUStorageBackend::host_is_pinned() const noexcept {
    return impl_->pinned;
}
StorageHandle CPUStorageBackend::reserve() { return reserve(impl_->slot_bytes); }
std::size_t CPUStorageBackend::allocation_bytes(std::size_t bytes) const {
    if (!bytes || bytes > impl_->slot_bytes)
        throw std::invalid_argument("Invalid CPU storage payload");
    return impl_->variable ? allocation_size(bytes) : impl_->slot_bytes;
}
std::size_t CPUStorageBackend::allocation_size(std::size_t bytes) {
    return cpu_detail::capacities[cpu_detail::size_class(
        std::max(bytes, CachingCPUDeviceAllocator::allocation_alignment))];
}
StorageHandle CPUStorageBackend::reserve(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto capacity = allocation_bytes(bytes);
    return impl_->allocations.create([&](std::size_t) {
        auto* data = impl_->allocator->allocate(capacity);
        if (!data)
            throw std::bad_alloc();
        return Impl::Allocation{data, capacity};
    });
}
void CPUStorageBackend::release(StorageHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto *ptr = impl_->allocations.at(handle).data;
    impl_->allocator->release(ptr); // 成功后才移除句柄，失败允许重试。
    impl_->allocations.erase(handle);
}
StorageMemoryView CPUStorageBackend::memory_view(StorageHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto allocation = impl_->allocations.at(handle);
    return {allocation.data, allocation.capacity, DeviceType::CPU, -1};
}
void CPUStorageBackend::read(StorageHandle handle, std::size_t offset, void* host,
                             std::size_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto allocation = impl_->allocations.at(handle);
    const auto *src = static_cast<const unsigned char *>(allocation.data);
    storage_detail::validate_io(allocation.capacity, offset, host, bytes);
    if (bytes)
        std::memmove(host, src + offset, bytes);
}
void CPUStorageBackend::write(StorageHandle handle, std::size_t offset, const void* host,
                              std::size_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto allocation = impl_->allocations.at(handle);
    auto *dst = static_cast<unsigned char *>(allocation.data);
    storage_detail::validate_io(allocation.capacity, offset, host, bytes);
    if (bytes)
        std::memmove(dst + offset, host, bytes);
}
void CPUStorageBackend::validate_stream(cudaStream_t stream) const {
    if (stream)
        throw std::invalid_argument("CPU storage lease does not accept a CUDA stream");
}
void CPUStorageBackend::wait(cudaStream_t stream) const {
    validate_stream(stream);
}
BackendPoolStats CPUStorageBackend::pool_stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (auto* pool = dynamic_cast<CachingPinnedHostAllocator*>(impl_->allocator.get())) {
        const auto s = pool->stats();
        return {s.active_bytes, s.cached_bytes, 0, s.reserved_bytes, true};
    }
    if (auto* pool = dynamic_cast<CachingCPUDeviceAllocator*>(impl_->allocator.get())) {
        const auto s = pool->stats();
        return {s.active_bytes, s.cached_bytes, 0, s.reserved_bytes, true};
    }
    return {0, 0, 0, 0, false};
}
} // namespace infer
