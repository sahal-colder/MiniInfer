#include "infer/cuda/cuda_memory_pool.h"
#include "storage_state.h"
#include "scope_exit.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace infer {
namespace {
// 每次操作临时选择后端设备，退出时恢复调用线程的设备。
class DeviceScope {
  public:
    explicit DeviceScope(int device) {
        check_cuda(cudaGetDevice(&previous_), "storage current device");
        if (previous_ != device)
            check_cuda(cudaSetDevice(device), "storage select device");
    }
    ~DeviceScope() { cudaSetDevice(previous_); }

  private:
    int previous_ = 0;
};
void require_pinned(const void* host, std::size_t bytes) {
    if (!bytes)
        return;
    cudaPointerAttributes attr{};
    check_cuda(cudaPointerGetAttributes(&attr, host), "storage host attributes");
    if (attr.type != cudaMemoryTypeHost)
        throw std::invalid_argument("GPU storage I/O requires pinned host memory");
}
} // namespace
struct GPUStorageBackend::Impl {
    struct Allocation {
        std::mutex mutex;
        void* data = nullptr;
        bool released = false;
        bool completion_unknown = false;
    };
    struct CopyEvent {
        cudaEvent_t event = nullptr;
        bool busy = false, failed = false;
    };
    std::size_t slot_bytes = 0;
    int device = -1;
    cudaStream_t transfer_stream = nullptr;
    std::unique_ptr<CachingCUDADeviceAllocator> pool;
    mutable std::mutex mutex;
    storage_detail::SlotTable<std::shared_ptr<Allocation>> allocations;
    std::vector<CopyEvent> events;
    ~Impl() {
        // 所有分配已释放且搬运流已完成，池可以安全回收 pending 缓存。
        pool.reset();
        for (auto& event : events)
            cudaEventDestroy(event.event);
        if (transfer_stream)
            cudaStreamDestroy(transfer_stream);
    }
    std::shared_ptr<Allocation> lookup(StorageHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex);
        return allocations.at(handle);
    }
    static void check_live(const Allocation& allocation) {
        if (allocation.released)
            throw std::out_of_range("Released GPU handle");
    }
    // 每次复制使用自己的完成事件。等待时只保护当前分配，不持全局表锁，
    // 因而其他块可以提交 DMA，另一个工作线程也能继续做 SSD I/O。
    void copy(StorageHandle handle, std::size_t offset, void* host, std::size_t bytes,
              bool to_gpu) {
        const auto allocation = lookup(handle);
        std::lock_guard<std::mutex> use(allocation->mutex);
        check_live(*allocation);
        storage_detail::validate_io(slot_bytes, offset, host, bytes);
        require_pinned(host, bytes);
        if (!bytes)
            return;
        auto* device_ptr = static_cast<unsigned char*>(allocation->data) + offset;
        copy_with_event(allocation->completion_unknown, [&] {
            check_cuda(cudaMemcpyAsync(to_gpu ? device_ptr : host, to_gpu ? host : device_ptr,
                                       bytes,
                                       to_gpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                       transfer_stream), "storage DMA");
        });
    }

    template <class Submit>
    void copy_with_event(bool& completion_unknown, Submit submit) {
        std::size_t index = 0;
        cudaEvent_t completed = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            while (index < events.size() && (events[index].busy || events[index].failed))
                ++index;
            if (index == events.size())
                events.emplace_back();
            if (!events[index].event)
                check_cuda(cudaEventCreateWithFlags(&events[index].event, cudaEventDisableTiming),
                           "storage copy event");
            completed = events[index].event;
            events[index].busy = true;
        }
        completion_unknown = true;
        storage_detail::ScopeExit complete([&] {
            std::lock_guard<std::mutex> lock(mutex);
            events[index].busy = false;
            events[index].failed = completion_unknown;
        });
        submit();
        check_cuda(cudaEventRecord(completed, transfer_stream), "storage copy record");
        check_cuda(cudaEventSynchronize(completed), "storage copy completion");
        completion_unknown = false;
    }
};
std::exception_ptr storage_detail::GPUMigrationBatch::copy(
    GPUStorageBackend& backend, const Copy* copies, std::size_t count, bool to_gpu,
    std::size_t& submitted, bool& completion_unknown) {
    auto& state = *backend.impl_;
    DeviceScope scope(state.device);
    std::exception_ptr submission_error;
    state.copy_with_event(completion_unknown, [&] {
        try {
            for (; submitted < count; ++submitted) {
                const auto& copy = copies[submitted];
                require_pinned(copy.host, copy.bytes);
                check_cuda(cudaMemcpyAsync(to_gpu ? copy.device : copy.host,
                                           to_gpu ? copy.host : copy.device, copy.bytes,
                                           to_gpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                           state.transfer_stream), "storage batch DMA");
            }
        } catch (...) {
            // Fence the accepted DMA even when a later submission fails.
            submission_error = std::current_exception();
        }
    });
    return submission_error;
}

GPUStorageBackend::GPUStorageBackend(std::size_t bytes, std::size_t budget, int device) {
    storage_detail::validate_slot_bytes(bytes);
    if (budget < bytes || device < 0)
        throw std::invalid_argument("Invalid GPU storage budget or device");
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "storage device count");
    if (device >= count)
        throw std::invalid_argument("GPU storage device does not exist");
    DeviceScope scope(device);
    auto state = std::make_unique<Impl>();
    state->slot_bytes = bytes;
    state->device = device;
    CachingCUDAAllocatorConfig config;
    config.max_reserved_bytes = budget;
    config.max_cached_bytes = budget;
    config.large_block_max_waste = 0;
    config.min_split_bytes = 0;
    state->pool = std::make_unique<CachingCUDADeviceAllocator>(config);
    check_cuda(cudaStreamCreateWithFlags(&state->transfer_stream, cudaStreamNonBlocking),
               "storage transfer stream");
    state->pool->register_stream(state->transfer_stream);
    impl_ = std::move(state);
}
GPUStorageBackend::~GPUStorageBackend() {
    // StorageState releases allocations only after their consumer streams finish.
    // Remaining handles have unresolved ownership; a device-wide wait would also
    // stall unrelated compute and cannot replace that lifecycle contract.
    if (impl_->allocations.size()) {
        std::fputs("GPU storage has unresolved allocations; resources quarantined until process exit\n",
                   stderr);
        (void)impl_.release();
        return;
    }
    try {
        DeviceScope scope(impl_->device);
        check_cuda(cudaStreamSynchronize(impl_->transfer_stream), "storage shutdown transfer wait");
        impl_.reset();
    } catch (...) {
        std::fputs("GPU storage shutdown could not confirm completion; resources quarantined until "
                   "process exit\n",
                   stderr);
        (void)impl_.release();
    }
}
std::size_t GPUStorageBackend::slot_bytes() const noexcept {
    return impl_->slot_bytes;
}
StorageHandle GPUStorageBackend::reserve() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    DeviceScope scope(impl_->device);
    return impl_->allocations.create([&](std::size_t) {
        auto allocation = std::make_shared<Impl::Allocation>();
        allocation->data = impl_->pool->allocate(impl_->slot_bytes, impl_->transfer_stream);
        return allocation;
    });
}
void GPUStorageBackend::release(StorageHandle handle) {
    const auto allocation = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(allocation->mutex);
    Impl::check_live(*allocation);
    DeviceScope scope(impl_->device);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // 上层已完成 consumer stream；本分配的同步复制也已退出记录锁。
    // 独立使用后端时也要保留失败后的等待保障，不能只依赖上层管理器。
    if (allocation->completion_unknown)
        check_cuda(cudaStreamSynchronize(impl_->transfer_stream),
                   "storage failed copy release wait");
    impl_->pool->release(allocation->data, impl_->transfer_stream);
    impl_->allocations.erase(handle);
    allocation->released = true;
}
StorageMemoryView GPUStorageBackend::memory_view(StorageHandle handle) const {
    const auto allocation = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(allocation->mutex);
    Impl::check_live(*allocation);
    return {allocation->data, impl_->slot_bytes, DeviceType::CUDA, impl_->device};
}
void GPUStorageBackend::read(StorageHandle handle, std::size_t offset, void* host,
                             std::size_t bytes) {
    DeviceScope scope(impl_->device);
    impl_->copy(handle, offset, host, bytes, false);
}
void GPUStorageBackend::write(StorageHandle handle, std::size_t offset, const void* host,
                              std::size_t bytes) {
    DeviceScope scope(impl_->device);
    impl_->copy(handle, offset, const_cast<void*>(host), bytes, true);
}
void GPUStorageBackend::validate_stream(cudaStream_t stream) const {
    if (!stream || stream == cudaStreamPerThread || stream == cudaStreamLegacy) {
        throw std::invalid_argument("GPU storage lease requires an explicit consumer stream");
    }
    DeviceScope scope(impl_->device);
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream, &capture), "storage consumer capture status");
    if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("Storage leases cannot be acquired during graph capture");
#if CUDART_VERSION >= 12080
    int device = -1;
    check_cuda(cudaStreamGetDevice(stream, &device), "storage stream device");
    if (device != impl_->device)
        throw std::invalid_argument("Storage consumer stream belongs to another device");
#else
    // 旧 Toolkit 没有 stream 设备查询，借同设备 event record 验证设备归属。
    cudaEvent_t event{};
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
               "storage validation event");
    const auto status = cudaEventRecord(event, stream);
    cudaEventDestroy(event);
    check_cuda(status, "storage stream device validation");
#endif
}
void GPUStorageBackend::wait(cudaStream_t stream) const {
    DeviceScope scope(impl_->device);
    check_cuda(cudaStreamSynchronize(stream), "storage consumer completion");
}
void GPUStorageBackend::wait_transfer() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    DeviceScope scope(impl_->device);
    check_cuda(cudaStreamSynchronize(impl_->transfer_stream), "storage transfer recovery");
    for (auto& event : impl_->events)
        event.failed = false;
}
BackendPoolStats GPUStorageBackend::pool_stats() const {
    DeviceScope scope(impl_->device);
    const auto s = impl_->pool->stats();
    return {s.active_capacity_bytes, s.ready_bytes, s.pending_bytes, s.reserved_bytes, true};
}
} // namespace infer
