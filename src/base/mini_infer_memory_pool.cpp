#include "mini_infer/base/mini_infer_memory_pool.h"
#include "infer/cuda/cuda_memory_pool.h"
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace mini_infer {
namespace {
int current_device() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get native CUDA device");
    return device;
}

class ScopedDevice {
  public:
    explicit ScopedDevice(int device) : previous_(current_device()), changed_(device != previous_) {
        if (changed_)
            check_cuda(cudaSetDevice(device), "select native CUDA device");
    }
    ~ScopedDevice() {
        if (changed_) {
            const auto error = cudaSetDevice(previous_);
            if (error != cudaSuccess)
                std::fprintf(stderr, "Infer cleanup: restore CUDA device: %s\n",
                             cudaGetErrorString(error));
        }
    }

  private:
    int previous_;
    bool changed_;
};
} // namespace

// Compatibility API only; all allocation, reuse and CUDA event handling live
// in infer::CachingCUDADeviceAllocator, shared with the vLLM integration.
class CachingCUDADeviceAllocator::Impl {
  public:
    explicit Impl(CachingCUDAAllocatorConfig config) {
        if (!config.small_block_threshold)
            throw std::invalid_argument("CUDA small_block_threshold must be positive");
        options.small_block_threshold = config.small_block_threshold;
        options.large_block_max_waste = config.large_block_max_waste;
        options.max_cached_bytes = config.max_cached_bytes;
    }
    infer::CachingCUDAAllocatorConfig options;
    std::mutex mutex;
    std::map<int, std::unique_ptr<infer::CachingCUDADeviceAllocator>> pools;
    std::unordered_map<void*, int> owners;
};

CachingCUDADeviceAllocator::CachingCUDADeviceAllocator(CachingCUDAAllocatorConfig config)
    : DeviceAllocator(DeviceType::CUDA), impl_(std::make_unique<Impl>(config)) {}

CachingCUDADeviceAllocator::~CachingCUDADeviceAllocator() = default;

void* CachingCUDADeviceAllocator::allocate(std::size_t bytes) const {
    if (!bytes)
        return nullptr;
    const int device = current_device();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& pool = impl_->pools[device];
    if (!pool)
        pool = std::make_unique<infer::CachingCUDADeviceAllocator>(impl_->options);
    void* ptr = pool->allocate(bytes);
    try {
        impl_->owners.emplace(ptr, device);
    } catch (...) {
        pool->release(ptr);
        throw;
    }
    return ptr;
}

void CachingCUDADeviceAllocator::release(void* ptr) const {
    if (!ptr)
        return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto owner = impl_->owners.find(ptr);
    if (owner == impl_->owners.end())
        throw std::invalid_argument("pointer is not active in this allocator");
    ScopedDevice selected(owner->second);
    impl_->pools.at(owner->second)->release(ptr);
    impl_->owners.erase(owner);
}

CUDAMemoryPoolStats CachingCUDADeviceAllocator::stats(int device) const {
    if (device == -1)
        device = current_device();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto pool = impl_->pools.find(device);
    if (pool == impl_->pools.end() || !pool->second)
        return {};
    const auto shared = pool->second->stats();
    CUDAMemoryPoolStats result;
    result.active_bytes = shared.active_capacity_bytes;
    result.cached_bytes = shared.ready_bytes + shared.pending_bytes;
    result.reserved_bytes = shared.reserved_bytes;
    result.active_blocks = shared.active_blocks;
    result.cached_blocks = shared.ready_blocks + shared.pending_blocks;
    result.cuda_malloc_calls = shared.upstream_allocations;
    result.cuda_free_calls = shared.upstream_releases;
    result.reuse_count = shared.reuse_count;
    return result;
}

void CachingCUDADeviceAllocator::trim(int device) const {
    if (device == -1)
        device = current_device();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto pool = impl_->pools.find(device);
    if (pool == impl_->pools.end() || !pool->second)
        return;
    ScopedDevice selected(device);
    pool->second->trim();
}

} // namespace mini_infer
