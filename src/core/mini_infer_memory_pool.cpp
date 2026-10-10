#include "mini_infer/core/mini_infer_memory.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_infer {

class CachingCUDADeviceAllocator::Impl {
public:
    struct Block {
        void* ptr = nullptr;
        std::size_t byte_size = 0;
        bool busy = false;
    };

    struct DevicePool {
        // 两张表直接对应课程中的 small / large 缓存。
        std::vector<Block> small_blocks;
        std::vector<Block> large_blocks;

        std::size_t active_bytes = 0;
        std::size_t cached_bytes = 0;
        std::size_t cuda_malloc_calls = 0;
        std::size_t cuda_free_calls = 0;
        std::size_t reuse_count = 0;
    };

    explicit Impl(CachingCUDAAllocatorConfig input_config)
        : config(std::move(input_config)) {}

    CachingCUDAAllocatorConfig config;

    // CUDA device 是线程局部状态，但缓存表本身可能被多个线程访问，
    // 所以元数据仍然需要一把锁保护。
    mutable std::mutex mutex;
    mutable std::unordered_map<int, DevicePool> pools;
};

namespace {

int current_cuda_device() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");
    return device;
}

// 在 vector 中做 best-fit：选择“能放下请求的最小空闲块”。
// 对 large block 还额外限制最大浪费量。
CachingCUDADeviceAllocator::Impl::Block* find_best_fit(
    std::vector<CachingCUDADeviceAllocator::Impl::Block>& blocks,
    std::size_t requested_bytes,
    bool limit_waste,
    std::size_t max_waste
) {
    CachingCUDADeviceAllocator::Impl::Block* best = nullptr;

    for (auto& block : blocks) {
        if (block.busy || block.byte_size < requested_bytes) {
            continue;
        }

        const std::size_t waste = block.byte_size - requested_bytes;
        if (limit_waste && waste > max_waste) {
            continue;
        }

        if (best == nullptr || block.byte_size < best->byte_size) {
            best = &block;
        }
    }

    return best;
}

void free_idle_blocks(
    std::vector<CachingCUDADeviceAllocator::Impl::Block>& blocks,
    CachingCUDADeviceAllocator::Impl::DevicePool& pool
) {
    for (auto it = blocks.begin(); it != blocks.end();) {
        if (it->busy) { ++it; continue; }
        check_cuda(cudaFree(it->ptr), "memory-pool cudaFree failed");
        pool.cached_bytes -= it->byte_size;
        ++pool.cuda_free_calls;
        it = blocks.erase(it);  // Commit each successful free before the next can fail.
    }
}

class ScopedCUDADevice {
public:
    explicit ScopedCUDADevice(int device) : original_(current_cuda_device()), changed_(device != original_) {
        if (changed_) check_cuda(cudaSetDevice(device), "memory-pool cudaSetDevice failed");
    }
    ~ScopedCUDADevice() {
        if (changed_) {
            const auto error = cudaSetDevice(original_);
            if (error != cudaSuccess)
                std::fprintf(stderr, "MiniInfer cleanup: restoring CUDA device failed: %s\n", cudaGetErrorString(error));
        }
    }
private:
    int original_;
    bool changed_;
};

void trim_device_locked(CachingCUDADeviceAllocator::Impl::DevicePool& pool, int device) {
    ScopedCUDADevice selected(device);
    free_idle_blocks(pool.small_blocks, pool);
    free_idle_blocks(pool.large_blocks, pool);
}

}  // namespace

CachingCUDADeviceAllocator::CachingCUDADeviceAllocator(
    CachingCUDAAllocatorConfig config
)
    : DeviceAllocator(DeviceType::CUDA),
      impl_(std::make_unique<Impl>(std::move(config))) {
    if (impl_->config.small_block_threshold == 0) {
        throw std::invalid_argument("CachingCUDADeviceAllocator: threshold must be > 0");
    }
}

CachingCUDADeviceAllocator::~CachingCUDADeviceAllocator() {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (auto& entry : impl_->pools) {
            try {
                ScopedCUDADevice selected(entry.first);
                for (auto* blocks : {&entry.second.small_blocks, &entry.second.large_blocks}) {
                    for (const auto& block : *blocks) {
                        const auto error = cudaFree(block.ptr);
                        if (error != cudaSuccess)
                            std::fprintf(stderr, "MiniInfer cleanup: cudaFree failed: %s\n", cudaGetErrorString(error));
                    }
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "MiniInfer cleanup: %s\n", e.what());
            }
        }
    } catch (...) {
        std::fputs("MiniInfer cleanup: memory pool cleanup failed\n", stderr);
    }
}

void* CachingCUDADeviceAllocator::allocate(std::size_t byte_size) const {
    if (byte_size == 0) {
        return nullptr;
    }

    const int device = current_cuda_device();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl::DevicePool& pool = impl_->pools[device];

    const bool is_large =
        byte_size > impl_->config.small_block_threshold;
    std::vector<Impl::Block>& blocks =
        is_large ? pool.large_blocks : pool.small_blocks;

    Impl::Block* reusable = find_best_fit(
        blocks,
        byte_size,
        is_large,
        impl_->config.large_block_max_waste
    );

    if (reusable != nullptr) {
        reusable->busy = true;
        pool.cached_bytes -= reusable->byte_size;
        pool.active_bytes += reusable->byte_size;
        ++pool.reuse_count;

        return reusable->ptr;
    }

    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, byte_size), "memory-pool cudaMalloc failed");

    try {
        blocks.push_back(Impl::Block{ptr, byte_size, true});
    } catch (...) {
        (void)cudaFree(ptr);
        throw;
    }
    pool.active_bytes += byte_size;
    ++pool.cuda_malloc_calls;

    return ptr;
}

void CachingCUDADeviceAllocator::release(void* ptr) const {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);

    for (auto& entry : impl_->pools) {
        const int device = entry.first;
        Impl::DevicePool& pool = entry.second;

        auto mark_idle = [&](std::vector<Impl::Block>& blocks) {
            for (Impl::Block& block : blocks) {
                if (block.ptr != ptr) {
                    continue;
                }

                if (!block.busy) {
                    throw std::logic_error("CachingCUDADeviceAllocator: double release");
                }

                block.busy = false;
                pool.active_bytes -= block.byte_size;
                pool.cached_bytes += block.byte_size;

                return true;
            }
            return false;
        };

        if (!mark_idle(pool.small_blocks) && !mark_idle(pool.large_blocks)) {
            continue;
        }

        // 超过阈值才清理；并且只清理空闲块。
        if (pool.cached_bytes > impl_->config.max_cached_bytes) {
            trim_device_locked(pool, device);
        }
        return;
    }

    throw std::invalid_argument("pointer was not allocated by this allocator");
}

CUDAMemoryPoolStats CachingCUDADeviceAllocator::stats(int device) const {
    if (device == -1) {
        device = current_cuda_device();
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->pools.find(device);
    if (it == impl_->pools.end()) {
        return {};
    }

    const Impl::DevicePool& pool = it->second;
    CUDAMemoryPoolStats result;
    result.active_bytes = pool.active_bytes;
    result.cached_bytes = pool.cached_bytes;
    result.reserved_bytes = pool.active_bytes + pool.cached_bytes;
    result.cuda_malloc_calls = pool.cuda_malloc_calls;
    result.cuda_free_calls = pool.cuda_free_calls;
    result.reuse_count = pool.reuse_count;

    const auto count_blocks = [](const std::vector<Impl::Block>& blocks,
                                 bool busy) {
        return static_cast<std::size_t>(std::count_if(
            blocks.begin(),
            blocks.end(),
            [busy](const Impl::Block& block) {
                return block.busy == busy;
            }
        ));
    };

    result.active_blocks =
        count_blocks(pool.small_blocks, true) +
        count_blocks(pool.large_blocks, true);
    result.cached_blocks =
        count_blocks(pool.small_blocks, false) +
        count_blocks(pool.large_blocks, false);
    return result;
}

void CachingCUDADeviceAllocator::trim(int device) const {
    if (device == -1) {
        device = current_cuda_device();
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->pools.find(device);
    if (it == impl_->pools.end()) {
        return;
    }

    trim_device_locked(it->second, device);
}

}  // namespace mini_infer
