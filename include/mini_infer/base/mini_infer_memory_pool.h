#pragma once

#include "mini_infer/base/mini_infer_memory.h"

#include <cstddef>
#include <memory>

namespace mini_infer {

// ============================================================
// 第 9 课：缓存式 CUDA 显存池
// ============================================================
//
// 普通 CUDADeviceAllocator 的一生是：
//
//   allocate -> cudaMalloc
//   release  -> cudaFree
//
// 缓存分配器则把 release 改成“归还到池中”：
//
//   allocate -> 先找空闲块 -> 找不到才 cudaMalloc
//   release  -> busy = false -> 暂时保留显存
//
// 此兼容接口现在委托给 infer::CachingCUDADeviceAllocator，
// 与 vLLM 使用同一份池实现；每张设备各有独立池。
// 通用 DeviceAllocator、CPU/CUDA 基础分配器和 Buffer 仍在
// mini_infer_memory.h 中，方便分别阅读两层职责。
//
// 底层支持块切分/合并与流完成事件；它仍不负责 KV 逻辑页或请求调度。
// ============================================================

struct CachingCUDAAllocatorConfig {
    // 课程按 1 MiB 将显存块分为 small / large 两组。
    std::size_t small_block_threshold = 1ULL * 1024 * 1024;

    // large 块只在浪费不超过 1 MiB 时复用，避免拿特别大的块
    // 去满足很小的请求。
    std::size_t large_block_max_waste = 1ULL * 1024 * 1024;

    // 空闲缓存超过 1 GiB 时，自动把当前设备上的空闲块 cudaFree。
    // 调用 trim() 可以随时主动清理。
    std::size_t max_cached_bytes = 1ULL * 1024 * 1024 * 1024;
};

// 这些统计值按 CUDA device 分开计算。
// active/cached 统计的是显存块容量，而不是某个 Tensor 请求的逻辑字节数。
struct CUDAMemoryPoolStats {
    std::size_t active_bytes = 0;
    std::size_t cached_bytes = 0;
    std::size_t reserved_bytes = 0;
    std::size_t active_blocks = 0;
    std::size_t cached_blocks = 0;
    std::size_t cuda_malloc_calls = 0;
    std::size_t cuda_free_calls = 0;
    std::size_t reuse_count = 0;
};

class CachingCUDADeviceAllocator final : public DeviceAllocator {
public:
    // 真实字段和缓存表只存在于 .cpp 中，便于课程阅读者先看接口。
    class Impl;

    explicit CachingCUDADeviceAllocator(
        CachingCUDAAllocatorConfig config = {}
    );
    ~CachingCUDADeviceAllocator() override;

    CachingCUDADeviceAllocator(const CachingCUDADeviceAllocator&) = delete;
    CachingCUDADeviceAllocator& operator=(
        const CachingCUDADeviceAllocator&
    ) = delete;

    void* allocate(std::size_t byte_size) const override;
    void release(void* ptr) const override;

    // device == -1 表示当前 CUDA device。
    CUDAMemoryPoolStats stats(int device = -1) const;

    // 主动 cudaFree 指定设备上的所有空闲块；busy 块绝不会被清理。
    void trim(int device = -1) const;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace mini_infer

