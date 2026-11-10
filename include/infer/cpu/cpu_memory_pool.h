#pragma once

#include "infer/core/allocator.h"
#include "infer/cpu/cpu_memory_stats.h"
#include <memory>

namespace infer {

// CPU/Pinned 共用的主机缓存引擎。<=4 KiB 可走 TLS 快路径；补货、跨线程归还、
// 大块和维护操作仍需中央锁。TLS 使用原子 gate，不是整个池无锁。
// stream 参数不管理主机内存生命周期；调用方须先确认所有 CPU/GPU 使用已完成。
class CachingCPUDeviceAllocator final : public DeviceAllocator {
  public:
    static constexpr std::size_t allocation_alignment = 64;
    static constexpr std::size_t thread_cache_max_allocation = 4096;

    explicit CachingCPUDeviceAllocator(CachingCPUAllocatorConfig config = {});
    // 默认上游为 CPUDeviceAllocator。自定义上游须返回独占、CPU 可访问、至少
    // 64 B 对齐的地址；release 同步完成，失败保留所有权，且不得重入或反向持有本池。
    CachingCPUDeviceAllocator(std::shared_ptr<DeviceAllocator> upstream,
                              CachingCPUAllocatorConfig config = {});
    ~CachingCPUDeviceAllocator() override;

    CachingCPUDeviceAllocator(const CachingCPUDeviceAllocator&) = delete;
    CachingCPUDeviceAllocator& operator=(const CachingCPUDeviceAllocator&) = delete;

    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;
    // 共用预算和缓存；4 KiB I/O 对齐仅支持内置 CPU/Pinned 上游。
    void* allocate_io(std::size_t size) const;
    // nullptr 无操作；按地址索引拒绝外来、内部及重复释放的指针，不读取用户前缀。
    // 复用已有元数据，不动态分配；上游释放失败时仍为 active，可重试。
    void release(void* ptr, cudaStream_t stream = nullptr) const override;

    CPUMemoryPoolStats stats() const;
    // 清理中央及 TLS 空闲块，不动 active 块。逐块释放成功后才改账；中途失败
    // 保留未释放的缓存和所有权，允许继续使用或重试。
    void trim() const;

  private:
    class Impl;
    // TLS Entry 只持 weak_ptr：线程退出可检测池已销毁，不延长池寿命或回调悬空 Impl。
    std::shared_ptr<Impl> impl_;
};

} // namespace infer
