#pragma once

#include "infer/cpu/cpu_memory_pool.h"

namespace infer {

// 与 CUDA 显存池并列使用：这是 CPU 地址空间的锁页缓存，用于 KV Swap / H2D / D2H。
// 上游固定为 cudaMallocHost/cudaFreeHost，缓存引擎和 TLS 规则与普通 CPU 池相同。
// pinned 是异步 DMA 的内存条件，并不保证所有 cudaMemcpyAsync 调用都瞬时返回。
// 必须在最后一次 DMA 完成后才 release 或重写。即使传入相同 stream，也不能用
// GPU FIFO 推断 CPU 写入已安全；推荐每个流水线槽持有 Buffer 和完成事件。
class CachingPinnedHostAllocator final : public DeviceAllocator {
  public:
    explicit CachingPinnedHostAllocator(CachingCPUAllocatorConfig config = {});
    ~CachingPinnedHostAllocator() override;
    CachingPinnedHostAllocator(const CachingPinnedHostAllocator&) = delete;
    CachingPinnedHostAllocator& operator=(const CachingPinnedHostAllocator&) = delete;
    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;
    void release(void* ptr, cudaStream_t stream = nullptr) const override;
    CPUMemoryPoolStats stats() const;
    void trim() const;

  private:
    CachingCPUDeviceAllocator pool_;
};

} // namespace infer
