// 声明 DeviceAllocator 和已有 CPU/CUDA/Pinned 分配器。
#pragma once

#include "infer/config.h"
#include "infer/utils/cuda_check.h"
#include <cstddef>
#include <memory>

// 0: SYNC，1: ASYNC，2: POOL（默认，缓存池）。
// 只控制 CUDA 分配/释放，不改变 copy 和 memset_zero 的行为。
#ifndef INFER_MEMORY_MODE
#define INFER_MEMORY_MODE 2
#endif

#if INFER_MEMORY_MODE != 0 && INFER_MEMORY_MODE != 1 && INFER_MEMORY_MODE != 2
#error "INFER_MEMORY_MODE must be 0 (SYNC), 1 (ASYNC), or 2 (POOL)."
#elif INFER_ENABLE_CUDA && INFER_MEMORY_MODE == 2 && CUDART_VERSION < 11020
#error "POOL mode requires CUDA Toolkit 11.2 or newer."
#elif INFER_ENABLE_CUDA && INFER_MEMORY_MODE == 1 && CUDART_VERSION < 11020
#error "ASYNC memory mode requires CUDA Toolkit 11.2 or newer."
#endif

namespace infer {

enum class DeviceType { UNKNOWN, CPU, CUDA };
enum class MemcpyKind { CPU2CPU, CPU2CUDA, CUDA2CPU, CUDA2CUDA };

class DeviceAllocator {
  public:
    explicit DeviceAllocator(DeviceType device_type);
    virtual ~DeviceAllocator() = default;

    virtual void* allocate(std::size_t size, cudaStream_t stream = nullptr) const = 0;
    virtual void release(void* ptr, cudaStream_t stream = nullptr) const = 0;

    // GPU 拷贝的 need_async=true 只省略主动同步；普通主机内存仍可能触发阻塞。
    // 异步传输使用 pinned host memory，调用者负责完成事件和缓冲区生命周期。
    void copy(const void* src, void* dst, std::size_t size, MemcpyKind kind, cudaStream_t stream,
              bool need_async) const;
    void memset_zero(void* ptr, std::size_t size, cudaStream_t stream) const;

    DeviceType device_type() const noexcept { return device_type_; }

  private:
    DeviceType device_type_;
};

class CPUDeviceAllocator final : public DeviceAllocator {
  public:
    CPUDeviceAllocator();

    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;
    void release(void* ptr, cudaStream_t stream = nullptr) const override;
};

// 锁页主机内存：仍属于 CPU，可直接由 CPU 读写；适合复用为异步传输缓冲区。
// 分配/释放使用 cudaMallocHost/cudaFreeHost，不受 INFER_MEMORY_MODE 控制。
// stream 参数仅保持统一接口，不会使分配/释放按该 stream 排序。
// 调用者必须在释放前等待所有 GPU 使用完成；本类不自动同步。
class PinnedHostAllocator final : public DeviceAllocator {
  public:
    PinnedHostAllocator();

    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;
    void release(void* ptr, cudaStream_t stream = nullptr) const override;
};

// POOL 模式的此包装适合固定流集合。需要注销/重建流或管理 Graph 时，
// 请直接使用 CachingCUDADeviceAllocator，以显式维护池中的流身份和地址生命周期。
class CUDADeviceAllocator final : public DeviceAllocator {
  public:
    CUDADeviceAllocator();
    ~CUDADeviceAllocator() override;

    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;
    void release(void* ptr, cudaStream_t stream = nullptr) const override;
#if INFER_ENABLE_CUDA && INFER_MEMORY_MODE == 2
  private:
    std::shared_ptr<DeviceAllocator> pool_;
#endif
};

} // namespace infer
