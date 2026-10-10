#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>

namespace mini_infer {

// ------------------------------------------------------------
// 数据所在设备
// ------------------------------------------------------------
enum class DeviceType {
    UNKNOWN = 0,
    CPU,
    CUDA
};

// ------------------------------------------------------------
// 数据拷贝方向
// ------------------------------------------------------------
enum class MemcpyKind {
    CPU2CPU = 0,
    CPU2CUDA,
    CUDA2CPU,
    CUDA2CUDA
};

// CUDA API 错误检查。
// 调用失败时抛出 std::runtime_error，由应用层决定如何处理。
void check_cuda(cudaError_t error, const char* message);

// ============================================================
// DeviceAllocator
// ============================================================
//
// 这是设备资源管理的统一抽象。
//
// 上层代码只调用：
//
//   allocator->allocate(...)
//   allocator->release(...)
//   allocator->copy(...)
//
// 而不用知道底层到底是：
//
//   malloc / free
//
// 还是：
//
//   cudaMalloc / cudaFree
//
// 以后 Tensor、KV Cache 都可以复用这一层。
// ============================================================
class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type);
    virtual ~DeviceAllocator() = default;

    DeviceType device_type() const;

    // 申请 byte_size 字节资源。
    virtual void* allocate(std::size_t byte_size) const = 0;

    // 释放资源。
    virtual void release(void* ptr) const = 0;

    // 统一数据拷贝接口。
    //
    // stream == nullptr:
    //   CUDA 拷贝走 cudaMemcpy，并等待默认 stream 完成；返回后
    //   目标数据可交给 nonblocking stream 使用。
    //
    // stream != nullptr:
    //   CUDA 拷贝走 cudaMemcpyAsync
    //
    // need_sync == true:
    //   CUDA 拷贝完成后调用 cudaDeviceSynchronize()；CPU 拷贝始终同步完成。
    virtual void copy(
        const void* src_ptr,
        void* dst_ptr,
        std::size_t byte_size,
        MemcpyKind kind,
        cudaStream_t stream = nullptr,
        bool need_sync = false
    ) const;

    // 将一块内存 / 显存清零。
    virtual void memset_zero(
        void* ptr,
        std::size_t byte_size,
        cudaStream_t stream = nullptr
    ) const;

private:
    DeviceType device_type_;
};

// ============================================================
// CPUDeviceAllocator
// ============================================================
//
// CPU:
//   allocate -> malloc
//   release  -> free
// ============================================================
class CPUDeviceAllocator final : public DeviceAllocator {
public:
    CPUDeviceAllocator();

    void* allocate(std::size_t byte_size) const override;
    void release(void* ptr) const override;
};

// ============================================================
// CUDADeviceAllocator
// ============================================================
//
// CUDA:
//   allocate -> cudaMalloc
//   release  -> cudaFree
// ============================================================
class CUDADeviceAllocator final : public DeviceAllocator {
public:
    CUDADeviceAllocator();

    void* allocate(std::size_t byte_size) const override;
    void release(void* ptr) const override;
};

// ============================================================
// Buffer
// ============================================================
//
// Buffer 解决的核心问题不是“哪里有一块指针”，而是：
//
//   谁拥有这块资源？
//   谁负责释放？
//
// 情况 A：Buffer 自己申请
//
//   Buffer --owns--> memory
//
//   析构时自动释放。
//
// 情况 B：外部传进来一个指针
//
//   external owner --owns--> memory
//                          ^
//                          |
//                       Buffer
//
//   Buffer 只借用，不负责释放。
//
// 这就是 RAII + ownership。
// ============================================================
class Buffer {
public:
    // Buffer 自己申请资源。
    Buffer(
        std::size_t byte_size,
        std::shared_ptr<DeviceAllocator> allocator
    );

    // Buffer 借用外部资源，不负责释放。
    Buffer(
        std::size_t byte_size,
        void* external_ptr,
        DeviceType device_type
    );

    ~Buffer();

    // 禁止 copy：
    // 避免两个 Buffer 同时认为自己拥有同一块资源，导致 double free。
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // 允许 move：
    // 所有权从 old Buffer 转移给 new Buffer。
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void* ptr() const;
    std::size_t size() const;
    DeviceType device_type() const;
    bool is_external() const;

private:
    void release_owned_memory() noexcept;
    void move_from(Buffer&& other);

private:
    std::size_t byte_size_ = 0;
    void* ptr_ = nullptr;

    // true  = 借用外部资源，不释放
    // false = Buffer 自己拥有，析构时释放
    bool use_external_ = false;

    DeviceType device_type_ = DeviceType::UNKNOWN;
    std::shared_ptr<DeviceAllocator> allocator_;
};

// ============================================================
// 缓存式 CUDA 显存池
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
// 按设备缓存完整分配块，支持统计和主动回收；不做块切分或分页映射。
// 异步操作完成后才能归还内存，调用者负责建立 stream 依赖。
// ============================================================

struct CachingCUDAAllocatorConfig {
    // 按 1 MiB 将显存块分为 small / large 两组。
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
