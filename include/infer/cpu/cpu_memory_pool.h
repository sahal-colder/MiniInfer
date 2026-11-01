#pragma once

#include "infer/core/allocator.h"
#include "infer/cpu/cpu_memory_stats.h"
#include <memory>

namespace infer {

/**
 * @brief 统一的 Host（主机端 CPU）带缓存内存分配器引擎
 *
 * 核心设计契约与并发模型：
 * 1. 策略解耦：默认上游（Upstream）分配器为普通的 CPUDeviceAllocator，但支持注入自定义分配策略（如
 * CUDA Pinned Memory 锁页内存分配器）。
 * 2. 极速快路径（Fast Path）：对于 <= 4KiB 的小内存，若在当前线程的
 * TLS（线程本地缓存）中命中分配或本线程内归还，全程无需获取中央互斥锁（mutex），且不发生任何额外的元数据动态内存分配。
 * 3. 中央慢路径（Slow Path）：跨线程释放内存（A 线程分配、B 线程释放）、大内存（> 4KiB）操作、TLS
 * 缓存不足向中央池补货、TLS 缓存过多溢出退回、直接调用上游分配/释放，以及调用 trim()/stats()
 * 等维护接口时，仍需获取中央互斥锁（非全局 Lock-Free 设计）。
 * 4. 异步流（Stream）无关性：cudaStream_t 参数仅为满足基类 DeviceAllocator
 * 的统一接口签名而被忽略；主机内存生命周期不与 GPU 流绑定。调用方在调用 release()
 * 或覆写内存前，必须自行确保所有关联的 CPU 线程及 GPU 异步流操作已同步完成。
 */
class CachingCPUDeviceAllocator final : public DeviceAllocator {
  public:
    // 分配内存的基准对齐字节数（64B，适配 CPU Cache Line 与 AVX-512）
    static constexpr std::size_t allocation_alignment = 64;
    // 允许进入线程本地缓存（TLS）的最大单块内存阈值（4KiB），对应内部实现的 small_limit
    static constexpr std::size_t thread_cache_max_allocation = 4096;

    /**
     * @brief 默认构造函数：使用标准的系统 CPU 内存分配器作为上游水源
     * @param config 内存池配置项（如缓存上限、水位线等）
     */
    explicit CachingCPUDeviceAllocator(CachingCPUAllocatorConfig config = {});

    /**
     * @brief 依赖注入构造函数：指定自定义的上游物理内存分配器
     *
     * 上游分配器（upstream）必须满足以下契约：
     * - 对齐保证：allocate() 返回的物理地址必须满足 >= 64B 对齐，且 CPU 可直接访问。
     * - 强异常安全：release()
     * 必须同步释放内存；若抛出异常或失败，必须保留原内存块的所有权（不能处于半释放的损坏状态）。
     * - 防循环引用与死锁：upstream 由当前内存池通过 std::shared_ptr 持有，因此 upstream
     * 内部绝对不能反向持有当前池的强引用（防止内存泄漏），也严禁在 upstream
     * 的回调中重入调用当前内存池的接口（防止死锁）。
     *
     * @param upstream 底层实际执行系统内存申请/释放的上游分配器
     * @param config   内存池配置项
     */
    CachingCPUDeviceAllocator(std::shared_ptr<DeviceAllocator> upstream,
                              CachingCPUAllocatorConfig config = {});

    // 析构函数：销毁内存池实例（配合 weak_ptr 机制安全解绑所有活跃线程的 TLS）
    ~CachingCPUDeviceAllocator() override;

    // 禁用拷贝构造与拷贝赋值：内存池管理着独占的底层硬件资源与线程状态，不可复制
    CachingCPUDeviceAllocator(const CachingCPUDeviceAllocator&) = delete;
    CachingCPUDeviceAllocator& operator=(const CachingCPUDeviceAllocator&) = delete;

    /**
     * @brief 从内存池申请指定字节数的对齐内存
     * @param size   请求分配的字节数
     * @param stream 忽略该参数（仅用于兼容 DeviceAllocator 基类接口）
     * @return void* 对齐到 64 字节的内存首地址
     */
    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;

    /**
     * @brief 将内存归还给内存池（放入空闲缓存或退回上游）
     *
     * 安全性与状态契约：
     * - 空指针安全：传入 nullptr 时直接返回，无任何操作。
     * -
     * 防御性校验（带外元数据优势）：通过内部地址索引校验指针合法性，绝不解引用或读取用户指针（ptr）前后的隐藏头部。能够安全识别并拒绝外来野指针、块内部偏移指针或已被重复释放（Double-Free）的地址。
     * - 零分配保证：释放路径全程复用已有的 Block 元数据节点，不触发任何动态内存分配。
     * - 失败可重试：若直接向底层 upstream 释放物理内存时失败（抛出异常），该 Block 状态仍维持为
     * active，调用方后续可安全地重试释放。
     *
     * @param ptr    待释放的内存首地址
     * @param stream 忽略该参数（调用前需由外部保证 GPU 流已完成对该主机内存的读写）
     */
    void release(void* ptr, cudaStream_t stream = nullptr) const override;

    /**
     * @brief 获取当前内存池的实时统计信息（活跃字节数、缓存字节数、块计数等）
     */
    CPUMemoryPoolStats stats() const;

    /**
     * @brief 强制瘦身：同步将中央池及所有线程 TLS 中的空闲缓存块（cached）全部释放回上游分配器
     *
     * 事务性与异常安全保证：
     * - 仅清理处于空闲状态（active == false）的缓存块，正在被用户使用的活跃块（active ==
     * true）不受影响。
     * - 逐块提交账本：每成功向上游释放一个物理块，才更新一次内部统计账本（Usage）。
     * -
     * 异常不回滚但保持一致性：若某次上游释放中途抛出异常，此前已成功释放给操作系统的块不会（也无法）回滚；但未释放的剩余缓存块状态依然完整有效，内存池可继续正常使用或再次调用
     * trim()。
     */
    void trim() const;

  private:
    // Pimpl 内部实现类前向声明（隐藏 FreeBins、ThreadCache、中央锁等实现细节，降低编译依赖）
    class Impl;

    /**
     * @brief 指向内部核心实现的共享指针
     *
     * 为什么在 Pimpl 中使用 std::shared_ptr 而不是 std::unique_ptr？
     * - 因为线程本地存储（thread_local /
     * TLS）的生命周期与“线程”绑定，而与“内存池对象”的生命周期异步。
     * - 内存池对象可能比线程先销毁。通过让池持有 shared_ptr<Impl>，而各个线程的 TLS 仅持有
     * weak_ptr<Impl>：
     *   1. TLS 不会强行延长内存池的寿命（池析构时 Impl 即可正常释放）。
     *   2. 当内存池已被销毁、而线程才退出或尝试回调时，TLS 可通过 weak_ptr::lock()
     * 安全检测到池已失效，避免悬挂指针（Dangling Pointer / 裸 this 回调）引发崩溃。
     */
    std::shared_ptr<Impl> impl_;
};

} // namespace infer