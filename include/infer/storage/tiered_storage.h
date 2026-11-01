#pragma once

#include "infer/storage/storage_backend.h"
#include "infer/storage/storage_block.h"
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace infer {
namespace storage_detail {
class MigrationExecutor;
}

// 逻辑槽预算：包含驻留、申请中及待清理容量；0 禁用该层，默认不初始化 CUDA。
// 默认固定尺寸，variable_cpu_slots 可按载荷尺寸桶分配；active + cached 不超过 cpu_bytes。
// 注入后端/allocator 的额外缓存另计；stats() 看逻辑槽，pool_stats() 看池占用。
// GPU 后端另设池 reserved 硬上限。
struct StorageBudget {
    std::size_t cpu_bytes = 32ULL * 1024 * 1024 * 1024;
    std::size_t gpu_bytes = 0;
    std::size_t ssd_bytes = 0; // 包含记录头及设备要求的对齐填充，默认禁用。
};
struct StorageConfig {
    std::size_t block_bytes = 1024 * 1024; // 固定物理槽：页对齐且为现有 CPU 池桶容量，默认 1 MiB。
    StorageBudget budget;
    int gpu_device = 0;
    std::string ssd_directory = "/tmp";
    std::size_t max_async_migrations = 2; // 运行中和排队任务之和有界；一批计一个任务。
    bool retain_ssd_copy = false;         // 迁回内存时保留文件槽；同版本再次下放可省去重写。
    FileIOConfig ssd_io;
    HostMemoryKind cpu_memory =
        HostMemoryKind::PAGEABLE; // PINNED 复用已有锁页池，按 cpu_bytes 限额。
    SSDVerification ssd_verification = SSDVerification::READ_BACK;
    SpdkBdevConfig ssd_spdk;
    bool variable_cpu_slots =
        false; // CPU charges size-class capacity; SSD address slots stay fixed.
    bool mapped_cpu_pool = false; // Optional contiguous batches for the full record size class.
};
struct StorageUsage {
    std::size_t limit_bytes = 0;
    std::size_t resident_bytes = 0;
    std::size_t migration_bytes = 0; // 申请中预留，以及迁移目标/待回收旧源。
    std::size_t pending_cleanup_bytes = 0;
    std::size_t replica_bytes = 0;  // 保留的 SSD 槽；WRITE 后失效的槽也继续计费。
    std::size_t reserved_bytes = 0; // 上面四项之和。
    std::size_t peak_reserved_bytes = 0;
};

class StorageBusy : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};
// 参数/预算/Busy 在搬运前抛异常；开始搬运后的错误通过结果返回，避免误判提交状态。
struct MigrationResult {
    bool committed = false;
    bool cleanup_pending = false;
    std::exception_ptr error;
};
struct StagingUsage {
    std::size_t reserved_bytes = 0; // 首次需要中转时，从已有 Pinned Pool 持有两个固定槽。
    std::size_t busy_slots = 0;     // 包括完成未知、仍被隔离的槽。
};

// 管理逻辑存储块，实际分配由 CPU/GPU StorageBackend 委托已有池完成。
// 支持同步/有界异步迁移和可选 SSD 副本；自动淘汰与 Remote 留待后续阶段。
// 块表操作加锁，GPU Lease 等待不持有这把锁。析构不能与外部成员调用并发。
class TieredStorage {
  public:
    TieredStorage();
    explicit TieredStorage(StorageConfig config, std::unique_ptr<StorageBackend> cpu_backend = {},
                           std::unique_ptr<StorageBackend> gpu_backend = {},
                           std::unique_ptr<StorageBackend> ssd_backend = {});
    // 兼容已有 CPU allocator 注入。其缓存等额外开销需通过池自身统计查看。
    // allocator 不能为空、不得重入本管理器，release 失败须保留原分配。
    explicit TieredStorage(std::shared_ptr<DeviceAllocator> cpu_allocator);
    ~TieredStorage();
    TieredStorage(const TieredStorage&) = delete;
    TieredStorage& operator=(const TieredStorage&) = delete;
    TieredStorage(TieredStorage&&) = delete;
    TieredStorage& operator=(TieredStorage&&) = delete;

    // 在 CPU/GPU 申请完整物理槽，有效载荷为 size。SSD 通过迁移接收已提交内容。
    // 0<size<=block_bytes；内容尚未初始化。
    BlockId create(std::size_t size, StorageLocation location = StorageLocation::CPU,
                   int priority = 0);
    // READ 仅允许已提交内容；WRITE 独占，调用者需初始化整个有效载荷再 finish。
    // CPU 不传 stream，GPU 必须传显式 stream，跨流依赖由调用者归入该 stream。
    BlockLease acquire(BlockId id, AccessMode mode = AccessMode::READ,
                       cudaStream_t stream = nullptr);
    // Check the requested location while taking the lease. A concurrent migration
    // returns StorageBusy before exposing an address or invalidating WRITE data.
    BlockLease acquire_at(BlockId id, StorageLocation location, AccessMode mode = AccessMode::READ,
                          cudaStream_t stream = nullptr);
    // Complete CPU replacement may change the valid length within the same
    // physical size class. Allocation and content identity stay authoritative here.
    BlockLease acquire_cpu_write(BlockId id, std::size_t bytes);
    // CPU/SSD 完整只读导出，不迁移位置。调用期间保护源，返回前完成复制和 SSD 校验。
    // dst 须为至少 bytes 长的有效主机缓冲；bytes 必须等于逻辑有效载荷。
    // 失败时 dst 可能已部分写入，不得消费；GPU 块仍使用已有 Lease/迁移接口。
    void read_to_host(BlockId id, void* dst, std::size_t bytes);
    // 一次保护整批源；失败时所有目标均不可消费，返回/抛出前全部 I/O 已完成。
    void read_to_host_many(const std::vector<BlockId>& ids, const std::vector<void*>& destinations,
                           std::size_t bytes);
    void read_to_host_many(const std::vector<BlockId> &ids, const std::vector<void *> &destinations,
                           const std::vector<std::size_t> &sizes);
    BlockMeta metadata(BlockId id) const; // 值拷贝，不修改访问顺序。
    StorageUsage stats(StorageLocation location) const;
    BackendPoolStats pool_stats(StorageLocation location) const;
    BackendIOStats io_stats(StorageLocation location) const;
    // 纯迁移不改变 ID/Version。无 Lease 才能开始；返回前完成拷贝和位置提交。
    MigrationResult migrate(BlockId id, StorageLocation target);
    // 提交时检查并预留预算/使用权；get() 返回实际完成结果。满载抛 StorageBusy。
    // 丢弃 future 不取消任务；管理器析构会排空并 join 工作线程。
    std::future<MigrationResult> migrate_async(BlockId id, StorageLocation target);
    // 一次提交 1..64 个不同 ID，共用一个任务/future，结果顺序与输入相同。
    // 提交时全部保护源并预留目标预算；参数/Busy/预算失败则整批撤销预留。
    // 执行时逐块提交，不是整批原子事务；普通失败不阻止后续块。
    // GPU 批次独占一个 Pinned 槽并顺序复用；DMA 完成未知时停止本批，
    // 撤销尚未执行块的预留，故障块的资源仍由 retry_cleanup/erase 回收。
    std::future<std::vector<MigrationResult>> migrate_batch_async(const std::vector<BlockId>& ids,
                                                                  StorageLocation target);
    // 回收迁移或副本释放遗留资源，不删除当前主副本；失败继续保留预算。
    bool retry_cleanup(BlockId id);
    // 显式释放保留的 SSD 副本（有效或失效）；当前内存内容不受影响。
    // Busy/Lease 时 false；释放失败抛异常并保留计费，可 retry_cleanup。
    bool drop_ssd_copy(BlockId id);
    StagingUsage staging_stats() const;

    // 有 Lease 或其他回收操作时 false；未知 ID 抛 out_of_range。
    // 释放/完成等待失败保留 ERROR 和全部容量，可重试 erase。
    bool erase(BlockId id);

  private:
    BlockLease acquire_impl(BlockId id, AccessMode mode, cudaStream_t stream,
                            StorageLocation expected, std::size_t bytes = 0);
    friend class KVOffloadCache;
    std::shared_ptr<storage_detail::StorageState> state_;
    std::mutex executor_mutex_;
    std::unique_ptr<storage_detail::MigrationExecutor> executor_;
};
} // namespace infer
