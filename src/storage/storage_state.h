#pragma once

#include "infer/storage/tiered_storage.h"
#include "slot_table.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <mutex>
#include <vector>

namespace infer {
class CachingPinnedHostAllocator;
}

namespace infer::storage_detail {

// Lease 与迁移共用同一适配器边界：取得地址并不意味着该视图可安全访问。
// 先验证再交出/复制 payload；调用方已经持有 Lease 或迁移操作保护。
inline StorageMemoryView checked_memory_view(StorageBackend& backend, StorageHandle handle,
                                             StorageLocation location, std::size_t capacity) {
    const auto view = backend.memory_view(handle);
    const auto device = location == StorageLocation::GPU ? DeviceType::CUDA : DeviceType::CPU;
    if (!view.data || view.capacity != backend.allocation_bytes(capacity) ||
        view.device != device) {
        throw std::logic_error("Storage backend returned an invalid memory view");
    }
    return view;
}

// 仅供 Storage 实现使用；公共调用者只需 tiered_storage.h。
// 一条 GPU Lease 对应一个槽，完成后可复用。等待失败时保留 stream 供 erase 重试。
struct ConsumerUse {
    cudaStream_t stream = nullptr;
    bool pending = false;
};

// 两槽生命周期属于 StorageState；busy 的槽不能重写或释放。
struct TransferSlots {
    explicit TransferSlots(std::size_t bytes, SSDStorageBackend* device = nullptr);
    ~TransferSlots();
    std::unique_ptr<CachingPinnedHostAllocator> pool;
    std::unique_ptr<std::array<std::shared_ptr<SSDDmaBuffer>, 2>> dma;
    std::array<void*, 2> data{};
    std::array<bool, 2> busy{};
    std::size_t bytes = 0;
    bool quarantined = false;
};
struct TransferResource {
    StorageBackend* backend = nullptr;
    StorageHandle handle = 0;
    std::size_t bytes = 0;
    bool pending = false;
};
// 提交前 auxiliary 持有目标，提交后持有旧源；不是第二个逻辑块。
// 位置切换只交换句柄，资源实际释放成功后才归还预算。
struct TransferOperation {
    TransferResource auxiliary;
    StorageBackend* gpu = nullptr;
    std::size_t staging = 2;
    bool shared_staging = false; // 批次拥有槽，正常单块清理不能提前归还。
    bool completion_unknown = false;
    bool source_quarantined = false;
    bool reuse_ssd_copy = false; // 同版本的文件槽只需验证，无需复制有效载荷。
    bool direct_host = false;    // GPU 与已锁页 CPU 槽直接 DMA，不占中转槽。
};

// 逻辑存储块档案，与 cpu_detail::Block / cuda_pool_detail::Block 的内存记录分开。
struct LogicalBlockRecord {
    BlockMeta meta;
    StorageHandle allocation = 0;      // 本层物理分配句柄，不是 BlockId。
    StorageBackend* backend = nullptr; // 由 StorageState 的 TierState 拥有。
    bool initialized = false;
    bool writing = false;
    bool cleanup_charged = false;
    std::vector<ConsumerUse> consumer_uses;
    std::unique_ptr<TransferOperation> transfer;
    TransferResource ssd_copy;
    std::uint64_t ssd_copy_version = 0; // 0 表示失效；槽可在下次写 SSD 时覆盖复用。
    bool operation_busy = false;
};
enum class StorageCharge { RESIDENT, MIGRATION, PENDING_CLEANUP, REPLICA };

// 只核算 Storage 持有和预留的后端槽，不分配物理内存或复制池统计；调用者持块表锁。
// reserve 增加总量；transition 只转分类；release 在实际释放成功后减少总量。
struct TierState {
    std::unique_ptr<StorageBackend> backend;
    StorageUsage usage;

    void check_capacity(std::size_t bytes) const {
        if (bytes > usage.limit_bytes || usage.reserved_bytes > usage.limit_bytes - bytes)
            throw std::bad_alloc();
    }
    void reserve(std::size_t bytes) {
        check_capacity(bytes);
        usage.reserved_bytes += bytes;
        usage.migration_bytes += bytes;
        usage.peak_reserved_bytes = std::max(usage.peak_reserved_bytes, usage.reserved_bytes);
    }
    void transition(StorageCharge from, StorageCharge to, std::size_t bytes) noexcept {
        auto& source = charged(from);
        assert(source >= bytes);
        source -= bytes;
        charged(to) += bytes;
    }
    void release(StorageCharge from, std::size_t bytes) noexcept {
        auto& source = charged(from);
        assert(source >= bytes && usage.reserved_bytes >= bytes);
        source -= bytes;
        usage.reserved_bytes -= bytes;
    }

  private:
    std::size_t& charged(StorageCharge charge) noexcept {
        switch (charge) {
        case StorageCharge::RESIDENT:
            return usage.resident_bytes;
        case StorageCharge::MIGRATION:
            return usage.migration_bytes;
        case StorageCharge::PENDING_CLEANUP:
            return usage.pending_cleanup_bytes;
        case StorageCharge::REPLICA:
            return usage.replica_bytes;
        }
        std::terminate(); // 仅接受实现内的分类，不对公共调用者暴露此枚举。
    }
};

// TieredStorage 和已发出的 Lease 共享状态，确保管理器先析构时 Lease 仍有效。
struct StorageState {
    ~StorageState();
    TierState& tier(StorageLocation location);
    void charge_cleanup(LogicalBlockRecord& record) noexcept; // 调用方持有 mutex。

    std::size_t available_staging(); // Caller holds mutex; reservation commits after preparation.
    void charge_pending(TransferResource& resource, StorageCharge from) noexcept {
        if (!resource.pending) {
            tier(resource.backend->location())
                .transition(from, StorageCharge::PENDING_CLEANUP, resource.bytes);
            resource.pending = true;
        }
    }

    // meta/usage/操作保护在锁内访问；锁外 I/O 依靠 Lease 或 operation_busy 保活。
    // 后端不得重入 Storage。等待 consumer stream 时不得持有本锁。
    std::mutex mutex;
    // Records keep stable addresses while the slot directory grows.
    SlotTable<std::unique_ptr<LogicalBlockRecord>> logical_blocks;
    TierState cpu_tier;
    TierState gpu_tier;
    TierState ssd_tier;
    std::unique_ptr<TransferSlots> staging;
    std::size_t block_bytes = 0;
    std::size_t access_clock = 0;
    std::size_t max_async_migrations = 2;
    bool retain_ssd_copy = false;
};

// 同步入口与工作队列共用同一套准备、提交、清理逻辑。
LogicalBlockRecord* prepare_transfer(StorageState& state, BlockId id, StorageLocation target);
// 批量预检独占块表锁；成功后所有非空记录保持 MOVING，直到执行或撤销。
std::vector<LogicalBlockRecord*> prepare_batch(StorageState& state, const std::vector<BlockId>& ids,
                                               StorageLocation target, std::size_t& staging);
// 调用者持块表锁，只用于尚未执行的预留；批次共享的 staging 由 executor 归还。
void cancel_prepared(StorageState& state, LogicalBlockRecord& record) noexcept;
MigrationResult execute_transfer(StorageState& state, LogicalBlockRecord& record,
                                 bool* quarantined = nullptr);
std::size_t execute_chunk_batch(StorageState& state, const std::vector<LogicalBlockRecord*>& records,
                                std::vector<MigrationResult>& results, std::size_t first,
                                bool& quarantined);
class MigrationExecutor {
  public:
    MigrationExecutor(std::shared_ptr<StorageState> state, std::size_t capacity);
    ~MigrationExecutor();
    std::future<MigrationResult> submit(BlockId id, StorageLocation target);
    std::future<std::vector<MigrationResult>> submit_batch(const std::vector<BlockId>& ids,
                                                           StorageLocation target);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 调用时已有操作保护、未持块表锁。失败仍持有资源，返回的异常用于 MigrationResult。
std::exception_ptr cleanup_transfer(StorageState& state, LogicalBlockRecord& record);
std::exception_ptr cleanup_ssd_copy(StorageState& state, LogicalBlockRecord& record);
} // namespace infer::storage_detail
