#include "infer/cpu/pinned_memory_pool.h"
#include "storage_state.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

namespace infer::storage_detail {

namespace {
LogicalBlockRecord* prepare_locked(StorageState& state, BlockId id, StorageLocation target,
                                   bool batch) {
    auto& record = *state.logical_blocks.at(id);
    if (record.meta.lease_count || record.operation_busy || record.transfer ||
        record.ssd_copy.pending || record.meta.state != BlockState::RESIDENT) {
        throw StorageBusy("Storage block has a lease, operation or pending cleanup");
    }
    if (!record.initialized)
        throw std::logic_error("Cannot migrate unpublished content");
    auto& target_tier = state.tier(target);
    if (!target_tier.backend)
        throw std::invalid_argument("Migration target is disabled");
    if (target == record.meta.location)
        return nullptr;
    auto op = std::make_unique<TransferOperation>();
    const bool retained_target = target == StorageLocation::SSD && record.ssd_copy.backend;
    op->reuse_ssd_copy = retained_target && record.ssd_copy_version == record.meta.version;
    // 先检查双槽，让第三个并发请求稳定返回 Busy，不因目标预算也满而模糊原因。
    const bool uses_gpu = !op->reuse_ssd_copy && (target == StorageLocation::GPU ||
                                                  record.meta.location == StorageLocation::GPU);
    op->direct_host =
        uses_gpu &&
        ((target == StorageLocation::CPU && target_tier.backend->host_is_pinned()) ||
         (record.meta.location == StorageLocation::CPU && record.backend->host_is_pinned()));
    std::size_t staging = 2;
    if (uses_gpu) {
        if (!batch && !op->direct_host) {
            if (!state.staging)
                state.staging = std::make_unique<TransferSlots>(state.block_bytes,
                    dynamic_cast<SSDStorageBackend*>(state.ssd_tier.backend.get()));
            staging = 0;
            while (staging < 2 && state.staging->busy[staging])
                ++staging;
            if (staging == 2)
                throw StorageBusy("Both pinned transfer slots are busy");
        }
        op->gpu = target == StorageLocation::GPU ? target_tier.backend.get() : record.backend;
    }
    const auto bytes = target_tier.backend->allocation_bytes(record.meta.size);
    // 完成所有可能失败的准备后才计费/发布 MOVING；保留副本已占总预算。
    if (!retained_target)
        target_tier.reserve(bytes);
    op->auxiliary = retained_target ? record.ssd_copy
                                    : TransferResource{target_tier.backend.get(), 0, bytes, false};
    op->staging = staging;
    record.transfer = std::move(op);
    record.operation_busy = true;
    record.meta.state = BlockState::MOVING;
    if (staging != 2)
        state.staging->busy[staging] = true;
    if (retained_target) {
        target_tier.transition(StorageCharge::REPLICA, StorageCharge::MIGRATION, bytes);
        record.ssd_copy = {};
        record.ssd_copy_version = 0;
    }
    return &record;
}

} // namespace

LogicalBlockRecord* prepare_transfer(StorageState& state, BlockId id, StorageLocation target) {
    std::lock_guard<std::mutex> lock(state.mutex);
    return prepare_locked(state, id, target, false);
}

// 调用者持有块表锁；只撤销尚未开始复制的预留。
void cancel_prepared(StorageState& state, LogicalBlockRecord& record) noexcept {
    auto& op = *record.transfer;
    auto& resource = op.auxiliary;
    auto& tier = state.tier(resource.backend->location());
    if (resource.handle) {
        // 预留时借用了已有 SSD 副本；尚未写入，按原版本原样交还。
        record.ssd_copy = resource;
        record.ssd_copy_version = op.reuse_ssd_copy ? record.meta.version : 0;
        tier.transition(StorageCharge::MIGRATION, StorageCharge::REPLICA, resource.bytes);
    } else
        tier.release(StorageCharge::MIGRATION, resource.bytes);
    record.transfer.reset();
    record.operation_busy = false;
    record.meta.state = BlockState::RESIDENT;
}

std::vector<LogicalBlockRecord*> prepare_batch(StorageState& state, const std::vector<BlockId>& ids,
                                               StorageLocation target, std::size_t& staging) {
    for (auto it = ids.begin(); it != ids.end(); ++it)
        if (std::find(ids.begin(), it, *it) != it)
            throw std::invalid_argument("Duplicate block in migration batch");
    std::vector<LogicalBlockRecord*> records;
    records.reserve(ids.size()); // 状态变更前完成结果索引分配。
    std::lock_guard<std::mutex> lock(state.mutex);
    bool uses_gpu = false;
    try {
        for (auto id : ids) {
            auto* record = prepare_locked(state, id, target, true);
            records.push_back(record);
            if (record && record->transfer->gpu && !record->transfer->direct_host)
                uses_gpu = true;
        }
        staging = 2;
        if (uses_gpu) {
            if (!state.staging)
                state.staging = std::make_unique<TransferSlots>(state.block_bytes,
                    dynamic_cast<SSDStorageBackend*>(state.ssd_tier.backend.get()));
            staging = 0;
            while (staging < 2 && state.staging->busy[staging])
                ++staging;
            if (staging == 2)
                throw StorageBusy("Both pinned transfer slots are busy");
            for (auto* record : records)
                if (record && record->transfer->gpu && !record->transfer->direct_host) {
                    record->transfer->staging = staging;
                    record->transfer->shared_staging = true;
                }
            state.staging->busy[staging] = true;
        }
    } catch (...) {
        for (auto* record : records)
            if (record)
                cancel_prepared(state, *record);
        throw;
    }
    return records;
}
namespace {
void copy_block(StorageState& state, LogicalBlockRecord& record, StorageBackend& target,
                StorageHandle handle) {
    auto& op = *record.transfer;
    const auto bytes = record.meta.size;
    if (!op.gpu) {
        // CPU 的源或目标本来就已计费；普通 CPU↔SSD 不再额外分配一个整块缓冲。
        if (target.location() == StorageLocation::SSD) {
            record.backend->validate(record.allocation, record.meta);
            const auto view = checked_memory_view(*record.backend, record.allocation,
                                                  StorageLocation::CPU, record.meta.size);
            target.write_published(handle, record.meta, view.data);
            return;
        } else {
            const auto view =
                checked_memory_view(target, handle, StorageLocation::CPU, record.meta.size);
            record.backend->read_validated(record.allocation, record.meta, view.data);
        }
    } else if (op.direct_host) {
        // 两端都由迁移事务保活，完成前不释放/复用 CPU 槽；省掉 pinned↔pageable memcpy。
        if (record.meta.location == StorageLocation::GPU) {
            record.backend->validate(record.allocation, record.meta);
            const auto view =
                checked_memory_view(target, handle, StorageLocation::CPU, record.meta.size);
            op.completion_unknown = true;
            record.backend->read(record.allocation, 0, view.data, bytes);
        } else {
            record.backend->validate(record.allocation, record.meta);
            const auto view = checked_memory_view(*record.backend, record.allocation,
                                                  StorageLocation::CPU, record.meta.size);
            op.completion_unknown = true;
            target.write(handle, 0, view.data, bytes);
        }
        op.completion_unknown = false;
    } else {
        auto* pinned = state.staging->data.at(op.staging);
        if (record.meta.location == StorageLocation::GPU) {
            record.backend->validate(record.allocation, record.meta);
            op.completion_unknown = true;
            record.backend->read(record.allocation, 0, pinned, bytes);
            op.completion_unknown = false;
            // Pinned 是 CPU 可寻址内存，文件后端可以直接使用它。
            if (target.location() == StorageLocation::SSD) {
                target.write_published(handle, record.meta, pinned);
                return;
            }
            target.write(handle, 0, pinned, bytes);
        } else {
            record.backend->read_validated(record.allocation, record.meta, pinned);
            op.completion_unknown = true;
            target.write(handle, 0, pinned, bytes);
        }
        op.completion_unknown = false;
    }
    // SSD 在这里发布身份/CRC；默认另做回读。发布失败仍处于 MOVING，不切换目标。
    target.publish(handle, record.meta);
}

} // namespace

MigrationResult execute_transfer(StorageState& state, LogicalBlockRecord& record,
                                 bool* quarantined) {
    const auto target = record.transfer->auxiliary.backend->location();
    const auto bytes = record.transfer->auxiliary.bytes;
    auto& target_tier = state.tier(target);
    std::unique_lock<std::mutex> lock(state.mutex, std::defer_lock);
    MigrationResult result;
    auto& auxiliary = record.transfer->auxiliary;
    try {
        if (!auxiliary.handle)
            auxiliary.handle = auxiliary.backend->reserve(record.meta.size);
        if (record.transfer->reuse_ssd_copy)
            auxiliary.backend->validate(auxiliary.handle, record.meta);
        else
            copy_block(state, record, *auxiliary.backend, auxiliary.handle);
        lock.lock();
        // 唯一提交点：新副本已完整可读。之后清理失败也不能再宣称回滚。
        auto& source_tier = state.tier(record.meta.location);
        const auto source_bytes = record.backend->allocation_bytes(record.meta.size);
        const bool retain_source =
            state.retain_ssd_copy && record.meta.location == StorageLocation::SSD;
        target_tier.transition(StorageCharge::MIGRATION, StorageCharge::RESIDENT, bytes);
        source_tier.transition(StorageCharge::RESIDENT, StorageCharge::MIGRATION, source_bytes);
        std::swap(record.allocation, auxiliary.handle);
        std::swap(record.backend, auxiliary.backend);
        auxiliary.bytes = source_bytes;
        record.meta.location = target;
        if (retain_source) {
            // 这是额外的有效副本，而非第二个逻辑块；各层仍统一按物理槽计费。
            record.ssd_copy = auxiliary;
            record.ssd_copy_version = record.meta.version;
            auxiliary = {};
            source_tier.transition(StorageCharge::MIGRATION, StorageCharge::REPLICA, source_bytes);
        }
        record.consumer_uses.clear();
        result.committed = true;
        lock.unlock();
    } catch (...) {
        result.error = std::current_exception();
    }

    auto cleanup_error = cleanup_transfer(state, record);
    if (!result.error)
        result.error = cleanup_error;
    lock.lock();
    result.cleanup_pending = bool(cleanup_error);
    if (record.transfer->source_quarantined && record.transfer->shared_staging) {
        // 在解除块保护前把槽交给故障记录；之后即使其他线程立即 retry/erase 也安全。
        record.transfer->shared_staging = false;
        if (quarantined)
            *quarantined = true;
    }
    record.meta.state =
        record.transfer->source_quarantined ? BlockState::ERROR : BlockState::RESIDENT;
    if (!result.cleanup_pending)
        record.transfer.reset();
    record.operation_busy = false;
    return result;
}

TransferSlots::TransferSlots(std::size_t slot_bytes, SSDStorageBackend* device) : bytes(slot_bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("Pinned slot capacity overflow");
#if INFER_ENABLE_CUDA
    if (device) {
        try {
            auto first = device->allocate_dma_buffer(bytes);
            if (first) {
                dma = std::make_unique<std::array<std::shared_ptr<SSDDmaBuffer>, 2>>();
                (*dma)[0] = std::move(first);
                (*dma)[1] = device->allocate_dma_buffer(bytes);
                if (!(*dma)[1]) throw std::runtime_error("Incomplete device DMA staging");
            }
        } catch (const std::bad_alloc&) {
            dma.reset(); // Older budgets cover only bounce slots; keep bounded pinned staging.
        }
        if (dma) {
            std::size_t registered = 0;
            try {
                for (auto& buffer : *dma) {
                    check_cuda(cudaHostRegister(buffer->data(), buffer->size(), cudaHostRegisterPortable),
                               "Register device transfer staging");
                    data[registered++] = buffer->data();
                }
            } catch (...) {
                bool retained = false;
                for (std::size_t i = 0; i < registered; ++i)
                    retained |= cudaHostUnregister(data[i]) != cudaSuccess;
                if (retained) (void)dma.release();
                throw;
            }
            return;
        }
    }
#else
    (void)device;
#endif
    CachingCPUAllocatorConfig config;
    config.max_reuse_waste = 0;
    pool = std::make_unique<CachingPinnedHostAllocator>(config);
    for (auto& ptr : data)
        ptr = pool->allocate(bytes);
}
TransferSlots::~TransferSlots() {
#if INFER_ENABLE_CUDA
    if (dma && !quarantined)
        for (auto* ptr : data)
            quarantined |= cudaHostUnregister(ptr) != cudaSuccess;
#endif
    if (quarantined) {
        // 设备故障使 DMA 完成未知；池及其内存一起保留到进程退出。
        std::fputs("Storage pinned slots quarantined until process exit\n", stderr);
        (void)pool.release();
        (void)dma.release(); // Keep device leases alive when CUDA completion is unknown.
    }
}

std::exception_ptr cleanup_ssd_copy(StorageState& state, LogicalBlockRecord& record) {
    auto& copy = record.ssd_copy;
    if (!copy.backend)
        return {};
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!copy.pending) {
            state.ssd_tier.transition(StorageCharge::REPLICA, StorageCharge::PENDING_CLEANUP,
                                      copy.bytes);
            copy.pending = true;
        }
        record.ssd_copy_version = 0;
    }
    try {
        copy.backend->release(copy.handle);
    } catch (...) {
        return std::current_exception();
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    state.ssd_tier.release(StorageCharge::PENDING_CLEANUP, copy.bytes);
    copy = {};
    return {};
}

std::exception_ptr cleanup_transfer(StorageState& state, LogicalBlockRecord& record) {
    auto& op = *record.transfer;
    if (op.completion_unknown) {
        try {
            op.gpu->wait_transfer();
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.charge_cleanup(record);
            op.source_quarantined = true;
            auto& resource = op.auxiliary;
            if (resource.backend && !resource.pending) {
                state.tier(resource.backend->location())
                    .transition(StorageCharge::MIGRATION, StorageCharge::PENDING_CLEANUP,
                                resource.bytes);
                resource.pending = true;
            }
            return std::current_exception();
        }
        op.completion_unknown = false;
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (op.source_quarantined) {
            // erase 已把当前副本也列入清理，只有保留当前副本时才恢复 resident 计费。
            if (record.meta.state != BlockState::EVICTED) {
                const auto bytes = record.backend->allocation_bytes(record.meta.size);
                state.tier(record.meta.location)
                    .transition(StorageCharge::PENDING_CLEANUP, StorageCharge::RESIDENT, bytes);
                record.cleanup_charged = false;
            }
            op.source_quarantined = false;
        }
        if (op.staging != 2) {
            if (!op.shared_staging)
                state.staging->busy[op.staging] = false;
            op.staging = 2;
        }
    }
    auto& resource = op.auxiliary;
    if (resource.backend) {
        try {
            if (resource.handle)
                resource.backend->release(resource.handle);
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!resource.pending) {
                state.tier(resource.backend->location())
                    .transition(StorageCharge::MIGRATION, StorageCharge::PENDING_CLEANUP,
                                resource.bytes);
                resource.pending = true;
            }
            return std::current_exception();
        }
        std::lock_guard<std::mutex> lock(state.mutex);
        state.tier(resource.backend->location())
            .release(resource.pending ? StorageCharge::PENDING_CLEANUP : StorageCharge::MIGRATION,
                     resource.bytes);
        resource = {};
    }
    return {};
}
} // namespace infer::storage_detail
