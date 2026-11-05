#include "infer/cpu/pinned_memory_pool.h"
#include "storage_state.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

namespace infer::storage_detail {

std::size_t StorageState::available_staging() {
    if (!staging)
        staging = std::make_unique<TransferSlots>(block_bytes,
            dynamic_cast<SSDStorageBackend*>(ssd_tier.backend.get()));
    const auto slot = std::find(staging->busy.begin(), staging->busy.end(), false);
    if (slot == staging->busy.end())
        throw StorageBusy("Both pinned transfer slots are busy");
    return slot - staging->busy.begin();
}

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
            staging = state.available_staging();
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
            staging = state.available_staging();
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

MigrationResult finish_transfer(StorageState& state, LogicalBlockRecord& record,
                                std::exception_ptr error, bool* quarantined) {
    const auto target = record.transfer->auxiliary.backend->location();
    const auto bytes = record.transfer->auxiliary.bytes;
    auto& target_tier = state.tier(target);
    MigrationResult result{false, false, error};
    auto& auxiliary = record.transfer->auxiliary;
    try {
        if (!error) {
            const auto source_bytes = record.backend->allocation_bytes(record.meta.size);
            std::lock_guard<std::mutex> lock(state.mutex);
            // 唯一提交点：新副本已完整可读。之后清理失败也不能再宣称回滚。
            auto& source_tier = state.tier(record.meta.location);
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
        }
    } catch (...) {
        result.error = std::current_exception();
    }

    auto cleanup_error = cleanup_transfer(state, record);
    if (!result.error)
        result.error = cleanup_error;
    std::lock_guard<std::mutex> lock(state.mutex);
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

} // namespace

MigrationResult execute_transfer(StorageState& state, LogicalBlockRecord& record,
                                 bool* quarantined) {
    std::exception_ptr error;
    auto& op = *record.transfer;
    try {
        if (!op.auxiliary.handle)
            op.auxiliary.handle = op.auxiliary.backend->reserve(record.meta.size);
        if (op.reuse_ssd_copy)
            op.auxiliary.backend->validate(op.auxiliary.handle, record.meta);
        else
            copy_block(state, record, *op.auxiliary.backend, op.auxiliary.handle);
    } catch (...) {
        error = std::current_exception();
    }
    return finish_transfer(state, record, error, quarantined);
}

std::size_t execute_chunk_batch(StorageState& state, const std::vector<LogicalBlockRecord*>& records,
                                std::vector<MigrationResult>& results, std::size_t first,
                                bool& quarantined) {
    auto* head = records[first];
    if (!head || head->transfer->reuse_ssd_copy)
        return first;
    const auto source = head->meta.location;
    const auto target = head->transfer->auxiliary.backend->location();
    const bool write = target == StorageLocation::SSD;
    auto* ssd = dynamic_cast<SSDStorageBackend*>(write ? head->transfer->auxiliary.backend
                                                      : head->backend);
    if (!ssd || !ssd->supports_chunk_io())
        return first;
    const bool gpu = head->transfer->gpu != nullptr;
    const auto limit = gpu ? state.staging->bytes / state.block_bytes : records.size() - first;
    auto end = first + 1;
    while (end < records.size() && end - first < limit) {
        const auto* record = records[end];
        if (!record || record->meta.location != source || record->transfer->reuse_ssd_copy)
            break;
        ++end;
    }
    if (end - first < 2)
        return first;

    std::size_t preparing = first;
    std::vector<void*> hosts;
    try {
        std::vector<StorageWrite> writes;
        std::vector<StorageRead> reads;
        hosts.reserve(end - first);
        if (write) writes.reserve(end - first);
        else reads.reserve(end - first);
        for (; preparing < end; ++preparing) {
            auto& record = *records[preparing];
            auto& op = *record.transfer;
            if (!op.auxiliary.handle)
                op.auxiliary.handle = op.auxiliary.backend->reserve(record.meta.size);
            auto* host = gpu
                ? static_cast<char*>(state.staging->data[op.staging]) + (preparing - first) * state.block_bytes
                : checked_memory_view(write ? *record.backend : *op.auxiliary.backend,
                                      write ? record.allocation : op.auxiliary.handle,
                                      StorageLocation::CPU, record.meta.size).data;
            hosts.push_back(host);
            if (!write) {
                reads.push_back({record.allocation, record.meta, host});
                continue;
            }
            record.backend->validate(record.allocation, record.meta);
            if (gpu) {
                op.completion_unknown = true;
                record.backend->read(record.allocation, 0, host, record.meta.size);
                op.completion_unknown = false;
            }
            writes.push_back({op.auxiliary.handle, record.meta, host});
        }
        if (write)
            ssd->write_published_many(writes);
        else
            ssd->read_validated_many(reads);
    } catch (...) {
        const auto error = std::current_exception();
        // One I/O error can cover multiple pages. Retry separately to preserve
        // the per-page result and let healthy pages in the batch commit.
        auto failed = end;
        if (preparing < end && records[preparing]->transfer->completion_unknown) {
            failed = preparing;
            results[failed] = finish_transfer(state, *records[failed], error, &quarantined);
        }
        for (auto i = first; i < end; ++i) {
            if (i == failed)
                continue;
            results[i] = quarantined ? finish_transfer(state, *records[i], error, &quarantined)
                                     : execute_transfer(state, *records[i], &quarantined);
        }
        return end;
    }
    for (auto i = first; i < end; ++i) {
        auto& record = *records[i];
        auto& op = *record.transfer;
        std::exception_ptr error;
        if (quarantined) {
            error = results[i - 1].error;
        } else if (!write) {
            try {
                if (gpu) {
                    op.completion_unknown = true;
                    op.auxiliary.backend->write(op.auxiliary.handle, 0, hosts[i - first], record.meta.size);
                    op.completion_unknown = false;
                }
                op.auxiliary.backend->publish(op.auxiliary.handle, record.meta);
            } catch (...) {
                error = std::current_exception();
            }
        }
        results[i] = finish_transfer(state, record, error, &quarantined);
    }
    return end;
}

TransferSlots::TransferSlots(std::size_t slot_bytes, SSDStorageBackend* device)
    : bytes(device && device->supports_chunk_io() ? 2 * 1024 * 1024 : slot_bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("Pinned slot capacity overflow");
#if INFER_ENABLE_CUDA
    if (device) {
        try {
            auto first = device->allocate_transfer_buffer(bytes);
            if (first) {
                auto second = device->allocate_transfer_buffer(bytes);
                if (second) {
                    dma = std::make_unique<std::array<std::shared_ptr<SSDDmaBuffer>, 2>>(
                        std::array<std::shared_ptr<SSDDmaBuffer>, 2>{std::move(first), std::move(second)});
                    for (std::size_t i = 0; i < data.size(); ++i)
                        data[i] = (*dma)[i]->data();
                    return;
                }
            }
        } catch (const std::bad_alloc&) {
            // Insufficient device DMA budget: use CUDA-owned pinned staging.
        }
    }
#endif
    CachingCPUAllocatorConfig config;
    config.max_reuse_waste = 0;
    pool = std::make_unique<CachingPinnedHostAllocator>(config);
    for (auto& ptr : data)
        ptr = pool->allocate_io(bytes);
}
TransferSlots::~TransferSlots() {
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
        state.charge_pending(copy, StorageCharge::REPLICA);
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
            if (resource.backend)
                state.charge_pending(resource, StorageCharge::MIGRATION);
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
            state.charge_pending(resource, StorageCharge::MIGRATION);
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
