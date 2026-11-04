#include "infer/io/direct_file.h"
#include "infer/storage/tiered_storage.h"
#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/pinned_memory_pool.h"
#include "storage_state.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace infer {
using storage_detail::StorageCharge;
namespace {
std::unique_ptr<StorageBackend> default_cpu_backend(const StorageConfig& config) {
    CachingCPUAllocatorConfig pool;
    // Central reuse avoids retaining tiny slots in TLS. The hard ceiling also
    // bounds active + cached when successive payloads use different size classes.
    pool.max_cached_bytes = config.budget.cpu_bytes;
    pool.max_reserved_bytes = config.budget.cpu_bytes;
    pool.max_reuse_waste = 0;
    pool.enable_thread_cache = false;
    if (config.mapped_cpu_pool && config.cpu_memory == HostMemoryKind::PAGEABLE &&
        config.block_bytes >= 256 * 1024)
        pool.mapped_block_bytes = config.block_bytes;
    if (config.cpu_memory == HostMemoryKind::PINNED)
        return std::make_unique<CPUStorageBackend>(
            config.block_bytes, std::make_shared<CachingPinnedHostAllocator>(pool),
            HostMemoryKind::PINNED, config.variable_cpu_slots);
    if (config.cpu_memory != HostMemoryKind::PAGEABLE)
        throw std::invalid_argument("Invalid host memory kind");
    return std::make_unique<CPUStorageBackend>(config.block_bytes,
                                               std::make_shared<CachingCPUDeviceAllocator>(pool),
                                               HostMemoryKind::PAGEABLE, config.variable_cpu_slots);
}
std::unique_ptr<StorageBackend> cpu_from_allocator(std::shared_ptr<DeviceAllocator> allocator) {
    if (!allocator)
        throw std::invalid_argument("Storage requires a non-null CPU allocator");
    return std::make_unique<CPUStorageBackend>(StorageConfig{}.block_bytes, std::move(allocator));
}
void validate_backend(const StorageBackend* backend, StorageLocation location,
                      std::size_t slot_bytes) {
    const auto charged = location == StorageLocation::SSD
                             ? SSDStorageBackend::allocation_size(slot_bytes)
                             : slot_bytes;
    if (backend && (backend->location() != location || backend->slot_bytes() != slot_bytes ||
                    (location == StorageLocation::SSD ? backend->allocation_bytes() < charged
                                                      : backend->allocation_bytes() != charged))) {
        throw std::invalid_argument("Storage backend location or slot size mismatch");
    }
}

// 同步主机导出的内部读凭证。只借用调用栈上的记录指针数组，不为单读分配堆内存。
// 构造前已在 state.mutex 下完成整批检查；构造后解锁执行 I/O，析构自动归还全部保护。
// records/StorageState 活得比本对象长；后端返回/抛出前必须完成所有主机缓冲写入。
class HostReadAccess {
  public:
    HostReadAccess(storage_detail::StorageState& state,
                   storage_detail::LogicalBlockRecord* const* records, std::size_t count) noexcept
        : state_(state), records_(records), count_(count) {
        for (std::size_t i = 0; i < count_; ++i) {
            auto& meta = records_[i]->meta;
            ++meta.lease_count;
            ++meta.access_count;
            meta.last_access = ++state_.access_clock;
        }
    }
    ~HostReadAccess() {
        std::lock_guard<std::mutex> lock(state_.mutex);
        for (std::size_t i = 0; i < count_; ++i)
            --records_[i]->meta.lease_count;
    }
    HostReadAccess(const HostReadAccess&) = delete;
    HostReadAccess& operator=(const HostReadAccess&) = delete;

  private:
    storage_detail::StorageState& state_;
    storage_detail::LogicalBlockRecord* const* records_;
    std::size_t count_;
};
} // namespace

storage_detail::TierState& storage_detail::StorageState::tier(StorageLocation location) {
    switch (location) {
    case StorageLocation::CPU:
        return cpu_tier;
    case StorageLocation::GPU:
        return gpu_tier;
    case StorageLocation::SSD:
        return ssd_tier;
    default:
        throw std::invalid_argument("Storage tier is not implemented");
    }
}
void storage_detail::StorageState::charge_cleanup(LogicalBlockRecord& record) noexcept {
    if (record.cleanup_charged)
        return;
    auto& current_tier = tier(record.meta.location);
    const auto bytes = record.backend->allocation_bytes(record.meta.size);
    current_tier.transition(StorageCharge::RESIDENT, StorageCharge::PENDING_CLEANUP, bytes);
    record.cleanup_charged = true;
}
storage_detail::StorageState::~StorageState() {
    bool quarantine_host = false;
    // 此时管理器及所有 Lease 都已销毁。GPU 后端还有自己的安全析构兜底。
    logical_blocks.for_each([&](const auto& item) {
        auto& record = *item;
        try {
            if (record.transfer) {
                auto error = cleanup_transfer(*this, record);
                if (error && record.transfer->completion_unknown) {
                    if (staging)
                        staging->quarantined = true;
                    quarantine_host |= record.transfer->direct_host;
                    return;
                }
            }
            for (auto& use : record.consumer_uses) {
                if (use.pending)
                    record.backend->wait(use.stream);
            }
            if (auto error = cleanup_ssd_copy(*this, record))
                std::rethrow_exception(error);
            record.backend->release(record.allocation);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Storage cleanup for logical block %llu failed: %s\n",
                         static_cast<unsigned long long>(record.meta.id), error.what());
        } catch (...) {
            std::fputs("Storage cleanup failed\n", stderr);
        }
    });
    // 直接 DMA 的 host 槽也必须隔离；不能让 CPU 后端析构把未确认完成的地址交回池。
    if (quarantine_host) {
        std::fputs("Storage pinned CPU backend quarantined until process exit\n", stderr);
        (void)cpu_tier.backend.release();
    }
}
TieredStorage::TieredStorage() : TieredStorage(StorageConfig{}) {}
TieredStorage::TieredStorage(std::shared_ptr<DeviceAllocator> allocator)
    : TieredStorage(StorageConfig{}, cpu_from_allocator(std::move(allocator))) {}
TieredStorage::TieredStorage(StorageConfig config, std::unique_ptr<StorageBackend> cpu,
                             std::unique_ptr<StorageBackend> gpu,
                             std::unique_ptr<StorageBackend> ssd) {
    storage_detail::validate_slot_bytes(config.block_bytes);
    if (!config.max_async_migrations || config.max_async_migrations > 64)
        throw std::invalid_argument("Async migration capacity must be 1..64");
    if ((config.budget.cpu_bytes && config.budget.cpu_bytes < config.block_bytes) ||
        (config.budget.gpu_bytes && config.budget.gpu_bytes < config.block_bytes) ||
        config.gpu_device < 0) {
        throw std::invalid_argument("Invalid storage budget or GPU device");
    }
    validate_backend(cpu.get(), StorageLocation::CPU, config.block_bytes);
    validate_backend(gpu.get(), StorageLocation::GPU, config.block_bytes);
    if (config.budget.ssd_bytes || ssd)
        validate_backend(ssd.get(), StorageLocation::SSD, config.block_bytes);
    if ((!config.budget.cpu_bytes && cpu) || (!config.budget.gpu_bytes && gpu) ||
        (!config.budget.ssd_bytes && ssd)) {
        throw std::invalid_argument("Backend provided for disabled storage tier");
    }
    state_ = std::make_shared<storage_detail::StorageState>();
    if (config.budget.cpu_bytes && !cpu)
        cpu = default_cpu_backend(config);
    if (config.budget.gpu_bytes && !gpu) {
        gpu = std::make_unique<GPUStorageBackend>(config.block_bytes, config.budget.gpu_bytes,
                                                  config.gpu_device);
    }
    if (config.budget.ssd_bytes) {
        if (config.budget.ssd_bytes < SSDStorageBackend::allocation_size(config.block_bytes))
            throw std::invalid_argument("SSD budget omits record header");
        if (!ssd) {
            if (config.ssd_direct && !config.ssd_spdk.bdev_name.empty())
                throw std::invalid_argument("Choose direct-file or SPDK, not both");
            if (config.ssd_direct)
                ssd = std::make_unique<SSDStorageBackend>(config.block_bytes, config.budget.ssd_bytes,
                    io::open_direct_file_device(config.ssd_directory,
                        config.budget.ssd_bytes / io_alignment * io_alignment, config.ssd_io.queue_depth),
                    config.ssd_verification);
            else if (config.ssd_spdk.bdev_name.empty())
                ssd = std::make_unique<SSDStorageBackend>(
                    config.block_bytes, config.budget.ssd_bytes, config.ssd_directory,
                    config.ssd_io, config.ssd_verification);
            else
                ssd = std::make_unique<SSDStorageBackend>(
                    config.block_bytes, config.budget.ssd_bytes, open_spdk_device(config.ssd_spdk),
                    config.ssd_verification);
        }
        if (ssd->allocation_bytes() > config.budget.ssd_bytes)
            throw std::invalid_argument("SSD budget omits device alignment");
    }
    state_->cpu_tier.backend = std::move(cpu);
    state_->gpu_tier.backend = std::move(gpu);
    state_->ssd_tier.backend = std::move(ssd);
    state_->cpu_tier.usage.limit_bytes = config.budget.cpu_bytes;
    state_->gpu_tier.usage.limit_bytes = config.budget.gpu_bytes;
    state_->ssd_tier.usage.limit_bytes = config.budget.ssd_bytes;
    state_->block_bytes = config.block_bytes;
    state_->max_async_migrations = config.max_async_migrations;
    state_->retain_ssd_copy = config.retain_ssd_copy;
}
TieredStorage::~TieredStorage() = default;

BlockId TieredStorage::create(std::size_t size, StorageLocation location, int priority) {
    if (!size || size > state_->block_bytes)
        throw std::invalid_argument("Invalid logical storage block size");
    if (location != StorageLocation::CPU && location != StorageLocation::GPU)
        throw std::invalid_argument("Initialize in CPU/GPU memory before migrating to SSD");
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto& tier = state_->tier(location);
    if (!tier.backend)
        throw std::invalid_argument("Storage tier is disabled");
    const auto bytes = tier.backend->allocation_bytes(size);
    // 减法检查避免 reserved + bytes 溢出；所有在途/待清理容量都占预算。
    tier.check_capacity(bytes);
    if (state_->access_clock == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Storage access clock exhausted");
    }
    const auto id = state_->logical_blocks.create([&](std::size_t) {
        auto record = std::make_unique<storage_detail::LogicalBlockRecord>();
        record->backend = tier.backend.get();
        record->allocation = tier.backend->reserve(size);
        return record;
    });
    auto& record = *state_->logical_blocks.at(id);
    record.meta.id = id;
    record.meta.location = location;
    record.meta.size = size;
    record.meta.priority = priority;
    record.meta.last_access = ++state_->access_clock;
    tier.reserve(bytes);
    tier.transition(StorageCharge::MIGRATION, StorageCharge::RESIDENT, bytes);
    return id;
}
BlockLease TieredStorage::acquire(BlockId id, AccessMode mode, cudaStream_t stream) {
    return acquire_impl(id, mode, stream, StorageLocation::REMOTE);
}
BlockLease TieredStorage::acquire_at(BlockId id, StorageLocation location, AccessMode mode,
                                   cudaStream_t stream) {
    if (location != StorageLocation::CPU && location != StorageLocation::GPU)
        throw std::invalid_argument("Lease requires a memory location");
    return acquire_impl(id, mode, stream, location);
}
BlockLease TieredStorage::acquire_impl(BlockId id, AccessMode mode, cudaStream_t stream,
                                     StorageLocation expected, std::size_t bytes) {
    if (mode != AccessMode::READ && mode != AccessMode::WRITE)
        throw std::invalid_argument("Invalid storage access mode");
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto& record = *state_->logical_blocks.at(id);
    auto& meta = record.meta;
    if (expected != StorageLocation::REMOTE && meta.location != expected)
        throw StorageBusy("Storage location changed before acquiring a lease");
    if (meta.location == StorageLocation::SSD)
        throw std::logic_error("Migrate SSD content to CPU/GPU before acquiring a lease");
    if (meta.state != BlockState::RESIDENT || record.writing || record.operation_busy ||
        (mode == AccessMode::WRITE && meta.lease_count) ||
        (mode == AccessMode::READ && !record.initialized)) {
        throw std::logic_error("Logical storage block is busy, uninitialized or unavailable");
    }
    if (state_->access_clock == SIZE_MAX || meta.lease_count == SIZE_MAX ||
        meta.access_count == SIZE_MAX ||
        (mode == AccessMode::WRITE && meta.version == UINT64_MAX)) {
        throw std::overflow_error("Storage access count or content version exhausted");
    }
    record.backend->validate_stream(stream);
    const auto view = storage_detail::checked_memory_view(*record.backend, record.allocation,
                                                          meta.location, meta.size);
    if (bytes && (bytes > state_->block_bytes ||
                  record.backend->allocation_bytes(bytes) != view.capacity))
        throw std::invalid_argument("Replacement requires the same physical capacity");
    std::size_t use_index = 0;
    if (meta.location == StorageLocation::GPU) {
        while (use_index < record.consumer_uses.size() && record.consumer_uses[use_index].pending)
            ++use_index;
        if (use_index == record.consumer_uses.size())
            record.consumer_uses.push_back({stream, true});
        else
            record.consumer_uses[use_index] = {stream, true};
    }
    // 从这里开始无可失败操作，原子提交使用权与访问计数。
    if (mode == AccessMode::WRITE) {
        record.writing = true;
        record.initialized = false;
        // 原地写可能被放弃，必须在交出地址前就使旧 SSD 内容失效。
        record.ssd_copy_version = 0;
        if (bytes) meta.size = bytes;
    }
    ++meta.lease_count;
    ++meta.access_count;
    meta.last_access = ++state_->access_clock;
    return BlockLease(state_, &record, mode, view.data, view.device_id, stream, use_index);
}
BlockLease TieredStorage::acquire_cpu_write(BlockId id, std::size_t bytes) {
    if (!bytes) throw std::invalid_argument("Replacement requires a nonzero payload");
    return acquire_impl(id, AccessMode::WRITE, nullptr, StorageLocation::CPU, bytes);
}
void TieredStorage::read_to_host(BlockId id, void* dst, std::size_t bytes) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto& record = *state_->logical_blocks.at(id);
    auto& meta = record.meta;
    if (!dst || bytes != meta.size)
        throw std::invalid_argument("Host read requires a complete valid buffer");
    if (meta.location != StorageLocation::CPU && meta.location != StorageLocation::SSD)
        throw std::invalid_argument("Host read only supports CPU/SSD blocks");
    if (meta.state != BlockState::RESIDENT || record.writing || record.operation_busy ||
        !record.initialized)
        throw std::logic_error("Storage block is busy or unpublished");
    if (meta.lease_count == SIZE_MAX || meta.access_count == SIZE_MAX ||
        state_->access_clock == SIZE_MAX)
        throw std::overflow_error("Host read access counter exhausted");
    auto* protected_record = &record;
    HostReadAccess access(*state_, &protected_record, 1);
    const auto expected = meta;
    lock.unlock();
    // 后端 I/O 不占块表锁；lease_count 阻止源的覆写、迁移和删除。
    record.backend->read_validated(record.allocation, expected, dst);
}
void TieredStorage::read_to_host_many(const std::vector<BlockId>& ids,
                                      const std::vector<void*>& destinations, std::size_t bytes) {
    read_to_host_many(ids, destinations, std::vector<std::size_t>(ids.size(), bytes));
}
void TieredStorage::read_to_host_many(const std::vector<BlockId> &ids,
                                      const std::vector<void *> &destinations,
                                      const std::vector<std::size_t> &sizes) {
    if (ids.empty() || ids.size() > 64 || ids.size() != destinations.size() ||
        ids.size() != sizes.size())
        throw std::invalid_argument("Invalid host read batch");
    std::vector<storage_detail::LogicalBlockRecord*> records;
    std::vector<StorageRead> cpu, ssd;
    records.reserve(ids.size());
    cpu.reserve(ids.size());
    ssd.reserve(ids.size());
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (ids.size() > SIZE_MAX - state_->access_clock)
        throw std::overflow_error("Host batch access clock exhausted");
    // 完整预检和请求表分配先于取得使用权，后项参数失败不修改任何源的计数。
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto bytes = sizes[i];
        if (std::find(ids.begin(), ids.begin() + i, ids[i]) != ids.begin() + i)
            throw std::invalid_argument("Duplicate read block");
        auto& record = *state_->logical_blocks.at(ids[i]);
        const auto& meta = record.meta;
        if (!destinations[i] || bytes != meta.size)
            throw std::invalid_argument("Invalid host batch destination");
        const auto begin = reinterpret_cast<std::uintptr_t>(destinations[i]);
        if (bytes > UINTPTR_MAX - begin)
            throw std::invalid_argument("Host batch address overflow");
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(destinations[j]);
            if (begin < other + sizes[j] && other < begin + bytes)
                throw std::invalid_argument("Overlapping host batch destinations");
        }
        if ((meta.location != StorageLocation::CPU && meta.location != StorageLocation::SSD) ||
            meta.state != BlockState::RESIDENT || record.writing || record.operation_busy ||
            !record.initialized)
            throw StorageBusy("Host batch source is busy");
        if (meta.lease_count == SIZE_MAX || meta.access_count == SIZE_MAX)
            throw std::overflow_error("Host batch lease overflow");
        records.push_back(&record);
        (meta.location == StorageLocation::CPU ? cpu : ssd)
            .push_back({record.allocation, meta, destinations[i]});
    }
    HostReadAccess access(*state_, records.data(), records.size());
    lock.unlock();
    if (!cpu.empty())
        state_->cpu_tier.backend->read_validated_many(cpu);
    if (!ssd.empty())
        state_->ssd_tier.backend->read_validated_many(ssd);
}

BlockMeta TieredStorage::metadata(BlockId id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->logical_blocks.at(id)->meta;
}
StorageUsage TieredStorage::stats(StorageLocation location) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->tier(location).usage;
}
BackendPoolStats TieredStorage::pool_stats(StorageLocation location) const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    const auto& tier = state_->tier(location);
    auto* backend = tier.backend.get();
    lock.unlock();
    return backend ? backend->pool_stats() : BackendPoolStats{};
}
BackendIOStats TieredStorage::io_stats(StorageLocation location) const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto* backend = state_->tier(location).backend.get();
    lock.unlock();
    return backend ? backend->io_stats() : BackendIOStats{};
}
bool TieredStorage::erase(BlockId id) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto& record = *state_->logical_blocks.at(id);
    if (record.meta.lease_count || record.operation_busy ||
        record.meta.state == BlockState::EVICTED || record.meta.state == BlockState::MOVING)
        return false;
    record.meta.state = BlockState::EVICTED;
    state_->charge_cleanup(record);
    // EVICTED 阻止其他 acquire/erase 删除此节点，等待 GPU 时不占用块表锁。
    lock.unlock();
    try {
        if (record.transfer) {
            auto error = storage_detail::cleanup_transfer(*state_, record);
            if (error)
                std::rethrow_exception(error);
        }
        if (auto error = storage_detail::cleanup_ssd_copy(*state_, record))
            std::rethrow_exception(error);
        for (auto& use : record.consumer_uses) {
            if (use.pending) {
                record.backend->wait(use.stream);
                use.pending = false;
            }
        }
        record.backend->release(record.allocation);
    } catch (...) {
        lock.lock();
        record.meta.state = BlockState::ERROR;
        throw;
    }
    lock.lock();
    state_->tier(record.meta.location)
        .release(StorageCharge::PENDING_CLEANUP,
                 record.backend->allocation_bytes(record.meta.size));
    state_->logical_blocks.erase(id);
    return true;
}

StagingUsage TieredStorage::staging_stats() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->staging)
        return {};
    const auto& slots = *state_->staging;
    return {2 * slots.bytes, std::size_t(slots.busy[0]) + std::size_t(slots.busy[1])};
}

MigrationResult TieredStorage::migrate(BlockId id, StorageLocation target) {
    auto* record = storage_detail::prepare_transfer(*state_, id, target);
    return record ? storage_detail::execute_transfer(*state_, *record)
                  : MigrationResult{true, false, {}};
}
std::future<MigrationResult> TieredStorage::migrate_async(BlockId id, StorageLocation target) {
    std::lock_guard<std::mutex> lock(executor_mutex_);
    if (!executor_)
        executor_ = std::make_unique<storage_detail::MigrationExecutor>(
            state_, state_->max_async_migrations);
    return executor_->submit(id, target);
}

std::future<std::vector<MigrationResult>>
TieredStorage::migrate_batch_async(const std::vector<BlockId>& ids, StorageLocation target) {
    std::lock_guard<std::mutex> lock(executor_mutex_);
    if (!executor_)
        executor_ = std::make_unique<storage_detail::MigrationExecutor>(
            state_, state_->max_async_migrations);
    return executor_->submit_batch(ids, target);
}

bool TieredStorage::retry_cleanup(BlockId id) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto& record = *state_->logical_blocks.at(id);
    if (record.operation_busy || record.meta.state == BlockState::MOVING ||
        record.meta.state == BlockState::EVICTED) {
        throw StorageBusy("Storage operation is in progress");
    }
    if (!record.transfer && !record.ssd_copy.pending)
        return true;
    record.operation_busy = true;
    lock.unlock();
    auto error =
        record.transfer ? storage_detail::cleanup_transfer(*state_, record) : std::exception_ptr{};
    if (!error && record.ssd_copy.pending)
        error = storage_detail::cleanup_ssd_copy(*state_, record);
    lock.lock();
    // Lease 的其他错误不能被迁移清理覆盖；仅恢复本次迁移隔离过的当前副本。
    if (!record.cleanup_charged)
        record.meta.state = BlockState::RESIDENT;
    if (!error)
        record.transfer.reset();
    record.operation_busy = false;
    return !error;
}
bool TieredStorage::drop_ssd_copy(BlockId id) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto& record = *state_->logical_blocks.at(id);
    if (record.meta.lease_count || record.operation_busy || record.transfer ||
        record.meta.state != BlockState::RESIDENT)
        return false;
    if (!record.ssd_copy.backend)
        return true;
    record.operation_busy = true;
    lock.unlock();
    auto error = storage_detail::cleanup_ssd_copy(*state_, record);
    lock.lock();
    record.operation_busy = false;
    if (error)
        std::rethrow_exception(error);
    return true;
}
} // namespace infer
