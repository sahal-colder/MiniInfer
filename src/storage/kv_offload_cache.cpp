#include "kv_offload_state.h"
#include "infer/cpu/cpu_block.h"
#include "scope_exit.h"

#include <algorithm>
#include <cerrno>
#include <system_error>

namespace infer {
using storage_detail::OffloadTimer;

namespace {
StorageConfig storage_config(const KVOffloadConfig& config) {
    if (!config.payload_bytes || !config.logical_slots || !config.cpu_slots ||
        config.cpu_slots > config.logical_slots)
        throw std::invalid_argument(
            "Offload requires payload, logical slots and 1..logical_slots CPU slots");
    if (!config.write_concurrency || config.write_concurrency > 16)
        throw std::invalid_argument("Write concurrency must be 1..16");
    const auto slot = KVOffloadCache::allocation_size(config.payload_bytes, config.compact_slots);
    const auto record = config.ssd_spdk.bdev_name.empty() && !config.ssd_direct
                            ? SSDStorageBackend::allocation_size(slot)
                            : SSDStorageBackend::allocation_size(slot, 4096);
    if (config.cpu_slots > SIZE_MAX / slot || config.logical_slots > SIZE_MAX / record)
        throw std::overflow_error("Offload capacity overflow");
    if (config.prefetch_depth > 1024 || !config.min_reuse_count)
        throw std::invalid_argument("Invalid KV cache policy");
    if (config.batch_promotions > 64)
        throw std::invalid_argument("Batch promotion bound exceeds maximum batch size");
    StorageConfig result;
    result.block_bytes = slot;
    result.budget.cpu_bytes = config.cpu_slots * slot;
    result.budget.ssd_bytes = config.enable_ssd ? config.logical_slots * record : 0;
    result.ssd_directory = config.ssd_directory;
    result.ssd_io = config.ssd_io;
    result.ssd_direct = config.ssd_direct;
    result.max_async_migrations = config.write_concurrency;
    result.retain_ssd_copy = config.retain_ssd_copy;
    result.cpu_memory = config.cpu_memory;
    result.variable_cpu_slots = config.variable_payloads;
    result.mapped_cpu_pool = config.mapped_cpu_pool.value_or(false);
    result.ssd_verification = config.ssd_verification;
    result.ssd_spdk = config.ssd_spdk;
    return result;
}
} // namespace

std::size_t KVOffloadCache::allocation_size(std::size_t payload_bytes, bool compact_slots) {
    if (!payload_bytes) throw std::invalid_argument("Offload requires a nonzero payload");
    std::size_t slot = 4096;
    while (slot < payload_bytes) {
        if (slot > SIZE_MAX / 2) throw std::overflow_error("Offload slot size overflow");
        slot *= 2;
    }
    if (compact_slots) {
        const auto pages = (payload_bytes + 4095) / 4096 * 4096;
        return cpu_detail::capacities[cpu_detail::size_class(pages)];
    }
    return slot;
}

KVOffloadCache::KVOffloadCache(KVOffloadConfig config, std::unique_ptr<StorageBackend> cpu,
                               std::unique_ptr<StorageBackend> ssd)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(cpu), std::move(ssd))) {}
KVOffloadCache::~KVOffloadCache() = default;

KVOffloadMeta KVOffloadCache::metadata(std::size_t index) const {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.block) return {};
    bool ssd_copy = false;
    const auto meta = impl_->storage.metadata(entry.id(), ssd_copy);
    return {meta.id, meta.size, meta.location, meta.version, ssd_copy};
}

bool KVOffloadCache::erase(std::size_t index) {
    std::unique_lock<std::mutex> writer(impl_->writer_mutex);
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.id()) return false;
    if (entry.generation == UINT64_MAX) throw std::overflow_error("KV slot generation exhausted");
    entry.busy = true;
    storage_detail::ScopeExit release([&] {
        if (!lock.owns_lock()) lock.lock();
        entry.busy = false;
        impl_->changed.notify_all();
    });
    impl_->changed.wait(lock, [&] { return !entry.readers; });
    lock.unlock();
    if (!impl_->storage.erase(entry.id())) throw StorageBusy("KV slot is still in use");
    lock.lock();
    impl_->discard_prefetch(entry);
    impl_->unlink_cpu_slot(index);
    entry.block = nullptr;
    entry.prefetch = {};
    entry.ssd_hit_count = 0;
    ++entry.generation;
    --impl_->counters.populated_slots;
    ++impl_->counters.invalidations;
    return true;
}

bool KVOffloadCache::contains(std::size_t index) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->entries.at(index).block != nullptr;
}

bool KVOffloadCache::should_store(KVStoreHint hint) const noexcept {
    return hint.reuse_count >= impl_->config.min_reuse_count &&
           (!impl_->config.prompt_only || hint.is_prompt);
}

bool KVOffloadCache::write_if_admitted(std::size_t index, const void* data, KVStoreHint hint) {
    if (should_store(hint)) {
        write(index, data);
        return true;
    }
    erase(index);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ++impl_->counters.admission_skips;
    return false;
}

KVOffloadStats KVOffloadCache::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto result = impl_->counters;
    result.protected_cpu_slots = impl_->protected_count;
    result.read_ns = impl_->read_ns.load();
    result.write_ns = impl_->write_ns.load();
    result.lock_wait_ns = impl_->lock_wait_ns.load();
    result.demotion_ns = impl_->demotion_ns.load();
    result.writer_wait_ns = impl_->writer_wait_ns.load();
    result.prefetch_queue_ns = impl_->prefetch_queue_ns.load();
    result.prefetch_ns = impl_->prefetch_ns.load();
    const auto io = impl_->storage.io_stats(StorageLocation::SSD);
    result.ssd_device = io.device;
    result.ssd_copied_read_bytes = io.device.copied_read_bytes;
    result.ssd_copied_write_bytes = io.device.copied_write_bytes;
    result.ssd_direct_read_bytes = io.device.direct_read_bytes;
    result.ssd_direct_written_bytes = io.device.direct_written_bytes;
    result.ssd_reserve_ns = io.reserve_ns;
    result.ssd_read_ns = io.read_ns;
    result.ssd_write_ns = io.write_ns;
    result.ssd_publish_ns = io.publish_ns;
    result.file_io = io.file_io;
    result.ssd_record_writes = io.record_writes;
    result.ssd_source_crc_bytes = io.source_crc_bytes;
    result.ssd_payload_written_bytes = io.payload_written_bytes;
    result.ssd_payload_read_bytes = io.payload_read_bytes;
    result.ssd_verification_read_bytes = io.verification_read_bytes;
    const auto cpu = impl_->storage.stats(StorageLocation::CPU),
               ssd = impl_->storage.stats(StorageLocation::SSD);
    result.slot_bytes = cpu.limit_bytes / impl_->config.cpu_slots;
    result.cpu_reserved_bytes = cpu.reserved_bytes;
    result.cpu_limit_bytes = cpu.limit_bytes;
    result.cpu_pool_reserved_bytes = impl_->storage.pool_stats(StorageLocation::CPU).reserved_bytes;
    result.ssd_reserved_bytes = ssd.reserved_bytes;
    result.ssd_limit_bytes = ssd.limit_bytes;
    result.ssd_replica_bytes = ssd.replica_bytes;
    result.cpu_pending_cleanup_bytes = cpu.pending_cleanup_bytes;
    result.ssd_pending_cleanup_bytes = ssd.pending_cleanup_bytes;
    return result;
}

std::shared_ptr<SSDDmaBuffer> KVOffloadCache::allocate_dma_buffer(std::size_t bytes) {
    return impl_->storage.allocate_dma_buffer(bytes);
}

KVOffloadCache::Impl::Impl(KVOffloadConfig value, std::unique_ptr<StorageBackend> cpu,
                           std::unique_ptr<StorageBackend> ssd)
    : config(std::move(value)), storage(storage_config(config), std::move(cpu), {}, std::move(ssd)),
      entries(config.logical_slots) {}

KVOffloadCache::Impl::~Impl() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    changed.notify_all();
    if (prefetch_worker.joinable()) prefetch_worker.join();
}

std::size_t KVOffloadCache::Impl::victim(bool cold_only) const noexcept {
    std::size_t fallback = none;
    for (auto index = lru_head; index != none; index = entries[index].next) {
        const auto& entry = entries[index];
        if (entry.busy || entry.readers) continue;
        if (!entry.hot) return index;
        if (fallback == none) fallback = index;
    }
    return cold_only ? none : fallback;
}

std::size_t KVOffloadCache::Impl::select_victims(const std::vector<std::size_t>& slots,
                                                 std::size_t needed,
                                                 std::array<std::size_t, 16>& selected) const {
    std::size_t count = 0;
    for (unsigned pass = 0; pass < 2 && count < needed; ++pass) {
        for (auto index = lru_head; index != none && count < needed; index = entries[index].next) {
            const auto& entry = entries[index];
            if (entry.hot == bool(pass) && !entry.busy && !entry.readers &&
                std::find(slots.begin(), slots.end(), index) == slots.end())
                selected[count++] = index;
        }
    }
    return count;
}

void KVOffloadCache::Impl::apply_migration(std::size_t index, const MigrationResult& result,
                                           StorageLocation target, bool prefetched, bool hot) {
    if (!result.committed) return;
    auto& entry = entries[index];
    if (target == StorageLocation::SSD) {
        discard_prefetch(entry);
        unlink_cpu_slot(index);
        ++counters.demotions;
        return;
    }
    entry.ssd_hit_count = 0;
    entry.prefetched = prefetched;
    touch_cpu_slot(index);
    if (hot) {
        mark_hot(index);
        ++counters.hot_promotions;
    }
    ++counters.promotions;
}

void KVOffloadCache::Impl::require_complete(const MigrationResult& result) {
    if (result.error) std::rethrow_exception(result.error);
    if (!result.committed || result.cleanup_pending)
        throw StorageBusy("Offload migration incomplete");
}

void KVOffloadCache::Impl::demote(std::unique_lock<std::mutex>& lock, std::size_t index) {
    lock.unlock();
    OffloadTimer timer{demotion_ns};
    const auto result = storage.migrate(entries[index].id(), StorageLocation::SSD);
    lock.lock();
    apply_migration(index, result, StorageLocation::SSD);
    require_complete(result);
}

void KVOffloadCache::Impl::make_room(std::unique_lock<std::mutex>& lock, bool cold_only,
                                     std::size_t bytes) {
    const auto usage = storage.stats(StorageLocation::CPU);
    if (bytes <= usage.limit_bytes - usage.reserved_bytes) return;
    if (!config.enable_ssd)
        throw std::system_error(ENOSPC, std::generic_category(), "CPU storage is full");
    std::size_t selected = none;
    for (;;) {
        selected = victim(cold_only);
        if (selected != none) break;
        if (cold_only) throw StorageBusy("No cold CPU slot available");
        if (lru_head == none) throw StorageBusy("CPU capacity held by pending cleanup");
        changed.wait(lock);
    }
    entries[selected].busy = true;
    storage_detail::ScopeExit release([&] {
        if (!lock.owns_lock()) lock.lock();
        entries[selected].busy = false;
        changed.notify_all();
    });
    demote(lock, selected);
}
} // namespace infer
