#include "infer/storage/kv_offload_cache.h"
#include "infer/cpu/cpu_block.h"
#include "infer/storage/tiered_storage.h"
#include "storage_state.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <system_error>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace infer {
namespace {
using Clock = std::chrono::steady_clock;
struct Timer {
    std::atomic<std::uint64_t>& total;
    Clock::time_point start = Clock::now();
    ~Timer() {
        total.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count(),
            std::memory_order_relaxed);
    }
};
std::shared_future<bool> ready_future(bool value) {
    std::promise<bool> result;
    result.set_value(value);
    return result.get_future().share();
}
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
    if (!payload_bytes)
        throw std::invalid_argument("Offload requires a nonzero payload");
    std::size_t slot = 4096;
    while (slot < payload_bytes) {
        if (slot > SIZE_MAX / 2)
            throw std::overflow_error("Offload slot size overflow");
        slot *= 2;
    }
    if (compact_slots) {
        const auto pages = (payload_bytes + 4095) / 4096 * 4096;
        return cpu_detail::capacities[cpu_detail::size_class(pages)];
    }
    return slot;
}

struct KVOffloadCache::Impl {
    static constexpr std::size_t none = SIZE_MAX;
    struct Entry {
        storage_detail::LogicalBlockRecord* block = nullptr;
        BlockId id() const noexcept { return block ? block->meta.id : invalid_block_id; }
        std::size_t size() const noexcept { return block ? block->meta.size : 0; }
        std::size_t previous = none, next = none;
        bool busy = false;
        std::size_t readers = 0;
        std::uint32_t ssd_hit_count = 0;
        unsigned cpu_reads = 0;
        bool hot = false;
        std::uint64_t generation = 0;
        bool prefetched = false;
        std::uint64_t prefetch_generation = 0;
        std::shared_future<bool> prefetch;
        bool prefetch_allows_eviction = true;
    };
    struct Prefetch {
        std::size_t index;
        std::uint64_t generation;
        std::promise<bool> result;
        bool allow_eviction;
        Clock::time_point submitted = Clock::now();
    };
    KVOffloadConfig config;
    TieredStorage storage;
    std::vector<Entry> entries;
    mutable std::mutex mutex;
    std::mutex writer_mutex; // 只串行修改/迁移；读取不经过这把锁。
    std::condition_variable changed;
    std::size_t head = none, tail = none;
    std::size_t protected_count = 0;
    KVOffloadStats counters;
    std::deque<Prefetch> prefetch_queue; // 与 entries 共用 mutex，避免两个锁的交叉顺序。
    std::thread prefetch_worker;
    bool stopping = false;
    std::atomic<std::uint64_t> read_ns{0}, write_ns{0}, lock_wait_ns{0}, demotion_ns{0};
    std::atomic<std::uint64_t> writer_wait_ns{0}, prefetch_queue_ns{0}, prefetch_ns{0};

    Impl(KVOffloadConfig value, std::unique_ptr<StorageBackend> cpu,
         std::unique_ptr<StorageBackend> ssd)
        : config(std::move(value)), storage(storage_config(config), std::move(cpu), {}, std::move(ssd)),
          entries(config.logical_slots) {
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (prefetch_worker.joinable())
            prefetch_worker.join();
    }
    // Entry is only a scheduler binding. The shared core owns size, identity and placement.
    storage_detail::LogicalBlockRecord* block(BlockId id) const {
        std::lock_guard<std::mutex> lock(storage.state_->mutex);
        return storage.state_->logical_blocks.at(id).get();
    }
    bool in_cpu(const Entry& entry) const {
        std::lock_guard<std::mutex> lock(storage.state_->mutex);
        return entry.block && entry.block->meta.location == StorageLocation::CPU;
    }
    void run_prefetch();
    // 需求提升与后台预取共用迁移执行；只有预取标记后续的 prefetch hit。
    bool promote(std::size_t index, std::uint64_t generation, bool allow_eviction, bool prefetched,
                 bool hot = false);
    void clear_hot(Entry& entry) noexcept {
        if (entry.hot) {
            entry.hot = false;
            --protected_count;
        }
    }
    void mark_hot(std::size_t index) noexcept {
        const auto limit = config.protect_hot_cpu ? config.cpu_slots / 2 : 0;
        auto& entry = entries[index];
        if (!limit || entry.hot)
            return;
        if (protected_count >= limit) {
            for (auto old = head; old != none; old = entries[old].next) {
                if (entries[old].hot) {
                    clear_hot(entries[old]);
                    break;
                }
            }
        }
        entry.hot = true;
        ++protected_count;
    }
    std::size_t victim(bool cold_only = false) const noexcept {
        std::size_t fallback = none;
        for (auto index = head; index != none; index = entries[index].next) {
            const auto& entry = entries[index];
            if (entry.busy || entry.readers)
                continue;
            if (!entry.hot)
                return index;
            if (fallback == none)
                fallback = index;
        }
        return cold_only ? none : fallback;
    }
    std::size_t select_victims(const std::vector<std::size_t>& slots, std::size_t needed,
                               std::array<std::size_t, 16>& selected) const {
        std::size_t count = 0;
        for (unsigned pass = 0; pass < 2 && count < needed; ++pass) {
            for (auto index = head; index != none && count < needed; index = entries[index].next) {
                const auto& entry = entries[index];
                if (entry.hot == bool(pass) && !entry.busy && !entry.readers &&
                    std::find(slots.begin(), slots.end(), index) == slots.end())
                    selected[count++] = index;
            }
        }
        return count;
    }
    // Called without mutex held. Drain every accepted task before releasing any
    // victim, including tasks accepted before submission/foreground write failure.
    // Returns with lock held; each caller keeps its original error/unlock boundary.
    void finish_demotions(std::unique_lock<std::mutex>& lock,
                           const std::array<std::size_t, 16>& victims, std::size_t count,
                           std::vector<std::future<MigrationResult>>& futures,
                           std::exception_ptr& failure, const char* incomplete) {
        for (std::size_t i = 0; i < futures.size(); ++i) {
            MigrationResult result;
            try {
                result = futures[i].get();
            } catch (...) {
                if (!failure)
                    failure = std::current_exception();
            }
            lock.lock();
            if (result.committed) {
                discard_prefetch(entries[victims[i]]);
                remove(victims[i]);
                ++counters.demotions;
            }
            lock.unlock();
            if (!failure && result.error)
                failure = result.error;
            if (!failure && (!result.committed || result.cleanup_pending))
                failure = std::make_exception_ptr(StorageBusy(incomplete));
        }
        lock.lock();
        for (std::size_t i = 0; i < count; ++i)
            entries[victims[i]].busy = false;
        changed.notify_all();
    }
    void discard_prefetch(Entry& entry) noexcept {
        if (entry.prefetched) {
            ++counters.prefetch_unused;
            entry.prefetched = false;
        }
    }
    // 预分配索引链表，发布新位置后不再分配 LRU 元数据。
    void remove(std::size_t index) noexcept {
        auto& entry = entries[index];
        if (entry.previous == none && entry.next == none && head != index)
            return;
        clear_hot(entry);
        if (entry.previous != none)
            entries[entry.previous].next = entry.next;
        else
            head = entry.next;
        if (entry.next != none)
            entries[entry.next].previous = entry.previous;
        else
            tail = entry.previous;
        entry.previous = entry.next = none;
    }
    void touch(std::size_t index) noexcept {
        if (tail == index)
            return; // 已是最新块，热点命中不重复摘链/挂链。
        const bool hot = entries[index].hot;
        remove(index);
        auto& entry = entries[index];
        entry.hot = hot;
        protected_count += int(hot);
        entry.previous = tail;
        if (tail != none)
            entries[tail].next = index;
        else
            head = index;
        tail = index;
    }
    void record_read(std::size_t index) noexcept {
        auto& entry = entries[index];
        if (in_cpu(entry)) {
            if (entry.prefetched) {
                ++counters.prefetch_hits;
                entry.prefetched = false;
            }
            touch(index);
            if (entry.cpu_reads < 2)
                ++entry.cpu_reads;
            if (entry.cpu_reads >= 2)
                mark_hot(index);
            ++counters.cpu_hits;
        } else {
            ++counters.ssd_hits;
            if (entry.ssd_hit_count < config.promote_hits)
                ++entry.ssd_hit_count;
        }
    }
    // 调用方持有 writer_mutex；元数据锁只覆盖索引操作，复制/I/O 在锁外。
    std::size_t cpu_allocation_size(std::size_t bytes) const {
        return storage.state_->cpu_tier.backend->allocation_bytes(bytes);
    }
    void write_one(std::size_t index, const void *data, std::size_t bytes = 0);
    void make_room_for_batch(const std::vector<std::size_t>& slots);
    void write_overlapped(const std::vector<std::size_t>& slots, const std::vector<void*>& buffers);

    void make_room(std::unique_lock<std::mutex> &lock, bool cold_only = false,
                   std::size_t bytes = 0) {
        const auto usage = storage.stats(StorageLocation::CPU);
        if (bytes ? bytes <= usage.limit_bytes - usage.reserved_bytes
                  : usage.reserved_bytes < usage.limit_bytes)
            return;
        if (!config.enable_ssd)
            throw std::system_error(ENOSPC, std::generic_category(), "CPU storage is full");
        std::size_t selected = none;
        for (;;) {
            selected = victim(cold_only);
            if (selected != none)
                break;
            if (cold_only)
                throw StorageBusy("No cold CPU slot available");
            if (head == none)
                throw StorageBusy("CPU capacity held by pending cleanup");
            changed.wait(lock);
        }
        auto& entry = entries[selected];
        entry.busy = true;
        const auto id = entry.id();
        lock.unlock();
        MigrationResult result;
        try {
            Timer timer{demotion_ns};
            result = storage.migrate(id, StorageLocation::SSD);
        } catch (...) {
            lock.lock();
            entry.busy = false;
            changed.notify_all();
            throw;
        }
        lock.lock();
        entry.busy = false;
        changed.notify_all();
        if (result.committed) {
            discard_prefetch(entry);
            remove(selected);
            ++counters.demotions;
        }
        if (result.error)
            std::rethrow_exception(result.error);
        if (!result.committed || result.cleanup_pending)
            throw StorageBusy("Offload demotion incomplete");
    }
};
KVOffloadCache::KVOffloadCache(KVOffloadConfig config, std::unique_ptr<StorageBackend> cpu,
                               std::unique_ptr<StorageBackend> ssd)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(cpu), std::move(ssd))) {}
KVOffloadCache::~KVOffloadCache() = default;

void KVOffloadCache::Impl::write_one(std::size_t index, const void *data, std::size_t bytes) {
    if (!bytes)
        bytes = config.payload_bytes;
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    {
        Timer waiting{lock_wait_ns};
        lock.lock();
    }
    auto& entry = entries.at(index);
    changed.wait(lock, [&] { return !entry.busy; });
    entry.busy = true;
    BlockId fresh = invalid_block_id;
    try {
        // Reserve the slot before draining existing readers so new readers
        // cannot starve this writer by repeatedly acquiring the same slot.
        changed.wait(lock, [&] { return !entry.readers; });
        if (entry.generation == UINT64_MAX)
            throw std::overflow_error("KV slot generation exhausted");
        const auto old = entry.id();
        const bool overwrite = in_cpu(entry) &&
            cpu_allocation_size(entry.size()) == cpu_allocation_size(bytes);
        if (!overwrite) {
            const auto capacity = cpu_allocation_size(bytes);
            for (;;) {
                const auto usage = storage.stats(StorageLocation::CPU);
                if (capacity <= usage.limit_bytes - usage.reserved_bytes)
                    break;
                // A resize cannot destroy acknowledged data to free its own
                // capacity. Preserve the source when no replacement fits.
                if (in_cpu(entry) && victim(false) == none) {
                    if (!config.enable_ssd)
                        throw std::system_error(ENOSPC, std::generic_category(),
                                                "No capacity for transactional KV resize");
                    // Preserve the old acknowledged contents on SSD before
                    // reclaiming their CPU capacity. The slot stays busy.
                    lock.unlock();
                    auto result = storage.migrate(old, StorageLocation::SSD);
                    lock.lock();
                    if (result.committed) {
                        discard_prefetch(entry);
                        remove(index);
                        ++counters.demotions;
                    }
                    if (result.error) std::rethrow_exception(result.error);
                    if (!result.committed || result.cleanup_pending)
                        throw StorageBusy("KV resize source demotion incomplete");
                    continue;
                }
                make_room(lock, false, capacity);
            }
        }
        lock.unlock();

        // CPU 命中直接覆盖；新槽先写完，再删除旧 SSD 内容并发布新 ID。
        // 两条路径共用一次 WRITE Lease，不为覆盖制造额外迁移或内存申请。
        const auto id = overwrite ? old : (fresh = storage.create(bytes));
        {
            auto lease = storage.acquire_cpu_write(id, bytes);
            std::memcpy(lease.writable_data(), data, bytes);
            lease.finish();
        }
        if (fresh && old && !storage.erase(old))
            throw StorageBusy("Old offload slot is busy");

        lock.lock();
        if (!old)
            ++counters.populated_slots;
        discard_prefetch(entry);
        ++entry.generation;
        entry.block = block(id);
        entry.ssd_hit_count = 0;
        entry.cpu_reads = 0;
        clear_hot(entry);
        touch(index);
        ++counters.writes;
    } catch (...) {
        // 尚未发布的新块由 Storage 继续保活，即使清理失败也不会丢失资源记录。
        if (fresh) {
            if (lock.owns_lock())
                lock.unlock();
            try {
                if (!storage.erase(fresh))
                    std::fputs("KV unpublished block cleanup is busy; Storage retains ownership\n",
                               stderr);
            } catch (const std::exception& error) {
                std::fprintf(stderr,
                             "KV unpublished block cleanup failed; Storage retains ownership: %s\n",
                             error.what());
            } catch (...) {
                std::fputs("KV unpublished block cleanup failed; Storage retains ownership\n",
                           stderr);
            }
        }
        if (!lock.owns_lock())
            lock.lock();
        entry.busy = false;
        changed.notify_all();
        throw;
    }
    entry.busy = false;
    changed.notify_all();
}

void KVOffloadCache::write(std::size_t index, const void* data) {
    if (!data)
        throw std::invalid_argument("Missing offload input");
    Timer timer{impl_->write_ns};
    std::unique_lock<std::mutex> writer(impl_->writer_mutex, std::defer_lock);
    {
        Timer waiting{impl_->writer_wait_ns};
        writer.lock();
    }
    impl_->write_one(index, data);
}
void KVOffloadCache::read(std::size_t index, void* data) {
    (void)read(index, data, impl_->config.payload_bytes);
}
std::size_t KVOffloadCache::read(std::size_t index, void* data, std::size_t capacity) {
    if (!data)
        throw std::invalid_argument("Missing offload output");
    Timer timer{impl_->read_ns};
    std::unique_lock<std::mutex> lock(impl_->mutex, std::defer_lock);
    {
        Timer waiting{impl_->lock_wait_ns};
        lock.lock();
    }
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.id())
        throw std::logic_error("Offload slot has not been written");
    const auto bytes = entry.size();
    if (capacity < bytes)
        throw std::length_error("Offload output is smaller than its payload");
    const auto id = entry.id();
    ++entry.readers; // 保护 id 到 read_to_host 的 lease 建立和完成，避免覆盖/删除窗口。
    lock.unlock();
    try {
        impl_->storage.read_to_host(id, data, bytes);
    } catch (...) {
        lock.lock();
        --entry.readers;
        impl_->changed.notify_all();
        throw;
    }
    lock.lock();
    --entry.readers;
    impl_->changed.notify_all();
    impl_->record_read(index);
    if (impl_->in_cpu(entry) || !impl_->config.promote_hits ||
        entry.ssd_hit_count < impl_->config.promote_hits)
        return bytes;
    const auto generation = entry.generation;
    lock.unlock();
    // 数据已经读出；机会式提升不淘汰热块，失败也不影响本次读取结果。
    try {
        impl_->promote(index, generation, impl_->config.promote_evict, false,
                       impl_->config.protect_hot_cpu);
    } catch (const StorageBusy&) {
        return bytes;
    } // 可选提升遇到背压，已完成的需求读取仍成功。
    catch (const std::bad_alloc&) {
        return bytes;
    } // 容量耗尽同样只跳过提升。
    catch (const std::exception& error) {
        std::fprintf(stderr, "KV demand read completed; optional promotion failed: %s\n",
                     error.what());
    } catch (...) {
        std::fputs("KV demand read completed; optional promotion failed\n", stderr);
    }
    return bytes;
}
void KVOffloadCache::Impl::make_room_for_batch(const std::vector<std::size_t>& slots) {
    if (!config.enable_ssd || config.write_concurrency == 1)
        return;
    std::unique_lock<std::mutex> lock(mutex);
    const auto usage = storage.stats(StorageLocation::CPU);
    const auto slot_bytes = usage.limit_bytes / config.cpu_slots;
    const auto free = (usage.limit_bytes - usage.reserved_bytes) / slot_bytes;
    std::size_t incoming = 0;
    for (auto index : slots)
        incoming += !in_cpu(entries[index]);
    auto needed = incoming > free ? incoming - free : 0;
    if (!needed)
        return;
    std::array<std::size_t, 16> victims{};
    std::vector<std::future<MigrationResult>> futures;
    futures.reserve(config.write_concurrency);
    while (needed) {
        // 每波重新选择并保护源；不会把上一波 I/O 期间新进入的读者当作空闲。
        const auto count =
            select_victims(slots, std::min(needed, config.write_concurrency), victims);
        if (!count)
            return; // 剩余忙槽由逐槽写入等待，不阻断读者。
        for (std::size_t i = 0; i < count; ++i)
            entries[victims[i]].busy = true;
        lock.unlock();
        Timer timer{demotion_ns};
        futures.clear();
        std::exception_ptr failure;
        try {
            for (std::size_t i = 0; i < count; ++i)
                futures.push_back(
                    storage.migrate_async(entries[victims[i]].id(), StorageLocation::SSD));
        } catch (...) {
            failure = std::current_exception();
        }
        finish_demotions(lock, victims, count, futures, failure, "Batch demotion incomplete");
        if (failure)
            std::rethrow_exception(failure);
        needed -= count;
    }
}

void KVOffloadCache::Impl::write_overlapped(const std::vector<std::size_t>& slots,
                                            const std::vector<void*>& buffers) {
    std::size_t next = 0;
    std::vector<std::future<MigrationResult>> futures;
    futures.reserve(config.write_concurrency);
    while (next < slots.size()) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto usage = storage.stats(StorageLocation::CPU);
        const auto slot_bytes = usage.limit_bytes / config.cpu_slots;
        auto free = (usage.limit_bytes - usage.reserved_bytes) / slot_bytes;
        std::size_t incoming = 0;
        for (auto i = next; i < slots.size(); ++i)
            incoming += !in_cpu(entries[slots[i]]);
        const auto needed = incoming > free ? incoming - free : 0;
        std::array<std::size_t, 16> victims{};
        const auto count =
            select_victims(slots, std::min(needed, config.write_concurrency), victims);
        if (!count) {
            lock.unlock();
            // No independent victim: the ordinary path handles full capacity
            // and requested-slot replacement, retaining its waiting semantics.
            write_one(slots[next], buffers[next]);
            ++next;
            continue;
        }
        for (std::size_t i = 0; i < count; ++i)
            entries[victims[i]].busy = true;
        lock.unlock();
        futures.clear();
        std::exception_ptr failure;
        Timer timer{demotion_ns};
        try {
            for (std::size_t i = 0; i < count; ++i)
                futures.push_back(
                    storage.migrate_async(entries[victims[i]].id(), StorageLocation::SSD));
            while (next < slots.size()) {
                lock.lock();
                const bool overwrite = in_cpu(entries[slots[next]]);
                lock.unlock();
                if (!overwrite && !free)
                    break;
                write_one(slots[next], buffers[next]);
                if (!overwrite)
                    --free;
                lock.lock();
                ++counters.overlapped_writes;
                lock.unlock();
                ++next;
            }
        } catch (...) {
            failure = std::current_exception();
        }
        finish_demotions(lock, victims, count, futures, failure, "Overlapped demotion incomplete");
        lock.unlock();
        if (failure)
            std::rethrow_exception(failure);
    }
}

void KVOffloadCache::write_many(const std::vector<std::size_t>& slots,
                                const std::vector<void*>& buffers) {
    write_many(slots, buffers, std::vector<std::size_t>(slots.size(), impl_->config.payload_bytes));
}
void KVOffloadCache::write_many(const std::vector<std::size_t> &slots,
                                const std::vector<void *> &buffers,
                                const std::vector<std::size_t> &sizes) {
    if (slots.empty() || slots.size() > 64 || slots.size() != buffers.size() ||
        slots.size() != sizes.size())
        throw std::invalid_argument("Invalid offload write batch");
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (!buffers[i] || !sizes[i] || sizes[i] > impl_->config.payload_bytes ||
            (!impl_->config.variable_payloads && sizes[i] != impl_->config.payload_bytes) ||
            slots[i] >= impl_->entries.size() ||
            std::find(slots.begin(), slots.begin() + i, slots[i]) != slots.begin() + i)
            throw std::invalid_argument("Invalid/duplicate offload write slot");
    }
    Timer timer{impl_->write_ns};
    // 一批只获取一次 writer_mutex，预降级和各槽发布之间不插入其他写入。
    // 每槽仍独立提交；后项失败不回滚已完成项。读取继续使用短元数据锁。
    std::unique_lock<std::mutex> writer(impl_->writer_mutex, std::defer_lock);
    {
        Timer waiting{impl_->writer_wait_ns};
        writer.lock();
    }
    if (std::any_of(sizes.begin(), sizes.end(),
                    [&](auto size) { return size != impl_->config.payload_bytes; })) {
        for (std::size_t i = 0; i < slots.size(); ++i)
            impl_->write_one(slots[i], buffers[i], sizes[i]);
        return;
    }
    // Overlap estimates free space in fixed slots. A variable-size replacement
    // can need the busy victim whose future this same writer must consume.
    if (impl_->config.enable_ssd && impl_->config.overlap_writeback && !impl_->config.variable_payloads) {
        impl_->write_overlapped(slots, buffers);
        return;
    }
    impl_->make_room_for_batch(slots);
    for (std::size_t i = 0; i < slots.size(); ++i)
        impl_->write_one(slots[i], buffers[i]);
}
void KVOffloadCache::read_many(const std::vector<std::size_t>& slots,
                               const std::vector<void*>& buffers) {
    if (slots.empty() || slots.size() > 64 || slots.size() != buffers.size())
        throw std::invalid_argument("Invalid offload read batch");
    Timer timer{impl_->read_ns};
    std::vector<BlockId> ids;
    std::vector<std::size_t> sizes;
    ids.reserve(slots.size());
    sizes.reserve(slots.size());
    std::unique_lock<std::mutex> lock(impl_->mutex);
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (!buffers[i] ||
            std::find(slots.begin(), slots.begin() + i, slots[i]) != slots.begin() + i)
            throw std::invalid_argument("Null/duplicate offload read batch");
        (void)impl_->entries.at(slots[i]);
    }
    impl_->changed.wait(lock, [&] {
        return std::all_of(slots.begin(), slots.end(),
                           [&](auto index) { return !impl_->entries[index].busy; });
    });
    for (auto index : slots) {
        if (!impl_->entries[index].id())
            throw std::logic_error("Offload batch slot has not been written");
        ids.push_back(impl_->entries[index].id());
        sizes.push_back(impl_->entries[index].size());
    }
    for (auto index : slots)
        ++impl_->entries[index].readers;
    lock.unlock();
    std::exception_ptr failure;
    try {
        impl_->storage.read_to_host_many(ids, buffers, sizes);
    } catch (...) {
        failure = std::current_exception();
    }
    lock.lock();
    std::array<std::pair<std::size_t, std::uint64_t>, 64> hot{};
    std::size_t hot_count = 0;
    for (auto index : slots) {
        auto& entry = impl_->entries[index];
        --entry.readers;
        if (!failure)
            impl_->record_read(index);
        if (!failure && !impl_->in_cpu(entry) && impl_->config.promote_hits && entry.ssd_hit_count >= impl_->config.promote_hits &&
            hot_count < impl_->config.batch_promotions)
            hot[hot_count++] = {index, entry.generation};
    }
    impl_->changed.notify_all();
    if (failure)
        std::rethrow_exception(failure);
    lock.unlock();
    // Demand data is already complete. Revalidate generation under the writer
    // lock; a lost race or lack of free capacity only skips optional retention.
    for (std::size_t i = 0; i < hot_count; ++i) {
        try {
            impl_->promote(hot[i].first, hot[i].second, false, false,
                           impl_->config.protect_hot_cpu);
        } catch (const StorageBusy&) {
            continue;
        } catch (const std::bad_alloc&) {
            continue;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "KV batch read completed; optional promotion failed: %s\n",
                         error.what());
        } catch (...) {
            std::fputs("KV batch read completed; optional promotion failed\n", stderr);
        }
    }
}
void KVOffloadCache::Impl::run_prefetch() {
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return stopping || !prefetch_queue.empty(); });
        if (prefetch_queue.empty())
            return;
        auto job = std::move(prefetch_queue.front());
        prefetch_queue.pop_front();
        prefetch_queue_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - job.submitted)
                .count(),
            std::memory_order_relaxed);
        lock.unlock();
        bool resident = false;
        std::exception_ptr error;
        try {
            Timer timer{prefetch_ns};
            resident = promote(job.index, job.generation, job.allow_eviction, true);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        auto& entry = entries[job.index];
        if (entry.prefetch_generation == job.generation)
            entry.prefetch = {};
        --counters.prefetch_inflight;
        if (error)
            ++counters.prefetch_failed;
        else if (resident)
            ++counters.prefetch_completed;
        else
            ++counters.prefetch_skipped;
        lock.unlock();
        // 完成结果发布前已归还队列额度；future 丢弃也不取消任务或泄漏预算。
        if (error)
            job.result.set_exception(error);
        else
            job.result.set_value(resident);
    }
}

bool KVOffloadCache::Impl::promote(std::size_t index, std::uint64_t generation, bool allow_eviction,
                                   bool prefetched, bool hot) {
    std::unique_lock<std::mutex> writer(writer_mutex, std::defer_lock);
    if (!allow_eviction) {
        if (!writer.try_lock())
            return false;
    } else {
        Timer waiting{writer_wait_ns};
        writer.lock();
    }
    std::unique_lock<std::mutex> lock(mutex);
    auto& entry = entries[index];
    if (!entry.id() || entry.generation != generation)
        return false;
    if (in_cpu(entry)) {
        touch(index);
        return true;
    }
    // 需求读取优先；预取不等待正在读取的源，也不等待所有 victim 的读者退场。
    if (entry.busy || entry.readers)
        return false;
    const auto usage = storage.stats(StorageLocation::CPU);
    const auto capacity = cpu_allocation_size(entry.size());
    const bool needs_room = capacity > usage.limit_bytes - usage.reserved_bytes;
    if (needs_room) {
        if (!allow_eviction && !hot)
            return false;
        if (victim(hot && !allow_eviction) == none)
            return false;
    }
    entry.busy = true;
    MigrationResult result;
    try {
        for (;;) {
            const auto available = storage.stats(StorageLocation::CPU);
            if (capacity <= available.limit_bytes - available.reserved_bytes)
                break;
            make_room(lock, hot && !allow_eviction, capacity);
        }
        lock.unlock();
        result = storage.migrate(entry.id(), StorageLocation::CPU);
        lock.lock();
        if (result.committed) {
            entry.ssd_hit_count = 0;
            entry.prefetched = prefetched;
            touch(index);
            if (hot) {
                mark_hot(index);
                ++counters.hot_promotions;
            }
            ++counters.promotions;
        }
        if (result.error)
            std::rethrow_exception(result.error);
        if (!result.committed || result.cleanup_pending)
            throw StorageBusy("Prefetch incomplete");
    } catch (...) {
        if (!lock.owns_lock())
            lock.lock();
        entry.busy = false;
        changed.notify_all();
        throw;
    }
    entry.busy = false;
    changed.notify_all();
    return true;
}

std::shared_future<bool> KVOffloadCache::prefetch_async(std::size_t index) {
    return submit_prefetch(index, true);
}
bool KVOffloadCache::prefetch(std::size_t index, bool allow_eviction) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.block)
        return false;
    const auto generation = entry.generation;
    lock.unlock();
    return impl_->promote(index, generation, allow_eviction, true);
}

KVOffloadMeta KVOffloadCache::metadata(std::size_t index) const {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.block)
        return {};
    std::lock_guard<std::mutex> core(impl_->storage.state_->mutex);
    const auto& record = *entry.block;
    return {record.meta.id, record.meta.size, record.meta.location, record.meta.version,
            record.meta.location == StorageLocation::SSD ||
                (record.ssd_copy.backend && !record.ssd_copy.pending &&
                 record.ssd_copy_version == record.meta.version)};
}
std::shared_future<bool> KVOffloadCache::prefetch_if_space_async(std::size_t index) {
    return submit_prefetch(index, false);
}
std::shared_future<bool> KVOffloadCache::submit_prefetch(std::size_t index, bool allow_eviction) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    if (entry.prefetch.valid() && entry.prefetch_generation == entry.generation) {
        if (!allow_eviction && entry.prefetch_allows_eviction) {
            return ready_future(false);
        }
        ++impl_->counters.prefetch_deduplicated;
        return entry.prefetch;
    }
    if (!entry.block || (!entry.busy && impl_->in_cpu(entry))) {
        return ready_future(bool(entry.block));
    }
    if (!allow_eviction) {
        const auto usage = impl_->storage.stats(StorageLocation::CPU);
        if (usage.reserved_bytes >= usage.limit_bytes) {
            ++impl_->counters.prefetch_skipped;
            return ready_future(false);
        }
    }
    if (impl_->stopping || impl_->counters.prefetch_inflight >= impl_->config.prefetch_depth) {
        ++impl_->counters.prefetch_rejected;
        throw StorageBusy("KV prefetch queue is full or disabled");
    }
    // 先分配任务、future 和 worker，再登记额度；分配失败不留下虚假的 pending。
    Impl::Prefetch job{index, entry.generation, {}, allow_eviction};
    auto result = job.result.get_future().share();
    if (!impl_->prefetch_worker.joinable())
        impl_->prefetch_worker = std::thread([state = impl_.get()] { state->run_prefetch(); });
    impl_->prefetch_queue.push_back(std::move(job));
    entry.prefetch_generation = entry.generation;
    entry.prefetch = result;
    entry.prefetch_allows_eviction = allow_eviction;
    ++impl_->counters.prefetch_submitted;
    ++impl_->counters.prefetch_inflight;
    impl_->counters.prefetch_peak =
        std::max(impl_->counters.prefetch_peak, impl_->counters.prefetch_inflight);
    impl_->changed.notify_all();
    return result;
}

bool KVOffloadCache::erase(std::size_t index) {
    std::unique_lock<std::mutex> writer(impl_->writer_mutex);
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.id())
        return false;
    if (entry.generation == UINT64_MAX)
        throw std::overflow_error("KV slot generation exhausted");
    entry.busy = true;
    try {
        impl_->changed.wait(lock, [&] { return !entry.readers; });
        lock.unlock();
        if (!impl_->storage.erase(entry.id()))
            throw StorageBusy("KV slot is still in use");
    } catch (...) {
        if (!lock.owns_lock())
            lock.lock();
        entry.busy = false;
        impl_->changed.notify_all();
        throw;
    }
    lock.lock();
    impl_->discard_prefetch(entry);
    impl_->remove(index);
    entry.block = nullptr;
    entry.ssd_hit_count = 0;
    ++entry.generation;
    entry.busy = false;
    --impl_->counters.populated_slots;
    ++impl_->counters.invalidations;
    impl_->changed.notify_all();
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
    auto* backend = dynamic_cast<SSDStorageBackend*>(
        impl_->storage.state_->tier(StorageLocation::SSD).backend.get());
    return backend ? backend->allocate_dma_buffer(bytes) : std::shared_ptr<SSDDmaBuffer>{};
}

} // namespace infer
