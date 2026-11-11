#pragma once

#include "infer/storage/kv_offload_cache.h"
#include "infer/storage/tiered_storage.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace infer {
namespace storage_detail {
using OffloadClock = std::chrono::steady_clock;
struct OffloadTimer {
    std::atomic<std::uint64_t>& total;
    OffloadClock::time_point start = OffloadClock::now();
    ~OffloadTimer() {
        total.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(OffloadClock::now() - start)
                .count(),
            std::memory_order_relaxed);
    }
};
} // namespace storage_detail

struct KVOffloadCache::Impl {
    static constexpr std::size_t none = SIZE_MAX;
    struct Entry {
        const BlockMeta* block = nullptr;
        BlockId id() const noexcept { return block ? block->id : invalid_block_id; }
        std::size_t size() const noexcept { return block ? block->size : 0; }
        std::size_t previous = none, next = none;
        bool busy = false;
        std::size_t readers = 0;
        std::uint32_t ssd_hit_count = 0;
        unsigned cpu_reads = 0;
        bool hot = false;
        std::uint64_t generation = 0;
        bool prefetched = false;
        std::shared_future<bool> prefetch;
        bool prefetch_allows_eviction = true;
    };
    struct Prefetch {
        std::size_t index;
        std::uint64_t generation;
        std::promise<bool> result;
        bool allow_eviction;
        storage_detail::OffloadClock::time_point submitted = storage_detail::OffloadClock::now();
    };
    KVOffloadConfig config;
    TieredStorage storage;
    std::vector<Entry> entries;
    mutable std::mutex mutex;
    std::mutex writer_mutex; // 只串行修改/迁移；读取不经过这把锁。
    std::condition_variable changed;
    std::size_t lru_head = none, lru_tail = none;
    std::size_t protected_count = 0;
    KVOffloadStats counters;
    std::deque<Prefetch> prefetch_queue; // 与 entries 共用 mutex，避免两个锁的交叉顺序。
    std::thread prefetch_worker;
    bool stopping = false;
    std::atomic<std::uint64_t> read_ns{0}, write_ns{0}, lock_wait_ns{0}, demotion_ns{0};
    std::atomic<std::uint64_t> writer_wait_ns{0}, prefetch_queue_ns{0}, prefetch_ns{0};

    Impl(KVOffloadConfig value, std::unique_ptr<StorageBackend> cpu,
         std::unique_ptr<StorageBackend> ssd);
    ~Impl();

    // 调用方持有 writer_mutex；元数据锁只覆盖索引操作，复制/I/O 在锁外。
    void write_one(std::size_t index, const void* data, std::size_t bytes);
    void write_batch(const std::vector<std::size_t>& slots, const std::vector<void*>& buffers,
                     bool overlap);
    void record_read(std::size_t index) noexcept;

    void run_prefetch();
    // 需求提升与后台预取共用迁移执行；只有预取标记后续的 prefetch hit。
    bool promote(std::size_t index, std::uint64_t generation, bool allow_eviction, bool prefetched,
                 bool hot = false);
    void discard_prefetch(Entry& entry) noexcept {
        if (entry.prefetched) {
            ++counters.prefetch_unused;
            entry.prefetched = false;
        }
    }

    std::size_t victim(bool cold_only) const noexcept;
    std::size_t select_victims(const std::vector<std::size_t>& slots, std::size_t needed,
                               std::array<std::size_t, 16>& selected) const;
    // Every migration publishes its placement before reporting cleanup errors.
    // Apply that committed placement even if the caller must subsequently throw.
    void apply_migration(std::size_t index, const MigrationResult& result, StorageLocation target,
                         bool prefetched = false, bool hot = false);
    static void require_complete(const MigrationResult& result);
    // Caller holds writer_mutex and protects the slot before releasing mutex.
    void demote(std::unique_lock<std::mutex>& lock, std::size_t index);
    void make_room(std::unique_lock<std::mutex>& lock, bool cold_only, std::size_t bytes);

    // Only CPU residents are linked. Readers prevent migration; writer_mutex
    // serializes placement changes, and busy hides an in-flight list update.
    bool is_cpu_resident(const Entry& entry) const noexcept {
        return entry.previous != none || entry.next != none ||
               (lru_head != none && &entries[lru_head] == &entry);
    }
    void clear_hot(Entry& entry) noexcept {
        if (entry.hot) {
            entry.hot = false;
            --protected_count;
        }
    }
    void mark_hot(std::size_t index) noexcept {
        const auto limit = config.protect_hot_cpu ? config.cpu_slots / 2 : 0;
        auto& entry = entries[index];
        if (!limit || entry.hot) return;
        if (protected_count >= limit) {
            for (auto old = lru_head; old != none; old = entries[old].next) {
                if (entries[old].hot) {
                    clear_hot(entries[old]);
                    break;
                }
            }
        }
        entry.hot = true;
        ++protected_count;
    }
    // 预分配索引链表，发布新位置后不再分配 LRU 元数据。
    void unlink_cpu_slot(std::size_t index) noexcept {
        auto& entry = entries[index];
        if (entry.previous == none && entry.next == none && lru_head != index) return;
        clear_hot(entry);
        if (entry.previous != none)
            entries[entry.previous].next = entry.next;
        else
            lru_head = entry.next;
        if (entry.next != none)
            entries[entry.next].previous = entry.previous;
        else
            lru_tail = entry.previous;
        entry.previous = entry.next = none;
    }
    void touch_cpu_slot(std::size_t index) noexcept {
        if (lru_tail == index) return; // 已是最新块，热点命中不重复摘链/挂链。
        const bool hot = entries[index].hot;
        unlink_cpu_slot(index);
        auto& entry = entries[index];
        entry.hot = hot;
        protected_count += int(hot);
        entry.previous = lru_tail;
        if (lru_tail != none)
            entries[lru_tail].next = index;
        else
            lru_head = index;
        lru_tail = index;
    }
};
} // namespace infer
