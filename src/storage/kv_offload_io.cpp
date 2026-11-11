#include "kv_offload_state.h"
#include "scope_exit.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <system_error>

namespace infer {
using storage_detail::OffloadTimer;

void KVOffloadCache::write(std::size_t index, const void* data) {
    if (!data) throw std::invalid_argument("Missing offload input");
    OffloadTimer timer{impl_->write_ns};
    std::unique_lock<std::mutex> writer(impl_->writer_mutex, std::defer_lock);
    {
        OffloadTimer waiting{impl_->writer_wait_ns};
        writer.lock();
    }
    impl_->write_one(index, data, impl_->config.payload_bytes);
}

void KVOffloadCache::write_many(const std::vector<std::size_t>& slots,
                                const std::vector<void*>& buffers) {
    write_many(slots, buffers, std::vector<std::size_t>(slots.size(), impl_->config.payload_bytes));
}

void KVOffloadCache::write_many(const std::vector<std::size_t>& slots,
                                const std::vector<void*>& buffers,
                                const std::vector<std::size_t>& sizes) {
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
    OffloadTimer timer{impl_->write_ns};
    // 一批只获取一次 writer_mutex，预降级和各槽发布之间不插入其他写入。
    // 每槽仍独立提交；后项失败不回滚已完成项。读取继续使用短元数据锁。
    std::unique_lock<std::mutex> writer(impl_->writer_mutex, std::defer_lock);
    {
        OffloadTimer waiting{impl_->writer_wait_ns};
        writer.lock();
    }
    // Overlap estimates free space in fixed slots. A variable-size replacement
    // can need the busy victim whose future this same writer must consume.
    const bool overlap = impl_->config.overlap_writeback && !impl_->config.variable_payloads;
    if (impl_->config.enable_ssd && (overlap || impl_->config.write_concurrency > 1) &&
        std::all_of(sizes.begin(), sizes.end(),
                    [&](auto size) { return size == impl_->config.payload_bytes; })) {
        impl_->write_batch(slots, buffers, overlap);
        return;
    }
    for (std::size_t i = 0; i < slots.size(); ++i)
        impl_->write_one(slots[i], buffers[i], sizes[i]);
}

void KVOffloadCache::Impl::write_one(std::size_t index, const void* data, std::size_t bytes) {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    {
        OffloadTimer waiting{lock_wait_ns};
        lock.lock();
    }
    auto& entry = entries.at(index);
    changed.wait(lock, [&] { return !entry.busy; });
    entry.busy = true;
    storage_detail::ScopeExit release([&] {
        if (!lock.owns_lock()) lock.lock();
        entry.busy = false;
        changed.notify_all();
    });
    BlockId fresh = invalid_block_id;
    try {
        // Reserve the slot before draining existing readers so new readers
        // cannot starve this writer by repeatedly acquiring the same slot.
        changed.wait(lock, [&] { return !entry.readers; });
        if (entry.generation == UINT64_MAX)
            throw std::overflow_error("KV slot generation exhausted");
        const auto old = entry.id();
        const bool overwrite =
            is_cpu_resident(entry) &&
            storage.cpu_allocation_size(entry.size()) == storage.cpu_allocation_size(bytes);
        if (!overwrite) {
            const auto capacity = storage.cpu_allocation_size(bytes);
            for (;;) {
                const auto usage = storage.stats(StorageLocation::CPU);
                if (capacity <= usage.limit_bytes - usage.reserved_bytes) break;
                // A resize cannot destroy acknowledged data to free its own
                // capacity. Preserve the source when no replacement fits.
                if (is_cpu_resident(entry) && victim(false) == none) {
                    if (!config.enable_ssd)
                        throw std::system_error(ENOSPC, std::generic_category(),
                                                "No capacity for transactional KV resize");
                    // Preserve the old acknowledged contents on SSD before
                    // reclaiming their CPU capacity. The slot stays busy.
                    demote(lock, index);
                    continue;
                }
                make_room(lock, false, capacity);
            }
        }
        lock.unlock();

        // CPU 命中直接覆盖；新槽先写完，再删除旧 SSD 内容并发布新 ID。
        // 两条路径共用一次 WRITE Lease，不为覆盖制造额外迁移或内存申请。
        if (!overwrite) fresh = storage.create(bytes);
        const auto id = overwrite ? old : fresh;
        {
            auto lease = storage.acquire_cpu_write(id, bytes);
            std::memcpy(lease.writable_data(), data, bytes);
            lease.finish();
        }
        if (fresh && old && !storage.erase(old)) throw StorageBusy("Old offload slot is busy");

        lock.lock();
        if (!old) ++counters.populated_slots;
        discard_prefetch(entry);
        ++entry.generation;
        entry.block = storage.bind_metadata(id);
        entry.prefetch = {};
        entry.ssd_hit_count = 0;
        entry.cpu_reads = 0;
        clear_hot(entry);
        touch_cpu_slot(index);
        ++counters.writes;
    } catch (...) {
        // 尚未发布的新块由 Storage 继续保活，即使清理失败也不会丢失资源记录。
        if (fresh) {
            if (lock.owns_lock()) lock.unlock();
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
        throw;
    }
}

void KVOffloadCache::Impl::write_batch(const std::vector<std::size_t>& slots,
                                       const std::vector<void*>& buffers, bool overlap) {
    std::size_t next = 0;
    std::array<std::size_t, 16> victims{};
    std::array<std::future<MigrationResult>, 16> futures;
    while (next < slots.size()) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto usage = storage.stats(StorageLocation::CPU);
        const auto slot_bytes = usage.limit_bytes / config.cpu_slots;
        auto free = (usage.limit_bytes - usage.reserved_bytes) / slot_bytes;
        std::size_t incoming = 0;
        for (auto i = next; i < slots.size(); ++i)
            incoming += !is_cpu_resident(entries[slots[i]]);
        const auto needed = incoming > free ? incoming - free : 0;
        const auto count =
            select_victims(slots, std::min(needed, config.write_concurrency), victims);
        if (!count) {
            lock.unlock();
            // Ordinary writes retain waiting and requested-slot replacement semantics.
            do {
                write_one(slots[next], buffers[next], config.payload_bytes);
                ++next;
            } while (!overlap && next < slots.size());
            continue;
        }
        for (std::size_t i = 0; i < count; ++i)
            entries[victims[i]].busy = true;
        storage_detail::ScopeExit release([&] {
            if (!lock.owns_lock()) lock.lock();
            for (std::size_t i = 0; i < count; ++i)
                entries[victims[i]].busy = false;
            changed.notify_all();
        });
        lock.unlock();
        std::size_t submitted = 0;
        std::exception_ptr failure;
        OffloadTimer timer{demotion_ns};
        try {
            for (; submitted < count; ++submitted)
                futures[submitted] =
                    storage.migrate_async(entries[victims[submitted]].id(), StorageLocation::SSD);
            while (overlap && next < slots.size()) {
                lock.lock();
                const bool overwrite = is_cpu_resident(entries[slots[next]]);
                lock.unlock();
                if (!overwrite && !free) break;
                write_one(slots[next], buffers[next], config.payload_bytes);
                if (!overwrite) --free;
                lock.lock();
                ++counters.overlapped_writes;
                lock.unlock();
                ++next;
            }
        } catch (...) {
            failure = std::current_exception();
        }
        // Drain all accepted tasks before releasing any victim, even after a
        // submission or foreground write failed. Keep the first failure.
        for (std::size_t i = 0; i < submitted; ++i) {
            try {
                const auto result = futures[i].get();
                lock.lock();
                apply_migration(victims[i], result, StorageLocation::SSD);
                lock.unlock();
                require_complete(result);
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
}

void KVOffloadCache::read(std::size_t index, void* data) {
    (void)read(index, data, impl_->config.payload_bytes);
}

std::size_t KVOffloadCache::read(std::size_t index, void* data, std::size_t capacity) {
    if (!data) throw std::invalid_argument("Missing offload output");
    OffloadTimer timer{impl_->read_ns};
    std::unique_lock<std::mutex> lock(impl_->mutex, std::defer_lock);
    {
        OffloadTimer waiting{impl_->lock_wait_ns};
        lock.lock();
    }
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.id()) throw std::logic_error("Offload slot has not been written");
    const auto bytes = entry.size();
    if (capacity < bytes) throw std::length_error("Offload output is smaller than its payload");
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
    if (impl_->is_cpu_resident(entry) || !impl_->config.promote_hits ||
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

void KVOffloadCache::read_many(const std::vector<std::size_t>& slots,
                               const std::vector<void*>& buffers) {
    if (slots.empty() || slots.size() > 64 || slots.size() != buffers.size())
        throw std::invalid_argument("Invalid offload read batch");
    OffloadTimer timer{impl_->read_ns};
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
        if (!failure) impl_->record_read(index);
        if (!failure && !impl_->is_cpu_resident(entry) && impl_->config.promote_hits &&
            entry.ssd_hit_count >= impl_->config.promote_hits &&
            hot_count < impl_->config.batch_promotions)
            hot[hot_count++] = {index, entry.generation};
    }
    impl_->changed.notify_all();
    if (failure) std::rethrow_exception(failure);
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

void KVOffloadCache::Impl::record_read(std::size_t index) noexcept {
    auto& entry = entries[index];
    if (!is_cpu_resident(entry)) {
        ++counters.ssd_hits;
        if (entry.ssd_hit_count < config.promote_hits) ++entry.ssd_hit_count;
        return;
    }
    if (entry.prefetched) {
        ++counters.prefetch_hits;
        entry.prefetched = false;
    }
    touch_cpu_slot(index);
    if (entry.cpu_reads < 2) ++entry.cpu_reads;
    if (entry.cpu_reads >= 2) mark_hot(index);
    ++counters.cpu_hits;
}
} // namespace infer
