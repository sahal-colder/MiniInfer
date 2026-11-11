#include "kv_offload_state.h"
#include "scope_exit.h"

#include <algorithm>

namespace infer {
using storage_detail::OffloadClock;
using storage_detail::OffloadTimer;

namespace {
std::shared_future<bool> ready_future(bool value) {
    std::promise<bool> result;
    result.set_value(value);
    return result.get_future().share();
}
} // namespace

std::shared_future<bool> KVOffloadCache::prefetch_async(std::size_t index) {
    return submit_prefetch(index, true);
}

bool KVOffloadCache::prefetch(std::size_t index, bool allow_eviction) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    impl_->changed.wait(lock, [&] { return !entry.busy; });
    if (!entry.block) return false;
    const auto generation = entry.generation;
    lock.unlock();
    return impl_->promote(index, generation, allow_eviction, true);
}

std::shared_future<bool> KVOffloadCache::prefetch_if_space_async(std::size_t index) {
    return submit_prefetch(index, false);
}

std::shared_future<bool> KVOffloadCache::submit_prefetch(std::size_t index, bool allow_eviction) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& entry = impl_->entries.at(index);
    if (entry.prefetch.valid()) {
        if (!allow_eviction && entry.prefetch_allows_eviction) {
            return ready_future(false);
        }
        ++impl_->counters.prefetch_deduplicated;
        return entry.prefetch;
    }
    if (!entry.block || (!entry.busy && impl_->is_cpu_resident(entry))) {
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
    entry.prefetch = result;
    entry.prefetch_allows_eviction = allow_eviction;
    ++impl_->counters.prefetch_submitted;
    ++impl_->counters.prefetch_inflight;
    impl_->counters.prefetch_peak =
        std::max(impl_->counters.prefetch_peak, impl_->counters.prefetch_inflight);
    impl_->changed.notify_all();
    return result;
}

void KVOffloadCache::Impl::run_prefetch() {
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return stopping || !prefetch_queue.empty(); });
        if (prefetch_queue.empty()) return;
        auto job = std::move(prefetch_queue.front());
        prefetch_queue.pop_front();
        prefetch_queue_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        OffloadClock::now() - job.submitted)
                                        .count(),
                                    std::memory_order_relaxed);
        lock.unlock();
        bool resident = false;
        std::exception_ptr error;
        try {
            OffloadTimer timer{prefetch_ns};
            resident = promote(job.index, job.generation, job.allow_eviction, true);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        auto& entry = entries[job.index];
        // 逻辑槽可能已被重写，只清除当前任务对应的版本。
        if (entry.generation == job.generation) entry.prefetch = {};
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
        if (!writer.try_lock()) return false;
    } else {
        OffloadTimer waiting{writer_wait_ns};
        writer.lock();
    }
    std::unique_lock<std::mutex> lock(mutex);
    auto& entry = entries[index];
    if (!entry.id() || entry.generation != generation) return false;
    if (is_cpu_resident(entry)) {
        touch_cpu_slot(index);
        return true;
    }
    // 需求读取优先；预取不等待正在读取的源，也不等待所有 victim 的读者退场。
    if (entry.busy || entry.readers) return false;
    const auto usage = storage.stats(StorageLocation::CPU);
    const auto capacity = storage.cpu_allocation_size(entry.size());
    const bool needs_room = capacity > usage.limit_bytes - usage.reserved_bytes;
    if (needs_room) {
        if (!allow_eviction && !hot) return false;
        if (victim(hot && !allow_eviction) == none) return false;
    }
    entry.busy = true;
    storage_detail::ScopeExit release([&] {
        if (!lock.owns_lock()) lock.lock();
        entry.busy = false;
        changed.notify_all();
    });
    for (;;) {
        const auto available = storage.stats(StorageLocation::CPU);
        if (capacity <= available.limit_bytes - available.reserved_bytes) break;
        make_room(lock, hot && !allow_eviction, capacity);
    }
    lock.unlock();
    const auto result = storage.migrate(entry.id(), StorageLocation::CPU);
    lock.lock();
    apply_migration(index, result, StorageLocation::CPU, prefetched, hot);
    require_complete(result);
    return true;
}
} // namespace infer
