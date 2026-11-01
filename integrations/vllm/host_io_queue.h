#pragma once

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Lazy read/write dispatchers feed the backend's existing native batch workers.
// Disjoint jobs can overlap; slot and buffer dependencies retain submission order.
// At most 16 batches retain caller buffers. No per-block promises or extra payload copies.
class HostIOQueue {
  public:
    using Work =
        std::function<void(bool, const std::vector<std::size_t>&, const std::vector<void*>&)>;
    explicit HostIOQueue(Work work, std::size_t bytes) : work_(std::move(work)), bytes_(bytes) {}
    ~HostIOQueue() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_)
            if (worker.joinable())
                worker.join();
    }
    std::uint64_t submit(bool write, std::vector<std::size_t> slots, std::vector<void*> buffers) {
        auto job = std::make_shared<Job>();
        job->write = write;
        job->slots = std::move(slots);
        job->buffers = std::move(buffers);
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || jobs_.size() >= 16 || !next_)
            throw std::runtime_error("Native I/O queue is full or stopping");
        if (!workers_[write].joinable())
            workers_[write] = std::thread([this, write] { run(write); });
        const auto token = next_++;
        jobs_.emplace(token, job);
        peak_ = std::max(peak_, jobs_.size());
        ready_.notify_all();
        return token;
    }
    void wait(std::uint64_t token) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = jobs_.find(token);
        if (it == jobs_.end() || it->second->waiting)
            throw std::invalid_argument("Unknown or already consumed native I/O token");
        auto job = it->second;
        job->waiting = true;
        ready_.wait(lock, [&] { return job->finished; });
        jobs_.erase(it);
        if (job->failure)
            std::rethrow_exception(job->failure);
    }
    std::size_t peak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

  private:
    struct Job {
        bool write = false, waiting = false, started = false, finished = false;
        std::vector<std::size_t> slots;
        std::vector<void*> buffers;
        std::exception_ptr failure;
    };
    bool conflicts(const Job& a, const Job& b) const {
        for (auto slot : a.slots)
            if (std::find(b.slots.begin(), b.slots.end(), slot) != b.slots.end())
                return true;
        if (a.write && b.write)
            return false; // Both jobs only read their input buffers.
        for (auto* first : a.buffers) {
            const auto begin = reinterpret_cast<std::uintptr_t>(first);
            for (auto* second : b.buffers) {
                const auto other = reinterpret_cast<std::uintptr_t>(second);
                if (begin < other + bytes_ && other < begin + bytes_)
                    return true;
            }
        }
        return false;
    }
    // Called under mutex_. Running jobs still block later dependent submissions.
    auto ready_job(bool write) {
        return std::find_if(jobs_.begin(), jobs_.end(), [&](const auto& item) {
            const auto& job = *item.second;
            if (job.write != write || job.started)
                return false;
            for (const auto& earlier : jobs_) {
                if (earlier.first >= item.first)
                    break;
                if (!earlier.second->finished && conflicts(*earlier.second, job))
                    return false;
            }
            return true;
        });
    }
    void run(bool write) noexcept {
        for (;;) {
            // Fixed batch handles: allocation failures are caught before any caller returns.
            std::array<std::shared_ptr<Job>, 16> group;
            std::size_t count = 0;
            std::exception_ptr failure;
            try {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock, [&] {
                        return (stopping_ && std::all_of(jobs_.begin(), jobs_.end(),
                                                        [](const auto& item) {
                                                            return item.second->started;
                                                        })) ||
                               ready_job(write) != jobs_.end();
                    });
                    const auto next = ready_job(write);
                    if (next == jobs_.end())
                        return;
                    next->second->started = true;
                    group[count++] = next->second;
                }
                // Take ownership before allocations: OOM must complete an accepted token.
                std::vector<std::size_t> slots;
                std::vector<void*> buffers;
                slots.reserve(64);
                buffers.reserve(64);
                slots.insert(slots.end(), group[0]->slots.begin(), group[0]->slots.end());
                buffers.insert(buffers.end(), group[0]->buffers.begin(), group[0]->buffers.end());
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    while (count < group.size()) {
                        const auto next = ready_job(write);
                        if (next == jobs_.end() || slots.size() + next->second->slots.size() > 64)
                            break;
                        const auto& job = next->second;
                        slots.insert(slots.end(), job->slots.begin(), job->slots.end());
                        buffers.insert(buffers.end(), job->buffers.begin(), job->buffers.end());
                        group[count++] = job;
                        job->started = true;
                    }
                }
                work_(write, slots, buffers);
            } catch (...) {
                failure = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (std::size_t i = 0; i < count; ++i) {
                    group[i]->finished = true;
                    group[i]->failure = failure;
                }
            }
            ready_.notify_all();
        }
    }
    Work work_;
    std::size_t bytes_, peak_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::map<std::uint64_t, std::shared_ptr<Job>> jobs_;
    std::array<std::thread, 2> workers_;
    std::uint64_t next_ = 1;
    bool stopping_ = false;
};
