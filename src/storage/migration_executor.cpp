// 有界异步调度。任务接受前准备资源，完成后归还额度，析构排空已接受任务。
// 迁移状态/预算/失败恢复由 transfer.cpp 唯一实现，本文件不拥有物理分配。
#include "storage_state.h"

#include <condition_variable>
#include <deque>
#include <optional>
#include <thread>
#include <utility>

namespace infer::storage_detail {

// 每个任务持有已经预留的记录；不创建无限数量的 std::async 线程。
// 队列只负责调度，事务语义仍由 prepare/execute_transfer 唯一实现。
struct MigrationExecutor::Impl {
    struct Batch {
        std::vector<LogicalBlockRecord*> records;
        std::vector<MigrationResult> results;
        std::promise<std::vector<MigrationResult>> result;
        std::size_t staging = 2;
        std::exception_ptr stopped;
    };
    struct Job {
        LogicalBlockRecord* record = nullptr;
        std::promise<MigrationResult> result;
        std::unique_ptr<Batch> batch;
    };
    std::shared_ptr<StorageState> state;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Job> queue;
    std::vector<std::thread> workers;
    std::size_t capacity, in_flight = 0;
    bool stopping = false;
    Impl(std::shared_ptr<StorageState> value, std::size_t limit)
        : state(std::move(value)), capacity(limit) {
        try {
            workers.reserve(limit);
            for (std::size_t i = 0; i < limit; ++i)
                workers.emplace_back([this] { run(); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~Impl() { stop(); }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        for (auto& worker : workers)
            if (worker.joinable())
                worker.join();
    }
    void run_batch(Batch& batch) {
        bool quarantined = false;
        for (std::size_t i = 0; i < batch.records.size(); ++i) {
            auto* record = batch.records[i];
            if (!record) {
                batch.results[i].committed = true;
                continue;
            }
            if (quarantined) {
                std::lock_guard<std::mutex> lock(state->mutex);
                cancel_prepared(*state, *record);
                batch.results[i].error = batch.stopped;
            } else {
                batch.results[i] = execute_transfer(*state, *record, &quarantined);
            }
            // execute_transfer 已交回该块的使用权，之后不能再解引用这个指针。
        }
        if (batch.staging != 2 && !quarantined) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->staging->busy[batch.staging] = false;
        }
    }
    void run() {
        for (;;) {
            std::optional<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty())
                    return;
                job.emplace(std::move(queue.front()));
                queue.pop_front();
            }
            MigrationResult result;
            std::exception_ptr unexpected;
            try {
                if (job->batch)
                    run_batch(*job->batch);
                else
                    result = execute_transfer(*state, *job->record);
            } catch (...) {
                unexpected = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                --in_flight;
            }
            if (job->batch) {
                if (unexpected)
                    job->batch->result.set_exception(unexpected);
                else
                    job->batch->result.set_value(std::move(job->batch->results));
            } else if (unexpected)
                job->result.set_exception(unexpected);
            else
                job->result.set_value(std::move(result));
        }
    }
};
MigrationExecutor::MigrationExecutor(std::shared_ptr<StorageState> state, std::size_t capacity)
    : impl_(std::make_unique<Impl>(std::move(state), capacity)) {}
MigrationExecutor::~MigrationExecutor() = default;
std::future<MigrationResult> MigrationExecutor::submit(BlockId id, StorageLocation target) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping || impl_->in_flight == impl_->capacity)
        throw StorageBusy("Async migration queue is full");
    // 先完成可能抛异常的 promise/队列分配，再保护块和计费；失败不会遗留 MOVING。
    impl_->queue.emplace_back();
    auto& job = impl_->queue.back();
    std::future<MigrationResult> future;
    try {
        future = job.result.get_future();
        job.record = prepare_transfer(*impl_->state, id, target);
    } catch (...) {
        impl_->queue.pop_back();
        throw;
    }
    if (!job.record) {
        job.result.set_value({true, false, {}});
        impl_->queue.pop_back();
    } else {
        ++impl_->in_flight;
        impl_->wake.notify_one();
    }
    return future;
}

std::future<std::vector<MigrationResult>>
MigrationExecutor::submit_batch(const std::vector<BlockId>& ids, StorageLocation target) {
    if (ids.empty() || ids.size() > 64)
        throw std::invalid_argument("Migration batch size must be 1..64");
    auto batch = std::make_unique<Impl::Batch>();
    batch->results.resize(ids.size());
    // 提前建立停止原因，避免 GPU 故障清理路径还需分配异常对象。
    batch->stopped =
        std::make_exception_ptr(StorageBusy("Batch stopped after unknown DMA completion"));
    auto future = batch->result.get_future();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping || impl_->in_flight == impl_->capacity)
        throw StorageBusy("Async migration queue is full");
    impl_->queue.emplace_back();
    auto& job = impl_->queue.back();
    job.batch = std::move(batch);
    try {
        job.batch->records = prepare_batch(*impl_->state, ids, target, job.batch->staging);
    } catch (...) {
        impl_->queue.pop_back();
        throw;
    }
    ++impl_->in_flight;
    impl_->wake.notify_one();
    return future;
}

} // namespace infer::storage_detail
