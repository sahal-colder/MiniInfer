#include "infer/storage/storage_block.h"
#include "storage_state.h"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace infer {
BlockLease::BlockLease(std::shared_ptr<storage_detail::StorageState> state,
                       storage_detail::LogicalBlockRecord* record, AccessMode mode, void* data,
                       int device, cudaStream_t stream, std::size_t use_index) noexcept
    : state_(std::move(state)), record_(record), mode_(mode), data_(data), device_id_(device),
      stream_(stream), use_index_(use_index) {}
BlockLease::~BlockLease() {
    reset();
}
BlockLease::BlockLease(BlockLease&& other) noexcept
    : state_(std::move(other.state_)), record_(std::exchange(other.record_, nullptr)),
      mode_(other.mode_), data_(other.data_), device_id_(other.device_id_), stream_(other.stream_),
      use_index_(other.use_index_) {}
BlockLease& BlockLease::operator=(BlockLease&& other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        record_ = std::exchange(other.record_, nullptr);
        mode_ = other.mode_;
        data_ = other.data_;
        device_id_ = other.device_id_;
        stream_ = other.stream_;
        use_index_ = other.use_index_;
    }
    return *this;
}
BlockId BlockLease::id() const noexcept {
    return record_ ? record_->meta.id : invalid_block_id;
}
const void* BlockLease::data() const noexcept {
    return record_ ? data_ : nullptr;
}
void* BlockLease::writable_data() const {
    if (!record_ || mode_ != AccessMode::WRITE)
        throw std::logic_error("A write lease is required");
    return data_;
}
std::size_t BlockLease::size() const noexcept {
    return record_ ? record_->meta.size : 0;
}
DeviceType BlockLease::device() const noexcept {
    if (!record_)
        return DeviceType::UNKNOWN;
    return record_->meta.location == StorageLocation::GPU ? DeviceType::CUDA : DeviceType::CPU;
}
int BlockLease::device_id() const noexcept {
    return record_ ? device_id_ : -1;
}
void BlockLease::detach(bool commit, bool completed) noexcept {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!completed) {
            record_->meta.state = BlockState::ERROR;
            state_->charge_cleanup(*record_);
        } else if (record_->meta.location == StorageLocation::GPU) {
            record_->consumer_uses[use_index_].pending = false;
        }
        if (mode_ == AccessMode::WRITE) {
            if (commit) {
                ++record_->meta.version;
                record_->initialized = true;
            }
            record_->writing = false;
        }
        --record_->meta.lease_count;
        record_ = nullptr;
    }
    // shared_ptr 的最后一次释放可能销毁 mutex，因此必须在锁作用域之外。
    state_.reset();
}
void BlockLease::finish() {
    if (!record_)
        return;
    record_->backend->wait(stream_); // 失败仍保留 Lease，调用者可以重试。
    detach(true, true);
}
void BlockLease::reset() noexcept {
    if (!record_)
        return;
    bool completed = false;
    try {
        record_->backend->wait(stream_);
        completed = true;
    } catch (const std::exception& error) {
        // 析构不能抛出；报告原因后由 detach 将资源列入 ERROR/待清理，禁止复用。
        std::fprintf(stderr, "Storage lease completion failed; resource quarantined: %s\n",
                     error.what());
    } catch (...) {
        std::fputs("Storage lease completion failed; resource quarantined\n", stderr);
    }
    detach(false, completed);
}
} // namespace infer
