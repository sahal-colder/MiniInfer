// 实现 stream-aware 异步回收及 pending -> ready 生命周期转换。
#include "infer/cuda/cuda_stream.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace infer::cuda_pool_detail {

StreamManager::StreamManager(const CachingCUDAAllocatorConfig& config, CUDAMemoryPoolStats& stats,
                             BlockMap& blocks, Bins& ready)
    : config_(config), stats_(stats), blocks_(blocks), ready_(ready) {}

void StreamManager::check_device() {
    int device;
    check_cuda(cudaGetDevice(&device), "pool current device");
    if (device_ < 0) {
        device_ = device;
    }
    if (device != device_) {
        throw std::invalid_argument("CUDA pool belongs to another device");
    }
}

void not_cuda_capturing(cudaStream_t handle) {
    cudaStreamCaptureStatus status;
    check_cuda(cudaStreamIsCapturing(handle, &status), "pool capture status");
    if (status != cudaStreamCaptureStatusNone) {
        throw std::runtime_error("Use pool graph capture hooks before CUDA capture");
    }
}

Stream& StreamManager::stream(cudaStream_t handle, StreamId requested_id) {
    if (handle == cudaStreamPerThread) {
        throw std::invalid_argument("Use an explicit stream instead of per-thread default stream");
    }
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (!handle) {
        throw std::invalid_argument("Use an explicit stream with PTDS builds");
    }
#endif
    // 1. 先验证 capture 和设备；capture 内 allocate/release 根本不会进入本函数。
    not_cuda_capturing(handle);
    check_device();
    auto found = streams_.find(handle);
    if (found != streams_.end()) {
        if (requested_id && found->second.id != requested_id) {
            throw std::invalid_argument("Stream already has another logical ID");
        }
        return found->second;
    }
#if CUDART_VERSION >= 12080
    // 首次登记时的额外诊断；旧 Toolkit 仍沿用 flags 验证路径。
    int device;
    check_cuda(cudaStreamGetDevice(handle, &device), "pool stream device");
    if (device != device_) {
        throw std::invalid_argument("CUDA pool stream belongs to another device");
    }
#else
    unsigned flags;
    check_cuda(cudaStreamGetFlags(handle, &flags), "pool validate stream");
#endif

    // 2. 保留显式 ID 去重和自动 ID 溢出检查。
    auto id_in_use = [&](StreamId id) {
        return std::any_of(streams_.begin(), streams_.end(),
                           [&](const auto& item) { return item.second.id == id; });
    };
    StreamId id = requested_id;
    if (id && id_in_use(id)) {
        throw std::invalid_argument("Duplicate logical stream ID");
    }
    if (!id) {
        while (next_stream_id_ && id_in_use(next_stream_id_)) {
            ++next_stream_id_;
        }
        if (!next_stream_id_) {
            throw std::overflow_error("CUDA pool stream ID overflow");
        }
        id = next_stream_id_++;
    }

    // 3. 主机节点先登记，event 创建失败则移除节点，不遗留半初始化的 Stream。
    auto it = streams_.try_emplace(handle).first;
    auto err = cudaEventCreateWithFlags(&it->second.event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        streams_.erase(it);
        check_cuda(err, "pool create event");
    }
    it->second.handle = handle;
    it->second.id = id;
    return it->second;
}

void StreamManager::add_retired(Stream& stream, Block& block) noexcept {
    block.retire_prev = stream.tail;
    block.retire_next = nullptr;
    if (stream.tail) {
        stream.tail->retire_next = &block;
    } else {
        stream.head = &block;
    }
    stream.tail = &block;
}

void StreamManager::remove_retired(Stream& stream, Block& block) noexcept {
    if (block.retire_prev) {
        block.retire_prev->retire_next = block.retire_next;
    } else {
        stream.head = block.retire_next;
    }
    if (block.retire_next) {
        block.retire_next->retire_prev = block.retire_prev;
    } else {
        stream.tail = block.retire_prev;
    }
    block.retire_prev = nullptr;
    block.retire_next = nullptr;
}

void StreamManager::inherit_retired(Block& head, Block& tail) noexcept {
    // split 尾块仍被旧 kernel 使用，必须留在头块之后，不能追加到 FIFO 末尾。
    // Block 模块已经复制 state/owner/epoch；这里仅衔接退休链表并增加块数。
    tail.retire_prev = &head;
    tail.retire_next = head.retire_next;
    if (tail.retire_next) {
        tail.retire_next->retire_prev = &tail;
    } else {
        head.owner->tail = &tail;
    }
    head.retire_next = &tail;
    ++stats_.pending_blocks;
}

void StreamManager::detach_pending(Block& block) noexcept {
    // 同流复用：GPU 顺序保证安全，但必须同时从容量桶和退休 FIFO 摘除。
    remove_bin(block.owner->bins, block);
    remove_retired(*block.owner, block);
    stats_.pending_bytes -= block.capacity;
    --stats_.pending_blocks;
}

void StreamManager::retire(Stream& stream, Block& block) {
    // 1. 先检查 epoch，并在达到批量阈值时记录 event。
    // event 记录失败时尚未改 Block/FIFO/active 账目，调用者可以重试 release。
    if (stream.epoch == ULLONG_MAX) {
        throw std::overflow_error("CUDA pool stream epoch overflow");
    }
    const auto epoch = stream.epoch + 1;
    if (stream.unrecorded + 1 >= config_.event_batch_size) {
        check_cuda(cudaEventRecord(stream.event, stream.handle), "pool release event");
        stream.recorded = epoch;
        stream.unrecorded = 0;
        ++stats_.event_records;
    } else {
        ++stream.unrecorded;
    }

    // 2. 不再有可失败操作，提交 active -> pending，再加入桶和退休 FIFO。
    stream.epoch = epoch;
    block.epoch = epoch;
    block.owner = &stream;
    block.state = State::pending;
    add_bin(stream.bins, block);
    add_retired(stream, block);
    stats_.active_requested_bytes -= block.requested;
    stats_.active_capacity_bytes -= block.capacity;
    --stats_.active_blocks;
    stats_.pending_bytes += block.capacity;
    ++stats_.pending_blocks;
    // 3. 此处不阻塞等待 GPU。collect 查询完成；trim/unregister 才会主动等待。
}

void StreamManager::promote(Stream& stream, std::size_t limit) {
    // FIFO 按 epoch 排序，只取已被 completed 覆盖的前缀；未封存的块继续 pending。
    while (stream.head && stream.head->epoch <= stream.completed && limit--) {
        Block& block = *stream.head;
        remove_retired(stream, block);
        remove_bin(stream.bins, block);
        block.state = State::ready;
        block.owner = nullptr;
        add_bin(ready_, block);
        stats_.pending_bytes -= block.capacity;
        --stats_.pending_blocks;
        stats_.ready_bytes += block.capacity;
        ++stats_.ready_blocks;
        coalesce(blocks_, ready_, block, stats_);
        // coalesce 可能销毁 block 节点，下一轮重新读取 stream.head。
    }
}

void StreamManager::query(Stream& stream, std::size_t limit) {
    if (!stream.head) {
        return;
    }
    if (stream.recorded > stream.completed) {
        ++stats_.event_queries;
        auto err = cudaEventQuery(stream.event);
        if (err == cudaSuccess) {
            stream.completed = stream.recorded;
        } else if (err != cudaErrorNotReady) {
            check_cuda(err, "pool query event");
        }
    }
    promote(stream, limit);
}

void StreamManager::collect(bool bounded) {
    if (!bounded) {
        for (auto& item : streams_) {
            query(item.second, SIZE_MAX);
        }
        return;
    }
    // allocate 的慢路径沿用原预算：每次最多轮询 4 条流、每流提升 64 块。
    auto count = std::min<std::size_t>(streams_.size(), 4);
    while (count--) {
        if (poll_ == streams_.end()) {
            poll_ = streams_.begin();
        }
        query((poll_++)->second, 64);
    }
}

void StreamManager::seal(Stream& stream) {
    if (!stream.tail || stream.tail->epoch <= stream.recorded) {
        return;
    }
    not_cuda_capturing(stream.handle);
    check_cuda(cudaEventRecord(stream.event, stream.handle), "pool flush event");
    stream.recorded = stream.epoch;
    stream.unrecorded = 0;
    ++stats_.event_records;
}

void StreamManager::wait(Stream& stream) {
    // 未达到 batch 阈值的最后一批也要先 seal，才能等待并迁移到 ready。
    seal(stream);
    if (stream.head && stream.recorded > stream.completed) {
        check_cuda(cudaEventSynchronize(stream.event), "pool wait");
        ++stats_.completion_waits;
        stream.completed = stream.recorded;
    }
    promote(stream, SIZE_MAX);
}

void StreamManager::flush() {
    for (auto& item : streams_) {
        seal(item.second);
    }
}

void StreamManager::wait_all() {
    for (auto& item : streams_) {
        wait(item.second);
    }
}

void StreamManager::unregister_stream(cudaStream_t handle) {
    auto it = streams_.find(handle);
    if (it == streams_.end()) {
        throw std::invalid_argument("Stream is not registered");
    }
    not_cuda_capturing(handle);
    check_device();
    Stream& stream = it->second;
    for (const auto& item : blocks_) {
        if (item.second.state == State::active && item.second.owner == &stream) {
            throw std::runtime_error("Cannot unregister stream with active blocks");
        }
    }
    // 清空 pending 后不再有 Block 引用此 Stream；event 销毁成功后才删除登记。
    wait(stream);
    check_cuda(cudaEventDestroy(stream.event), "pool destroy stream event");
    if (poll_ == it) {
        ++poll_;
    }
    streams_.erase(it);
}

void StreamManager::destroy_events() noexcept {
    for (auto& item : streams_) {
        cudaEventDestroy(item.second.event);
    }
}

} // namespace infer::cuda_pool_detail
