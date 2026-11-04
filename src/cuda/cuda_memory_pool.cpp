// 保留 public API，统一加锁、调度模块并管理 cudaMalloc/cudaFree 生命周期。
#include "infer/cuda/cuda_memory_pool.h"
#include "infer/cuda/cuda_block.h"
#include "infer/cuda/cuda_graph_allocator.h"
#include "infer/cuda/cuda_stream.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace infer {
using namespace cuda_pool_detail;

class CachingCUDADeviceAllocator::Impl {
    CachingCUDAAllocatorConfig config_;
    CUDAMemoryPoolStats stats_;
    std::mutex mutex_;
    BlockMap blocks_;
    Bins ready_{};

    // 成员按声明顺序构造、逆序销毁；被引用的数据必须比内部模块更早构造。
    StreamManager streams_;
    GraphAllocator graphs_;

    // 以下内部调度函数不再加锁，调用者必须已经持有 mutex_。
    void* activate(Block& block, std::size_t requested, std::size_t capacity, Stream& owner) {
        if (can_split(block, capacity, config_.min_split_bytes)) {
            auto& bins = block.state == State::pending ? block.owner->bins : ready_;
            Block& tail = split_block(blocks_, bins, block, capacity);
            // split 元数据成功后，余下衔接不抛异常；字节总量不变，块数增加一。
            if (block.state == State::pending) {
                streams_.inherit_retired(block, tail);
            } else {
                ++stats_.ready_blocks;
            }
            ++stats_.splits;
        }
        if (block.state == State::pending) {
            streams_.detach_pending(block);
        } else {
            remove_bin(ready_, block);
            stats_.ready_bytes -= block.capacity;
            --stats_.ready_blocks;
        }
        block.state = State::active;
        block.requested = requested;
        block.owner = &owner;
        stats_.active_requested_bytes += requested;
        stats_.active_capacity_bytes += block.capacity;
        ++stats_.active_blocks;
        ++stats_.reuse_count;
        return block.ptr;
    }

    void trim_unlocked() {
        // 1. 封存并等待所有流的 pending 前缀；提升过程中由 Block 模块合并邻居。
        streams_.wait_all();
        // 2. 只释放完整 segment。还有 active 兄弟的 ready 子块必须留在池中。
        for (auto& head : ready_) {
            Block* block = head;
            while (block) {
                Block* next = block->next;
                if (whole_segment(*block)) {
                    // cudaFree 成功之后才提交账目和摘除元数据；失败时仍可重试 trim。
                    check_cuda(cudaFree(block->ptr), "pool trim free");
                    auto ptr = block->ptr;
                    stats_.reserved_bytes -= block->capacity;
                    stats_.ready_bytes -= block->capacity;
                    --stats_.ready_blocks;
                    --stats_.segments;
                    ++stats_.upstream_releases;
                    remove_bin(ready_, *block);
                    blocks_.erase(ptr);
                }
                block = next;
            }
        }
    }

    bool exceeds_budget(std::size_t capacity) const noexcept {
        // 用减法比较避免 reserved + capacity 溢出；0 表示不设置容量预算。
        return config_.max_reserved_bytes &&
               (capacity > config_.max_reserved_bytes ||
                stats_.reserved_bytes > config_.max_reserved_bytes - capacity);
    }

    void account_reserve(std::size_t capacity) noexcept {
        stats_.reserved_bytes += capacity;
        stats_.peak_reserved_bytes = std::max(stats_.peak_reserved_bytes, stats_.reserved_bytes);
        ++stats_.upstream_allocations;
    }

  public:
    explicit Impl(CachingCUDAAllocatorConfig config)
        : config_(config), streams_(config_, stats_, blocks_, ready_), graphs_(config_, stats_) {
        if (!config_.event_batch_size || !config_.graph_max_allocations ||
            config_.min_split_bytes % allocation_alignment ||
            (config_.large_allocation_granularity &&
             (config_.large_allocation_granularity < allocation_alignment ||
              (config_.large_allocation_granularity &
               (config_.large_allocation_granularity - 1))))) {
            throw std::invalid_argument(
                "Invalid CUDA pool batch, graph allocation limit, or allocation alignment");
        }
    }

    ~Impl() {
        const int device = streams_.device();
        if (device < 0) {
            return;
        }
        // 正常契约要求 Capture/GraphExec 先结束；保留原有违规生命周期的兜底行为。
        // 此时不能在录制内同步/free/event destroy，故只报告并保留设备资源。
        if (graphs_.capturing()) {
            std::fputs("CUDA pool destroyed during capture; device storage retained\n", stderr);
            return;
        }
        int previous = -1;
        cudaGetDevice(&previous);
        if (cudaSetDevice(device) != cudaSuccess) {
            std::fputs("CUDA pool cleanup: cannot select device\n", stderr);
            return;
        }
        cudaError_t err = cudaSuccess;
        if (!stats_.active_blocks && !graphs_.workspace()) {
            // Retired eager allocations are covered by the pool's own events.
            // Waiting for unrelated device work cannot strengthen this ownership.
            try {
                streams_.wait_all();
            } catch (...) {
                std::fputs("CUDA pool cleanup: pending completion failed; device storage retained\n",
                           stderr);
                if (previous >= 0 && previous != device)
                    cudaSetDevice(previous);
                return;
            }
        } else {
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "CUDA pool cleanup sync: %s\n", cudaGetErrorString(err));
            }
        }
        for (auto& item : blocks_) {
            if (!item.second.phys_prev) {
                // 只 free segment 根地址一次，即使调用者遗漏归还 active 子块。
                err = cudaFree(item.first);
                if (err != cudaSuccess) {
                    std::fprintf(stderr, "CUDA pool cleanup free: %s\n", cudaGetErrorString(err));
                }
            }
        }
        if (graphs_.workspace() && !stats_.graph_count) {
            cudaFree(graphs_.workspace());
        }
        if (stats_.graph_count) {
            std::fputs("CUDA pool destroyed with live graphs; private workspace retained\n", stderr);
        }
        streams_.destroy_events();
        if (previous >= 0 && previous != device) {
            cudaSetDevice(previous);
        }
    }

    void* allocate(std::size_t bytes, cudaStream_t handle) {
        if (!bytes) {
            throw std::invalid_argument("Cannot allocate zero bytes");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (graphs_.capturing()) {
            return graphs_.allocate(bytes, handle);
        }
        auto [bin, capacity] = allocation_bucket(bytes, config_);
        Stream& stream = streams_.stream(handle);
        // 1. 当前流 pending：依靠同流执行顺序直接复用，无需等 GPU 完成。
        if (auto* block = find(stream.bins, bin, capacity, config_)) {
            return activate(*block, bytes, capacity, stream);
        }
        // 2. 全局 ready：已确认完成，可交给任意登记流。
        if (auto* block = find(ready_, bin, capacity, config_)) {
            return activate(*block, bytes, capacity, stream);
        }
        // 3. 有界查询 event，把新完成的块提升到 ready，再查一次。
        streams_.collect(true);
        if (auto* block = find(ready_, bin, capacity, config_)) {
            return activate(*block, bytes, capacity, stream);
        }

        // 4. cudaMalloc fallback：预算不足先 trim，OOM 时保留原来的一次 trim/重试。
        // Amortize upstream CUDA calls with splittable segments. Explicit
        // budgets and cache limits still cap the physical reservation.
        const auto segment_target = capacity <= (1ULL << 20) ? (2ULL << 20) : (16ULL << 20);
        const auto segment_capacity = config_.min_split_bytes &&
                segment_target <= config_.max_cached_bytes &&
                (!config_.max_reserved_bytes || segment_target <= config_.max_reserved_bytes -
                    std::min(config_.max_reserved_bytes,stats_.reserved_bytes))
            ? std::max<std::size_t>(capacity,segment_target) : capacity;
        if (exceeds_budget(capacity)) {
            trim_unlocked();
            // 等待后可能只得到 ready 子块，active 邻居使 segment 无法整体释放。
            if (auto* block = find(ready_, bin, capacity, config_)) {
                return activate(*block, bytes, capacity, stream);
            }
        }
        if (exceeds_budget(capacity)) {
            throw std::bad_alloc();
        }
        if (stats_.reserved_bytes > SIZE_MAX - segment_capacity) {
            throw std::overflow_error("CUDA pool total capacity overflow");
        }
        // 先预留主机节点和哈希桶，避免显存申请成功后才发生主机登记失败。
        blocks_.try_emplace(nullptr);
        void* ptr = nullptr;
        auto allocated_capacity = segment_capacity;
        try {
            auto err = cudaMalloc(&ptr, segment_capacity);
            if (err == cudaErrorMemoryAllocation) {
                ++stats_.oom_retries;
                trim_unlocked();
                if (auto* block = find(ready_, bin, capacity, config_)) {
                    // 不再申请新 segment，先撤销其占位节点；activate 失败仍可重试。
                    blocks_.erase(nullptr);
                    return activate(*block, bytes, capacity, stream);
                }
                // Slabs are a reuse preference, never a minimum free-VRAM
                // requirement for an otherwise satisfiable allocation.
                allocated_capacity = capacity;
                err = cudaMalloc(&ptr, allocated_capacity);
            }
            if (err == cudaErrorMemoryAllocation) {
                throw std::bad_alloc();
            }
            check_cuda(err, "pool cudaMalloc");
        } catch (...) {
            blocks_.erase(nullptr);
            throw;
        }
        auto node = blocks_.extract(nullptr);
        node.key() = ptr;
        auto& block = node.mapped();
        block.ptr = ptr;
        block.capacity = allocated_capacity;
        block.requested = bytes;
        block.bin = free_bin(allocated_capacity);
        block.owner = &stream;
        // C++17 node handle：复用预留节点；桶容量已足够，指针 hash/equality 不抛异常。
        blocks_.insert(std::move(node));
        auto& reserved = blocks_.at(ptr);
        reserved.state = State::ready;
        add_bin(ready_,reserved);
        stats_.ready_bytes += allocated_capacity;
        ++stats_.ready_blocks;
        account_reserve(allocated_capacity);
        ++stats_.segments;
        auto* result = activate(reserved,bytes,capacity,stream);
        --stats_.reuse_count; // An upstream segment is not a reuse hit.
        return result;
    }

    void release(void* ptr, cudaStream_t handle) {
        if (!ptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (graphs_.capturing()) {
            graphs_.release(ptr, handle);
            return;
        }
        auto it = blocks_.find(ptr);
        if (it == blocks_.end() || it->second.state != State::active) {
            throw std::invalid_argument("CUDA pool release requires a live block start");
        }
        Stream& stream = streams_.stream(handle);
        Block& block = it->second;
        const auto cached = stats_.pending_bytes + stats_.ready_bytes;
        // A dedicated pool whose entire reservation fits in the cache can
        // retire every allocation through events, including its last segment.
        if (stats_.reserved_bytes > config_.max_cached_bytes && whole_segment(block) &&
            (cached > config_.max_cached_bytes || block.capacity > config_.max_cached_bytes - cached)) {
            // 缓存上限分支沿用原同步释放；失败前不撤销调用者的 active 所有权。
            check_cuda(cudaStreamSynchronize(handle), "pool cache limit wait");
            ++stats_.completion_waits;
            check_cuda(cudaFree(ptr), "pool cache limit free");
            stats_.active_requested_bytes -= block.requested;
            stats_.active_capacity_bytes -= block.capacity;
            --stats_.active_blocks;
            stats_.reserved_bytes -= block.capacity;
            ++stats_.upstream_releases;
            --stats_.segments;
            blocks_.erase(it);
            return;
        }
        // 正常归还交给 Stream Manager：检查 epoch/记录 event -> active 转 pending
        // -> 加入该流桶和退休 FIFO -> 后续 collect/wait 确认完成，再转 ready。
        // event 必须在提交状态之前记录，才能保留失败可重试的异常安全保证。
        streams_.retire(stream, block);
    }

    StreamId register_stream(cudaStream_t handle, StreamId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        return streams_.stream(handle, id).id;
    }

    void unregister_stream(cudaStream_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        streams_.unregister_stream(handle);
    }

    void reserve_graph_workspace(std::size_t bytes) {
        const auto capacity = GraphAllocator::align_size(bytes);
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        if (graphs_.workspace()) {
            if (capacity <= stats_.graph_reserved_bytes) {
                return;
            }
            throw std::runtime_error("Release graph workspace before growing it");
        }
        not_cuda_capturing(nullptr);
        streams_.check_device();
        if (exceeds_budget(capacity)) {
            throw std::bad_alloc();
        }
        if (stats_.reserved_bytes > SIZE_MAX - capacity) {
            throw std::overflow_error("CUDA pool total capacity overflow");
        }
        // 先准备主机表，再申请显存，最后提交；任一步失败都不登记半成品 workspace。
        auto records = graphs_.prepare_records();
        void* ptr{};
        auto err = cudaMalloc(&ptr, capacity);
        if (err == cudaErrorMemoryAllocation) {
            throw std::bad_alloc();
        }
        check_cuda(err, "pool allocate graph workspace");
        graphs_.attach_workspace(ptr, records);
        stats_.graph_reserved_bytes = capacity;
        account_reserve(capacity);
    }

    void begin_graph_capture(GraphId id, cudaStream_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        graphs_.validate_begin(id, handle);
        streams_.stream(handle); // CUDA 检查/event 创建必须先于真正的 CUDA Capture。
        graphs_.begin_capture(id, handle);
    }

    void end_graph_capture(GraphId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.finish_capture(id, false);
    }

    void abort_graph_capture(GraphId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.finish_capture(id, true);
    }

    std::size_t graph_workspace_used(GraphId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return graphs_.workspace_used(id);
    }

    void forget_graph(GraphId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        graphs_.forget_graph(id);
    }

    void release_graph_workspace() {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        graphs_.check_no_graphs();
        if (!graphs_.workspace()) {
            return;
        }
        not_cuda_capturing(nullptr);
        streams_.check_device();
        check_cuda(cudaDeviceSynchronize(), "pool wait graph workspace");
        // free 失败时保留 scratch 和统计，让 release_graph_workspace 可以重试。
        check_cuda(cudaFree(graphs_.workspace()), "pool free graph workspace");
        stats_.reserved_bytes -= stats_.graph_reserved_bytes;
        stats_.graph_reserved_bytes = 0;
        ++stats_.upstream_releases;
        graphs_.clear_workspace();
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        if (streams_.device() < 0) {
            return;
        }
        streams_.check_device();
        streams_.flush(); // 只封存，不等待、不提升 pending。
    }

    void collect() {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        if (streams_.device() < 0) {
            return;
        }
        streams_.check_device();
        streams_.collect(false); // 完整轮询，仍不主动等待 GPU。
    }

    void trim() {
        std::lock_guard<std::mutex> lock(mutex_);
        graphs_.not_pool_capturing();
        if (streams_.device() < 0) {
            return;
        }
        streams_.check_device();
        trim_unlocked();
    }

    CUDAMemoryPoolStats stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_; // 按值返回一致快照，调用者不会持有内部可写引用。
    }
};

// 对外仍是原来的 PImpl + unique_ptr；以下仅转发，不改变签名或默认参数。
CachingCUDADeviceAllocator::CachingCUDADeviceAllocator(CachingCUDAAllocatorConfig config)
    : DeviceAllocator(DeviceType::CUDA), impl_(std::make_unique<Impl>(config)) {}
CachingCUDADeviceAllocator::~CachingCUDADeviceAllocator() = default;

void* CachingCUDADeviceAllocator::allocate(std::size_t bytes, cudaStream_t stream) const {
    return impl_->allocate(bytes, stream);
}
void CachingCUDADeviceAllocator::release(void* ptr, cudaStream_t stream) const {
    impl_->release(ptr, stream);
}
CachingCUDADeviceAllocator::StreamId
CachingCUDADeviceAllocator::register_stream(cudaStream_t stream, StreamId id) const {
    return impl_->register_stream(stream, id);
}
void CachingCUDADeviceAllocator::unregister_stream(cudaStream_t stream) const {
    impl_->unregister_stream(stream);
}
void CachingCUDADeviceAllocator::reserve_graph_workspace(std::size_t bytes) const {
    impl_->reserve_graph_workspace(bytes);
}
void CachingCUDADeviceAllocator::begin_graph_capture(GraphId id, cudaStream_t stream) const {
    impl_->begin_graph_capture(id, stream);
}
void CachingCUDADeviceAllocator::end_graph_capture(GraphId id) const {
    impl_->end_graph_capture(id);
}
void CachingCUDADeviceAllocator::abort_graph_capture(GraphId id) const {
    impl_->abort_graph_capture(id);
}
std::size_t CachingCUDADeviceAllocator::graph_workspace_used(GraphId id) const {
    return impl_->graph_workspace_used(id);
}
void CachingCUDADeviceAllocator::forget_graph(GraphId id) const {
    impl_->forget_graph(id);
}
void CachingCUDADeviceAllocator::release_graph_workspace() const {
    impl_->release_graph_workspace();
}
void CachingCUDADeviceAllocator::flush() const {
    impl_->flush();
}
void CachingCUDADeviceAllocator::collect() const {
    impl_->collect();
}
void CachingCUDADeviceAllocator::trim() const {
    impl_->trim();
}
CUDAMemoryPoolStats CachingCUDADeviceAllocator::stats() const {
    return impl_->stats();
}

GraphCaptureScope::GraphCaptureScope(const CachingCUDADeviceAllocator& allocator,
                                   CachingCUDADeviceAllocator::GraphId id, cudaStream_t stream)
    : allocator_(allocator), id_(id), stream_(stream) {
    allocator_.begin_graph_capture(id_, stream_);
    const auto error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (error != cudaSuccess) {
        allocator_.abort_graph_capture(id_);
        check_cuda(error, "begin graph capture");
    }
    capturing_ = true;
}

GraphCaptureScope::~GraphCaptureScope() {
    if (capturing_) cudaStreamEndCapture(stream_, &graph_);
    if (graph_) cudaGraphDestroy(graph_);
    if (!active_) return;
    try {
        allocator_.abort_graph_capture(id_);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Abort graph capture: %s\n", error.what());
    }
}

cudaGraph_t GraphCaptureScope::finish() {
    if (!active_ || !capturing_) throw std::logic_error("Graph capture is already finished");
    const auto error = cudaStreamEndCapture(stream_, &graph_);
    capturing_ = false;
    check_cuda(error, "end graph capture");
    allocator_.end_graph_capture(id_);
    active_ = false;
    const auto graph = graph_;
    graph_ = nullptr;
    return graph;
}

} // namespace infer
