// 实现 scratch 切片、capture 归属检查、结束/中止及 Graph 使用量记录。
#include "infer/cuda/cuda_graph_allocator.h"
#include "infer/cuda/cuda_stream.h"

#include <algorithm>
#include <stdexcept>

namespace infer::cuda_pool_detail {

GraphAllocator::GraphAllocator(const CachingCUDAAllocatorConfig& config, CUDAMemoryPoolStats& stats)
    : config_(config), stats_(stats) {}

std::size_t GraphAllocator::align_size(std::size_t bytes) {
    constexpr auto alignment = CachingCUDADeviceAllocator::allocation_alignment;
    if (!bytes) {
        throw std::invalid_argument("Cannot allocate zero bytes");
    }
    if (bytes > SIZE_MAX - (alignment - 1)) {
        throw std::overflow_error("CUDA alignment overflow");
    }
    return (bytes + alignment - 1) & ~(alignment - 1);
}

void GraphAllocator::not_pool_capturing() const {
    // 必须在任何 CUDA API 前检查：capture scope 内 flush/collect/trim 也不能查询 event。
    if (capture_) {
        throw std::runtime_error("Operation unavailable inside pool graph capture scope");
    }
}

GraphAllocator::Records GraphAllocator::prepare_records() const {
    Records records;
    records.reserve(config_.graph_max_allocations);
    return records;
}

void GraphAllocator::attach_workspace(void* ptr, Records& records) noexcept {
    scratch_ = ptr;
    graph_allocations_.swap(records);
}

void GraphAllocator::capture_owner(cudaStream_t stream) const {
    if (capture_thread_ != std::this_thread::get_id() || stream != capture_stream_) {
        throw std::runtime_error("Graph workspace belongs to capture thread and stream");
    }
}

void* GraphAllocator::allocate(std::size_t bytes, cudaStream_t stream) {
    // 1. 检查线程/流、对齐溢出、剩余空间和记录数量，失败时 offset 不变。
    capture_owner(stream);
    const auto capacity = align_size(bytes);
    if (capacity > stats_.graph_reserved_bytes - graph_offset_ ||
        graph_allocations_.size() == config_.graph_max_allocations) {
        throw std::bad_alloc();
    }
    void* ptr = static_cast<char*>(scratch_) + graph_offset_;

    // 2. vector 在 capture 前已 reserve；以下只更新主机记录，不申请内存、不调用 CUDA。
    graph_allocations_.push_back({ptr, true});
    graph_offset_ += capacity;
    ++graph_live_;
    return ptr;
}

void GraphAllocator::release(void* ptr, cudaStream_t stream) {
    capture_owner(stream);
    // 分配记录按地址升序，用 lower_bound 校验确切首地址和 live 状态。
    // 多个 Graph 可能共享同一地址，所以这些记录不能登记到 eager BlockMap。
    auto it = std::lower_bound(graph_allocations_.begin(), graph_allocations_.end(), ptr,
                               [](const GraphAllocation& allocation, void* address) {
                                   return std::less<void*>{}(allocation.ptr, address);
                               });
    if (it == graph_allocations_.end() || it->ptr != ptr || !it->live) {
        throw std::invalid_argument("Graph release requires a live allocation in this capture");
    }
    it->live = false;
    --graph_live_;
    // 只归还主机 lease，不回退 offset：录制中的 kernel 仍可能使用先前的地址。
}

void GraphAllocator::validate_begin(GraphId id, cudaStream_t stream) const {
    if (!scratch_) {
        throw std::runtime_error("Reserve graph workspace before capture");
    }
    if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        throw std::invalid_argument("Graph capture requires an explicit stream");
    }
    // 与原 contains 相同的语义，使用 C++17 支持的 find。
    if (graphs_.find(id) != graphs_.end()) {
        throw std::invalid_argument("Graph ID is already registered");
    }
}

void GraphAllocator::begin_capture(GraphId id, cudaStream_t stream) {
    // map 分配是这里唯一可能失败的步骤；成功前不进入 capture 状态。
    auto it = graphs_.try_emplace(id).first;
    capture_ = &it->second;
    capture_id_ = id;
    capture_stream_ = stream;
    capture_thread_ = std::this_thread::get_id();
    graph_offset_ = 0;
    graph_live_ = 0;
    graph_allocations_.clear();
}

void GraphAllocator::finish_capture(GraphId id, bool abort) {
    if (!capture_ || id != capture_id_) {
        throw std::invalid_argument("Unknown active graph capture");
    }
    capture_owner(capture_stream_);
    // 真正的 CUDA capture 必须先结束；钩子顺序错误时保留全部状态以供重试。
    not_cuda_capturing(capture_stream_);
    if (graph_live_ && !abort) {
        throw std::runtime_error("Release graph host leases before ending capture");
    }
    if (abort) {
        graphs_.erase(id);
    } else {
        capture_->used = graph_offset_;
        capture_->complete = true;
        ++stats_.graph_count;
        stats_.graph_peak_used_bytes = std::max(stats_.graph_peak_used_bytes, graph_offset_);
    }
    capture_ = nullptr;
    graph_live_ = 0;
    graph_offset_ = 0;
    graph_allocations_.clear();
}

std::size_t GraphAllocator::workspace_used(GraphId id) const {
    auto it = graphs_.find(id);
    if (it == graphs_.end() || !it->second.complete) {
        throw std::invalid_argument("Graph has not completed capture");
    }
    return it->second.used;
}

void GraphAllocator::forget_graph(GraphId id) {
    if (!graphs_.erase(id)) {
        throw std::invalid_argument("Graph is not registered");
    }
    --stats_.graph_count;
}

void GraphAllocator::check_no_graphs() const {
    if (!graphs_.empty()) {
        throw std::runtime_error("Destroy and forget graphs before releasing workspace");
    }
}

void GraphAllocator::clear_workspace() noexcept {
    scratch_ = nullptr;
    graph_allocations_.clear();
}

} // namespace infer::cuda_pool_detail
