// 管理 CUDA Graph 的共享 scratch workspace 和 capture 期间的 bump 分配。
#pragma once

#include "infer/cuda/cuda_memory_pool.h"

#include <map>
#include <thread>
#include <vector>

namespace infer::cuda_pool_detail {

class GraphAllocator {
  public:
    using GraphId = CachingCUDADeviceAllocator::GraphId;
    struct GraphAllocation {
        void* ptr{};
        bool live{}; // 主机侧 lease，不代表 GraphExec 已结束使用这个地址。
    };
    using Records = std::vector<GraphAllocation>;

    GraphAllocator(const CachingCUDAAllocatorConfig& config, CUDAMemoryPoolStats& stats);
    GraphAllocator(const GraphAllocator&) = delete;
    GraphAllocator& operator=(const GraphAllocator&) = delete;

    static std::size_t align_size(std::size_t bytes);
    bool capturing() const noexcept { return capture_ != nullptr; }
    void not_pool_capturing() const;
    void* workspace() const noexcept { return scratch_; }

    // Pool 先准备主机记录，再 cudaMalloc；只有两步成功后才 attach 提交状态。
    Records prepare_records() const;
    void attach_workspace(void* ptr, Records& records) noexcept;
    void check_no_graphs() const;
    void clear_workspace() noexcept; // 仅在 Pool 成功 cudaFree 后调用。

    // validate -> Stream Manager 登记/检查流 -> begin；所有 CUDA 检查都发生在录制前。
    void validate_begin(GraphId id, cudaStream_t stream) const;
    void begin_capture(GraphId id, cudaStream_t stream);
    void finish_capture(GraphId id, bool abort);
    std::size_t workspace_used(GraphId id) const;
    void forget_graph(GraphId id);

    void* allocate(std::size_t bytes, cudaStream_t stream);
    void release(void* ptr, cudaStream_t stream);

  private:
    struct Graph {
        std::size_t used{};
        bool complete{};
    };
    void capture_owner(cudaStream_t stream) const;

    const CachingCUDAAllocatorConfig& config_;
    CUDAMemoryPoolStats& stats_;
    void* scratch_{}; // 整段物理显存由 Pool 的 RAII 生命周期负责释放。
    std::map<GraphId, Graph> graphs_;
    Records graph_allocations_;
    Graph* capture_{}; // 借用 map 节点，map 插入不会使地址失效。
    GraphId capture_id_{};
    cudaStream_t capture_stream_{};
    std::thread::id capture_thread_{};
    std::size_t graph_offset_{};
    std::size_t graph_live_{};
};

} // namespace infer::cuda_pool_detail
