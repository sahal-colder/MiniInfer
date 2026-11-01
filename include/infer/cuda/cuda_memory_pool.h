// 声明 CUDA 显存池的原有配置和 public API。
#pragma once

#include "infer/core/memory.h"
#include "infer/cuda/cuda_memory_stats.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace infer {

struct CachingCUDAAllocatorConfig {
    std::size_t small_block_threshold = 1ULL * 1024 * 1024;
    std::size_t large_block_max_waste = 1ULL * 1024 * 1024;
    std::size_t max_cached_bytes = 64ULL * 1024 * 1024;
    std::size_t max_reserved_bytes = 0;
    unsigned event_batch_size = 1;
    std::size_t min_split_bytes = 256;
    std::size_t graph_max_allocations = 4096;
    // Requests >= 16 MiB use this granularity; 0 retains geometric rounding.
    std::size_t large_allocation_granularity = 2ULL * 1024 * 1024;
};

class CachingCUDADeviceAllocator final : public DeviceAllocator {
  public:
    using GraphId = std::uint64_t;
    using StreamId = std::uint64_t;

    static constexpr std::size_t allocation_alignment = 256;

    explicit CachingCUDADeviceAllocator(CachingCUDAAllocatorConfig config = {});
    ~CachingCUDADeviceAllocator() override;

    CachingCUDADeviceAllocator(const CachingCUDADeviceAllocator&) = delete;
    CachingCUDADeviceAllocator& operator=(const CachingCUDADeviceAllocator&) = delete;

    void* allocate(std::size_t size, cudaStream_t stream = nullptr) const override;
    void release(void* ptr, cudaStream_t stream = nullptr) const override;

    StreamId register_stream(cudaStream_t stream, StreamId id = 0) const;
    void unregister_stream(cudaStream_t stream) const;

    void reserve_graph_workspace(std::size_t bytes) const;
    void begin_graph_capture(GraphId graph_id, cudaStream_t stream) const;
    void end_graph_capture(GraphId graph_id) const;
    void abort_graph_capture(GraphId graph_id) const;
    std::size_t graph_workspace_used(GraphId graph_id) const;
    void forget_graph(GraphId graph_id) const;
    void release_graph_workspace() const;

    void flush() const;
    void collect() const;
    void trim() const;
    CUDAMemoryPoolStats stats() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infer
