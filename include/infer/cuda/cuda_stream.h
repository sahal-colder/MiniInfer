// 管理流身份、event、epoch、每流 pending 桶和退休 FIFO。
#pragma once

#include "infer/cuda/cuda_block.h"
#include "infer/cuda/cuda_memory_pool.h"

#include <map>

namespace infer::cuda_pool_detail {

struct Stream {
    cudaStream_t handle{}; // 借用调用者的流；池不会创建或销毁 CUDA stream。
    CachingCUDADeviceAllocator::StreamId id{};
    cudaEvent_t event{}; // 池创建的 event，由 Pool 析构流程或注销流程销毁。

    unsigned long long epoch{};     // 已提交的最后一次 release 序号。
    unsigned long long recorded{};  // 最近一次成功记录的 event 覆盖到哪里。
    unsigned long long completed{}; // 已确认 GPU 完成到哪里。
    unsigned unrecorded{};          // 尚未封入 event 的 release 数量。

    Bins bins{}; // 只有当前 stream 可以不等待 event 而复用这些 pending 块。
    Block* head{};
    Block* tail{}; // FIFO 中 epoch 单调不减，split 尾块继承旧位置和旧 epoch。
};

// 在 eager 路径或 capture 钩子边界调用；capture 内的分配/释放不能调用它。
void not_cuda_capturing(cudaStream_t handle);

// 内部模块不另加锁；所有调用都由 Pool 的同一把 mutex 串行保护。
// 引用的 config/stats/blocks/ready 由 Pool 持有，生命周期长于本对象。
class StreamManager {
  public:
    using StreamId = CachingCUDADeviceAllocator::StreamId;

    StreamManager(const CachingCUDAAllocatorConfig& config, CUDAMemoryPoolStats& stats,
                  BlockMap& blocks, Bins& ready);
    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;

    void check_device();
    int device() const noexcept { return device_; }
    Stream& stream(cudaStream_t handle, StreamId requested_id = 0);
    void unregister_stream(cudaStream_t handle);

    void retire(Stream& stream, Block& block);
    void detach_pending(Block& block) noexcept;
    void inherit_retired(Block& head, Block& tail) noexcept;

    void collect(bool bounded);
    void flush();
    void wait_all();

    // Pool 的 RAII 析构先选择设备、同步并释放显存，再调用这里。
    // 不在本模块的隐式析构中调用 CUDA，以保留 capture 中析构时的原有兜底行为。
    void destroy_events() noexcept;

  private:
    static void add_retired(Stream& stream, Block& block) noexcept;
    static void remove_retired(Stream& stream, Block& block) noexcept;
    void promote(Stream& stream, std::size_t limit);
    void query(Stream& stream, std::size_t limit);
    void seal(Stream& stream);
    void wait(Stream& stream);

    const CachingCUDAAllocatorConfig& config_;
    CUDAMemoryPoolStats& stats_;
    BlockMap& blocks_;
    Bins& ready_;
    int device_ = -1;

    // map 节点地址稳定；流身份由登记生命周期决定，不依赖驱动提供稳定 ID。
    std::map<cudaStream_t, Stream, std::less<cudaStream_t>> streams_;
    decltype(streams_)::iterator poll_ = streams_.end();
    StreamId next_stream_id_ = 1;
};

} // namespace infer::cuda_pool_detail
