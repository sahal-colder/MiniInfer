#include "mini_infer/model/mini_infer_cuda_graph.h"
#include "infer/cuda/cuda_memory_pool.h"

#include <array>
#include <stdexcept>

namespace mini_infer {
namespace {
// Tensor handles borrow graph-owned storage. Dropping a host lease cannot
// reuse this bump-allocated slice; only destruction of every graph frees it.
class GraphWorkspaceAllocator final : public DeviceAllocator {
  public:
    GraphWorkspaceAllocator(infer::CachingCUDADeviceAllocator& pool, cudaStream_t stream)
        : DeviceAllocator(DeviceType::CUDA), pool_(pool), stream_(stream) {}
    void* allocate(std::size_t bytes) const override {
        auto* ptr = pool_.allocate(bytes, stream_);
        pool_.release(ptr, stream_);
        return ptr;
    }
    void release(void*) const override {}

  private:
    infer::CachingCUDADeviceAllocator& pool_;
    cudaStream_t stream_;
};
}

struct CUDAGraphRunner::Impl {
    struct Graph {
        cudaGraph_t captured = nullptr;
        cudaGraphExec_t executable = nullptr;
    };
    infer::CachingCUDADeviceAllocator pool;
    infer::PinnedHostAllocator pinned;
    cudaStream_t stream;
    int device = 0;
    int32_t capacity;
    std::vector<infer::AsyncKVPipeline::LayerView> slots;
    bool resident;
    std::vector<Graph> graphs;
    Tensor hidden;
    int32_t* host_position = nullptr;
    int32_t* device_position = nullptr;
    cudaEvent_t parameters_copied = nullptr;
    std::uint64_t replays = 0;

    Impl(const ModelConfig& config, infer::AsyncKVPipeline& pipeline)
        : stream(pipeline.compute_stream()), capacity(config.seq_len),
          resident(pipeline.gpu_resident()) {
        for (std::size_t i=0;i<(resident ? pipeline.layer_count() : 2);++i) slots.push_back(pipeline.graph_slot(i));
        check_cuda(cudaGetDevice(&device), "get graph device");
    }

    void initialize(const ModelConfig& config,
                    const std::vector<std::unique_ptr<TransformerBlock>>& blocks) {
        const auto aligned = [](std::size_t bytes) { return (bytes + 255) & ~std::size_t(255); };
        const auto kv_dim = config.kv_head_num * (config.dim / config.head_num);
        // Eight dim vectors, two FFN vectors, scores, K/V scratch and position.
        pool.reserve_graph_workspace(8 * aligned(std::size_t(config.dim) * sizeof(float)) +
            2 * aligned(std::size_t(config.hidden_dim) * sizeof(float)) +
            aligned(std::size_t(config.head_num) * config.seq_len * sizeof(float)) +
            2 * aligned(std::size_t(kv_dim) * sizeof(float)) + aligned(sizeof(int32_t)));
        host_position = static_cast<int32_t*>(pinned.allocate(sizeof(int32_t)));
        check_cuda(cudaEventCreateWithFlags(&parameters_copied, cudaEventDisableTiming),
                   "create graph parameter event");
        graphs.reserve(blocks.size() * slots.size());
        auto allocator = std::make_shared<GraphWorkspaceAllocator>(pool, stream);
        for (std::size_t layer=0;layer<blocks.size();++layer) {
            const auto& block=blocks[layer];
            for (std::size_t index=0;index<(resident ? 1 : slots.size());++index) {
                const auto& slot=slots[resident ? layer : index];
                graphs.emplace_back();
                infer::GraphCaptureScope capture(pool, graphs.size(), stream);
                TransformerWorkspace work(config, allocator);
                Tensor state(DataType::FLOAT32, {config.dim}, allocator);
                Tensor key(DataType::FLOAT32, {kv_dim}, allocator);
                Tensor value(DataType::FLOAT32, {kv_dim}, allocator);
                auto* position = static_cast<int32_t*>(allocator->allocate(sizeof(int32_t)));
                block->forward_paged(state, state, key, value, slot.pages,
                                     slot.tokens_per_page, 0, work, stream, position);
                auto& graph = graphs.back();
                graph.captured = capture.finish();
                check_cuda(cudaGraphInstantiate(&graph.executable, graph.captured, nullptr, nullptr, 0),
                           "instantiate layer graph");
                check_cuda(cudaGraphUpload(graph.executable, stream), "upload layer graph");
                hidden = Tensor::borrow(DataType::FLOAT32, {config.dim}, state.ptr<float>(), DeviceType::CUDA);
                device_position = position;
            }
        }
    }

    ~Impl() {
        int previous = -1;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        // The stream belongs to the still-live pipeline. Wait before destroying
        // GraphExec, then forget graphs, then let pool destroy the workspace.
        cudaStreamSynchronize(stream);
        for (std::size_t i = 0; i < graphs.size(); ++i) {
            auto& graph = graphs[i];
            if (graph.executable) cudaGraphExecDestroy(graph.executable);
            if (!graph.captured) continue;
            cudaGraphDestroy(graph.captured);
            pool.forget_graph(i + 1);
        }
        if (parameters_copied) cudaEventDestroy(parameters_copied);
        if (host_position) pinned.release(host_position);
        // Release while our device is selected, before restoring the caller's.
        pool.release_graph_workspace();
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
};

CUDAGraphRunner::CUDAGraphRunner(const ModelConfig& config,
    const std::vector<std::unique_ptr<TransformerBlock>>& blocks, infer::AsyncKVPipeline& pipeline)
    : impl_(std::make_unique<Impl>(config, pipeline)) {
    impl_->initialize(config, blocks);
}
CUDAGraphRunner::~CUDAGraphRunner() = default;

void CUDAGraphRunner::prepare(const Tensor& input, int32_t position) {
    auto& s = *impl_;
    if (position < 0 || position >= s.capacity || input.dims() != s.hidden.dims() ||
        input.device_type() != DeviceType::CUDA || input.data_type() != DataType::FLOAT32)
        throw std::invalid_argument("Invalid graph input or position");
    // The host must not overwrite pinned memory still read by a prior H2D.
    check_cuda(cudaEventSynchronize(s.parameters_copied), "wait for graph parameter copy");
    *s.host_position = position;
    check_cuda(cudaMemcpyAsync(s.device_position, s.host_position, sizeof(int32_t),
                               cudaMemcpyHostToDevice, s.stream), "update graph position");
    check_cuda(cudaEventRecord(s.parameters_copied, s.stream), "record graph parameter copy");
    s.hidden.copy_from(input, s.stream);
}

void CUDAGraphRunner::replay(std::size_t layer, infer::AsyncKVPipeline::LayerView kv, cudaStream_t stream) {
    auto& s = *impl_;
    const auto slot = layer % s.slots.size();
    if (stream != s.stream || kv.pages != s.slots[slot].pages)
        throw std::invalid_argument("Graph replay requires its fixed slot and compute stream");
    // AsyncKVPipeline enqueues ready -> this launch -> compute_done. Transfers
    // and NVMe admission never enter capture or share the scratch workspace.
    check_cuda(cudaGraphLaunch(s.graphs.at(s.resident ? layer : layer * s.slots.size() + slot).executable, stream),
               "replay layer graph");
    ++s.replays;
}

Tensor CUDAGraphRunner::hidden() const { return impl_->hidden; }
CUDAGraphStats CUDAGraphRunner::stats() const {
    const auto memory = impl_->pool.stats();
    return {memory.graph_count, memory.graph_reserved_bytes, impl_->replays};
}
} // namespace mini_infer
