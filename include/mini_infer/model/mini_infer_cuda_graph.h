#pragma once

#include "mini_infer/model/mini_infer_transformer.h"

namespace mini_infer {
struct CUDAGraphStats {
    std::size_t graphs = 0, workspace_bytes = 0;
    std::uint64_t replays = 0;
};

// Per-layer, per-slot graphs share scratch and must execute serially on the
// pipeline's compute stream. The pipeline, blocks and weights must outlive us.
class CUDAGraphRunner {
  public:
    CUDAGraphRunner(const ModelConfig& config,
                   const std::vector<std::unique_ptr<TransformerBlock>>& blocks,
                   infer::AsyncKVPipeline& pipeline);
    ~CUDAGraphRunner();
    CUDAGraphRunner(const CUDAGraphRunner&) = delete;
    CUDAGraphRunner& operator=(const CUDAGraphRunner&) = delete;

    void prepare(const Tensor& input, int32_t position);
    void replay(std::size_t layer, infer::AsyncKVPipeline::LayerView kv, cudaStream_t stream);
    Tensor hidden() const;
    CUDAGraphStats stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mini_infer
