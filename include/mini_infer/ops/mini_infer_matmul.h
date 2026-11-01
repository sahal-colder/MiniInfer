#pragma once

#include "mini_infer/ops/mini_infer_layer.h"

namespace mini_infer {

// ============================================================
// 第 11 课：带一个权重 Tensor 的 Matmul 算子
// ============================================================
//
// weight = [out_features, in_features]，按行存储。
// input  = [..., in_features]
// output = [..., out_features]
//
// 单个输入向量时：y[o] = sum(weight[o,i] * x[i])。
// 有多行输入时：Y = X * transpose(W)。这里没有 bias，也不做量化。
// CPU/CUDA 通过 mini_infer_matmul_backend.h 统一选择；调用 CUDA 层前，
// 输入、输出和权重都必须是 CUDA Tensor。
class MatmulLayer final : public Layer {
public:
    MatmulLayer(DeviceType device, int32_t out_features, int32_t in_features);

    // 复用 Layer::forward 的统一调用链：先 check，再 compute。
    void check() const override;

private:
    void compute(cudaStream_t stream) override;

    int32_t out_features_;
    int32_t in_features_;
};

} // namespace mini_infer
