#pragma once

#include "mini_infer/ops/mini_infer_layer.h"
#include "mini_infer/ops/mini_infer_quantized_sgemv.h"

namespace mini_infer {

// 第 14 课量化算子的 Layer 入口，沿用 set_input/set_output/forward。
//
// QuantizedMatmulLayer 只负责把第 14 课的量化 GEMV 接入已有 Layer
// 生命周期：绑定参数、检查输入输出、调用对应设备的计算函数。
// 它不负责生成量化参数，也不负责读取模型文件。
// 原 Layer::set_weight 仅允许 FP32，因此这里明确使用一个由
// INT8 values + FP32 scales + group_size 组成的参数，不放宽旧层的检查。
// Tensor 赋值共享 Buffer；层会保活两个权重 Tensor 的自有存储。
class QuantizedMatmulLayer final : public Layer {
public:
    QuantizedMatmulLayer(DeviceType device, int32_t out_features,
                        int32_t in_features, float output_scale = 1.0f);

    // CPU 权重绑定 CUDA 层前必须先执行 weight.to_cuda(allocator)。
    // 检查通过后才更新参数；绑定失败保留原有权重。
    void set_quantized_weight(const QuantizedTensor& weight);
    const QuantizedTensor& get_quantized_weight() const { return weight_; }
    void check() const override;

private:
    void compute(cudaStream_t stream) override;
    int32_t out_features_;
    int32_t in_features_;
    float output_scale_;
    QuantizedTensor weight_;
};

}  // namespace mini_infer
