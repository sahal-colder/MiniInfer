#pragma once

#include "mini_infer/ops/base/mini_infer_layer.h"
#include "mini_infer/core/mini_infer_tensor.h"
#include "mini_infer/ops/matmul/mini_infer_quantized_sgemv.h"

namespace mini_infer {

// ============================================================
// 带一个权重 Tensor 的 Matmul 算子
// ============================================================
//
// weight = [out_features, in_features]，按行存储。
// input  = [..., in_features]
// output = [..., out_features]
//
// 单个输入向量时：y[o] = sum(weight[o,i] * x[i])。
// 有多行输入时：Y = X * transpose(W)。这里没有 bias，也不做量化。
// CPU/CUDA 通过 get_matmul_kernel 统一选择；调用 CUDA 层前，
// 输入、输出和权重都必须是 CUDA Tensor。
class MatmulLayer final : public ParamLayer {
public:
    MatmulLayer(DeviceType device, int32_t out_features, int32_t in_features);

    // 复用 Layer::forward 的统一调用链：先 check，再 compute。
    void check() const override;

private:
    void compute(cudaStream_t stream) override;

    int32_t out_features_;
    int32_t in_features_;
};

// CPU/CUDA 的 Matmul 必须拥有相同签名，forward 才能只负责选择并调用函数指针。
using MatmulKernel = void (*)(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);

MatmulKernel get_matmul_kernel(DeviceType device);
void matmul_kernel_cpu(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);
void matmul_kernel_cuda(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);

// 量化算子的 Layer 入口，沿用 set_input/set_output/forward。
//
// QuantizedMatmulLayer 只负责把量化 GEMV 接入已有 Layer
// 生命周期：绑定参数、检查输入输出、调用对应设备的计算函数。
// 它不负责生成量化参数，也不负责读取模型文件。
// 原 ParamLayer::set_weight 仅允许 FP32，因此这里明确使用一个由
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
