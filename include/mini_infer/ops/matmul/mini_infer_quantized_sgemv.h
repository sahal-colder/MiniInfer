#pragma once

// INT8 权重 GEMV：.cpp 包含检查和 CPU 实现，.cu 提供 CUDA 实现。
// QuantizedMatmulLayer 在 mini_infer_matmul.h 中提供 Layer 绑定接口。

#include "mini_infer/weights/mini_infer_quantization.h"

namespace mini_infer {

// ============================================================
// FP32 输入 × INT8 权重，输出仍为 FP32
// ============================================================
// 本文件只负责量化权重 GEMV 的计算接口和参数检查。
// 计算过程：
// “读取 INT8 值 -> 按扁平 group 找 scale -> 恢复 float -> 累加”。
// 它不负责模型文件格式，也不负责完整 Transformer。
// 只量化权重。input/output 不转换为 INT8，累加器也是 float。
// 对 weight.values=[out,in] 的每个元素，边读取边恢复其浮点权重：
//   weight_idx = o * in + i
//   group_idx  = weight_idx / group_size
//   r_weight   = float(values[weight_idx]) * scales[group_idx]
//   y[o]      += x[i] * r_weight
// 最后 y[o] *= output_scale。output_scale 与每组 scales 是两个概念。
//
// 分组覆盖整个行主序权重数组，不在每行重新开始；组允许跨行，
// 最后不足 group_size 个权重的尾组也有自己的 scale。
// input=[...,in] -> output=[...,out]，每个向量独立使用同一份权重。
// CUDA 直接读取 INT8 权重，不先生成一份完整的 FP32 权重矩阵。
using QuantizedSgemvKernel = void (*)(const Tensor&, const QuantizedTensor&,
                                     Tensor&, float, cudaStream_t);

QuantizedSgemvKernel get_quantized_sgemv_kernel(DeviceType device);
void quantized_sgemv_kernel_cpu(const Tensor& input, const QuantizedTensor& weight,
                                Tensor& output, float output_scale, cudaStream_t stream);
void quantized_sgemv_kernel_cuda(const Tensor& input, const QuantizedTensor& weight,
                                 Tensor& output, float output_scale, cudaStream_t stream);

// ROW_PER_BLOCK 的分工：一个 block 顺序处理最多四个输出。
// 最后不足四个时要截断；每次复用 CUB 工作区前必须同步。
void quantized_sgemv_kernel_cuda_rows4(const Tensor& input,
                                       const QuantizedTensor& weight, Tensor& output,
                                       float output_scale, cudaStream_t stream);

// 三个读取对象 input/values/scales 与 output 必须位于同一后端。
// output 不可覆盖其中任何一个。CPU 返回时完成；CUDA 提交到 stream 后
// 异步返回，调用者负责同步后再读取结果或释放 Tensor。
// scales 的数值要求有限且严格为正。CPU 会检查数值；CUDA wrapper
// 只检查主机元数据，不为检查 scale 而隐藏一次设备到主机复制/同步。
// 正常路径由 quantize_symmetric 生成合法 scales，再一起上传 GPU。
namespace detail {
void check_quantized_sgemv_weight(const QuantizedTensor& weight, DeviceType device);
void check_quantized_sgemv_tensors(const Tensor& input, const QuantizedTensor& weight,
                                   const Tensor& output, DeviceType device,
                                   float output_scale);
}  // namespace detail

}  // namespace mini_infer
