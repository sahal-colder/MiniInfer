#pragma once

// FP32 GEMV：参数检查、CPU 标量/Armadillo 实现在 .cpp；
// CUDA CUB、手写归约和每块四行变体在 .cu 中共享点积实现。

#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {

// ============================================================
// SGEMV，单精度矩阵与向量相乘
// ============================================================
// 本文件只负责 FP32 Sgemv 的公开接口、后端选择和参数检查。
// 它不负责模型文件读取、INT8 量化、KV Cache 或完整推理循环。
//
// S = single precision（float），GEMV = general matrix-vector multiply。
// 使用公式 y[o] = scale * sum_i(W[o,i] * x[i])，没有 bias，
// 每次覆盖 output；并非带 alpha/beta、转置和步长参数的完整 BLAS 接口。
//
// Tensor 仍按连续行主序存储：
//   weight = [out_features, in_features]
//   input  = [in_features]       -> output = [out_features]
//   input  = [..., in_features]  -> output = [..., out_features]
// 多个输入向量时，对每个向量独立执行 GEMV，复用同一份权重。
// MatmulLayer 的多行输入复用同一接口。
//
// 三个 Tensor 必须有存储、都是 FLOAT32，并位于指定后端；output
// 不得与 input/weight 重叠。算子只借用 Tensor，不取得 Buffer 所有权：
// 调用者负责保证这些 Tensor 在 CPU 计算结束或 CUDA stream 完成前有效。
using SgemvKernel = void (*)(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);

// CPU/CUDA 函数签名一致，选择器只返回函数指针。
SgemvKernel get_sgemv_kernel(DeviceType device);

// CPU：用 Armadillo 的外部内存视图处理行/列主序转换。
// CMake 未找到 Armadillo（或显式关闭）时，回退到下面的参考循环。
// CPU 运算在返回前完成；stream 参数仅用于统一后端签名。
void sgemv_kernel_cpu(const Tensor& input, const Tensor& weight, Tensor& output,
                     float scale, cudaStream_t stream);

// 独立标量参考实现，用于数值对照，不依赖 Armadillo。
void sgemv_kernel_cpu_reference(const Tensor& input, const Tensor& weight,
                               Tensor& output, float scale, cudaStream_t stream);

// CUDA：128 线程处理一行，float4/标量访存 + CUB BlockReduce。
// kernel 提交到传入的 stream，返回时不等待 GPU 完成；即便使用默认
// stream，也要在读取结果或释放 Tensor 前同步（或同步拷回 CPU）。
void sgemv_kernel_cuda(const Tensor& input, const Tensor& weight, Tensor& output,
                      float scale, cudaStream_t stream);

// 不依赖 CUB 的两级归约：warp 内 shuffle 求和，
// 各 warp 的 lane 0 写共享数组，再由 warp 0 合并。这是可运行的对照。
void sgemv_kernel_cuda_manual(const Tensor& input, const Tensor& weight, Tensor& output,
                              float scale, cudaStream_t stream);

// ROW_PER_BLOCK=4：每个 block 顺序算最多四个输出元素。
void sgemv_kernel_cuda_rows4(const Tensor& input, const Tensor& weight, Tensor& output,
                             float scale, cudaStream_t stream);

namespace detail {
// CPU/CUDA wrapper 共用检查；只读 shape/dtype/device 等主机元数据，
// 不解引用 GPU 指针。先检查再取 dims().back()，避免空 Tensor 越界。
void check_sgemv_tensors(const Tensor& input, const Tensor& weight,
                        const Tensor& output, DeviceType device);
}  // namespace detail

}  // namespace mini_infer
