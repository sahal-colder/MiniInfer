#pragma once

// 第 19 课阅读顺序：本接口 -> .cpp 的逐元素加法 -> .cu 的线程编号。
// Embedding/Softmax 已在独立的 mini_infer_embedding.* / mini_infer_softmax.*。
#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {

// FP32 逐元素加法：output[i] = a[i] + b[i]。
// 三个 Tensor 必须非空、同设备、同形状；支持 [dim]、[seq,dim] 等。
// 不做广播，output 不能与任一输入重叠；a 和 b 可以共享存储。
// 这一规则与原有 AddLayer 一致，残差计算应准备独立的输出 Tensor。
// CUDA 异步提交；输入/输出需存活到 stream 完成，读回前先同步。
using ElementwiseAddKernel = void (*)(const Tensor&, const Tensor&, Tensor&, cudaStream_t);
ElementwiseAddKernel get_elementwise_add_kernel(DeviceType device);
void elementwise_add_cpu(const Tensor& a, const Tensor& b, Tensor& output,
                         cudaStream_t stream = nullptr);
void elementwise_add_cuda(const Tensor& a, const Tensor& b, Tensor& output,
                          cudaStream_t stream = nullptr);

// 提供与 EmbeddingLayer/SoftmaxLayer 一样直接接收 Tensor 的学习入口。
// 原有 AddLayer 的声明、实现和绑定方式继续保留。
class ElementwiseAdd {
public:
    explicit ElementwiseAdd(DeviceType device);
    void forward(const Tensor& a, const Tensor& b, Tensor& output,
                 cudaStream_t stream = nullptr) const;
private:
    DeviceType device_;
};

namespace detail {
// The low-level residual kernel permits exact in-place writes, never shifted overlap.
void check_elementwise_add_tensors(const Tensor&, const Tensor&, const Tensor&, DeviceType,
                                   bool allow_inplace = false);
}  // namespace detail
}  // namespace mini_infer
