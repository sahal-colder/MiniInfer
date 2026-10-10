#pragma once

// 第 18 课阅读顺序：本接口 -> .cpp 的三个循环 -> .cu 的块内归约。
#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {

// 对最后一维做 FP32 Softmax，前面的维度都合并成 rows。
// [width] 是一行，[heads,width] 是 heads 行，[a,b,width] 是 a*b 行。
//   m = max(x); e[i] = exp(x[i]-m); y[i] = e[i]/sum(e)
// 先减最大值，避免较大的有限输入使 exp 上溢。
// 输入应为有限值；允许部分位置为 -inf 作掩码，但每行至少一个有限值。
// NaN、+inf、全为 -inf 的行不在数值支持范围内。
// 允许 output 与 input 完全重合；拒绝错位/部分重叠。
// CUDA 调用异步：Tensor 必须活到 stream 完成，读回前需要同步。
using SoftmaxKernel = void (*)(const Tensor&, Tensor&, cudaStream_t);
SoftmaxKernel get_softmax_kernel(DeviceType device);
void softmax_kernel_cpu(const Tensor& input, Tensor& output,
                        cudaStream_t stream = nullptr);
void softmax_kernel_cuda(const Tensor& input, Tensor& output,
                         cudaStream_t stream = nullptr);

class SoftmaxLayer {
public:
    explicit SoftmaxLayer(DeviceType device);
    void forward(const Tensor& input, Tensor& output,
                 cudaStream_t stream = nullptr) const;
private:
    DeviceType device_;
};

namespace detail {
void check_softmax_tensors(const Tensor&, const Tensor&, DeviceType);

// Attention 的 score 每行预留 max_seq_len，但只归一化前 length 个数。
// stride 是两行起点的元素间距，width 是本次参与 Softmax 的有效长度。
// 这些内部函数假定调用者已验证存储/边界；尾部 [width,stride) 不改动。
void softmax_rows_cpu(const float* input, float* output, std::size_t rows,
                      int32_t width, int32_t stride);
void softmax_rows_cuda(const float* input, float* output, std::size_t rows,
                       int32_t width, int32_t stride, cudaStream_t stream);
}  // namespace detail
}  // namespace mini_infer
