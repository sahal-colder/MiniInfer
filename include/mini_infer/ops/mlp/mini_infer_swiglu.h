#pragma once

// 第 17 课：SwiGLU = SiLU(gate) * up，其中 * 表示逐元素乘法。
// 先看 .cpp 的 CPU 标量循环，再看 .cu 的一线程一元素，最后对照
// _armadillo.cpp 的外部内存视图与 % 运算符；三个版本计算同一个公式。
#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {

// gate、up、output 必须同形状、同设备、FP32，无广播。
// 支持 output 与任意输入完全重合，包括 gate==up==output。
// 禁止错位的部分重叠：写 output[i] 不能破坏另一个线程将读取的元素。
// 非原位调用不会修改任何输入；这比课件 CPU 示例中无条件修改 input1
// 更明确。所有公开后端也会自行检查，直接调用不必先构造 Layer。
using SwiGLUKernel = void (*)(const Tensor&, const Tensor&, Tensor&, cudaStream_t);
SwiGLUKernel get_swiglu_kernel(DeviceType device);
void swiglu_kernel_cpu_reference(const Tensor& gate, const Tensor& up,
                                 Tensor& output, cudaStream_t stream = nullptr);
void swiglu_kernel_cpu(const Tensor& gate, const Tensor& up,
                       Tensor& output, cudaStream_t stream = nullptr);
void swiglu_kernel_cuda(const Tensor& gate, const Tensor& up,
                        Tensor& output, cudaStream_t stream = nullptr);

// 保留课件的动态共享内存版本用于对照。主入口使用寄存器即可：
// 每个线程只消费自己的两个数，共享内存并不能减少全局内存读取次数。
void swiglu_kernel_cuda_shared(const Tensor& gate, const Tensor& up,
                               Tensor& output, cudaStream_t stream = nullptr);

namespace detail {
void check_swiglu_tensors(const Tensor& gate, const Tensor& up,
                          const Tensor& output, DeviceType device);
}  // namespace detail

// 第 1～12 课的 LayerType 尚无 SWIGLU，保持其源码/API 不变。
// 本课使用独立的小包装，仍然遵循 forward -> check -> 后端函数指针。
// 它不持有工作区、无权重；Tensor 的 Buffer 由调用者负责保活。
class SwiGLULayer {
public:
    explicit SwiGLULayer(DeviceType device);
    void check(const Tensor& gate, const Tensor& up, const Tensor& output) const;
    void forward(const Tensor& gate, const Tensor& up, Tensor& output,
                 cudaStream_t stream = nullptr) const;

private:
    DeviceType device_;
};

// CPU 同步完成；CUDA 仅提交计算，包括默认 stream 也不代表已完成。
// 读取结果、跨 stream 使用或销毁 Tensor 前，应同步或建立 event 依赖。
}  // namespace mini_infer
