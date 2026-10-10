#pragma once
#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {
// 内部计算接口：由 Layer 检查好 dtype/shape/device/别名后再调用。
using AddKernel = void (*)(const Tensor&, const Tensor&, Tensor&, cudaStream_t);
using RMSNormKernel = void (*)(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);

// 第 6/8 课用三个实现比较同一个数学算子：
//   WARP     : 一个 warp 处理一行，作为简单基线；
//   BLOCK    : 多 warp 协作一行，避免重复工作；
//   VECTOR4  : 在 block 版本的基础上尝试 float4 访存，未对齐行自动回退。
enum class RMSNormKernelMode {
    WARP,
    BLOCK,
    VECTOR4
};

// Add kernels allow exact input/output aliasing for residual updates; shifted
// overlap is rejected. AddLayer and ElementwiseAdd require separate outputs.
AddKernel get_add_kernel(DeviceType device);
// 保留第 3～5 课的单参数符号；带 mode 的重载用于课程 6/8 对比。
RMSNormKernel get_rmsnorm_kernel(DeviceType device);
RMSNormKernel get_rmsnorm_kernel(DeviceType device, RMSNormKernelMode mode);
void add_kernel_cpu(const Tensor& a, const Tensor& b, Tensor& y, cudaStream_t stream);
void add_kernel_cuda(const Tensor& a, const Tensor& b, Tensor& y, cudaStream_t stream);
void rmsnorm_kernel_cpu(const Tensor& x, const Tensor& w, Tensor& y, float eps, cudaStream_t stream);
void rmsnorm_kernel_cuda(const Tensor& x, const Tensor& w, Tensor& y, float eps, cudaStream_t stream);

// 用于对照的单 warp 版本；与其他 kernel 一样，需要先验证形状等前提。
void rmsnorm_kernel_cuda_warp(const Tensor& x, const Tensor& w, Tensor& y,
                              float eps, cudaStream_t stream);
void rmsnorm_kernel_cuda_vectorized(const Tensor& x, const Tensor& w, Tensor& y,
                                    float eps, cudaStream_t stream);
}  // namespace mini_infer
