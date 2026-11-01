#pragma once

#include "mini_infer/ops/mini_infer_sgemv.h"

#include "mini_infer/tensor/mini_infer_tensor.h"

namespace mini_infer {

// CPU/CUDA 的 Matmul 必须拥有相同签名，forward 才能只负责选择并调用函数指针。
using MatmulKernel = SgemvKernel;

MatmulKernel get_matmul_kernel(DeviceType device);
void matmul_kernel_cpu(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);
void matmul_kernel_cuda(const Tensor&, const Tensor&, Tensor&, float, cudaStream_t);

}  // namespace mini_infer
