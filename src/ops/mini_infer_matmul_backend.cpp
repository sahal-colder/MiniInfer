#include "mini_infer/ops/mini_infer_matmul_backend.h"
#include "mini_infer/ops/mini_infer_sgemv.h"

#include <stdexcept>

namespace mini_infer {

void matmul_kernel_cpu(const Tensor& input, const Tensor& weight, Tensor& output,
                       float scale, cudaStream_t stream) {
    // 保留第 12 课函数名/签名；第 13 课计算实现从独立文件中复用。
    sgemv_kernel_cpu(input, weight, output, scale, stream);
}

MatmulKernel get_matmul_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return matmul_kernel_cpu;
        case DeviceType::CUDA: return matmul_kernel_cuda;
        default: throw std::invalid_argument("Unknown Matmul backend");
    }
}

}  // namespace mini_infer
