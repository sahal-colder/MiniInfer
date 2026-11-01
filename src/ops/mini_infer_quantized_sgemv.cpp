// 第 14 课第 1 步：只读 CPU 内积，关注 weight_idx/group_size。
#include "mini_infer/ops/mini_infer_quantized_sgemv.h"

#include <stdexcept>

namespace mini_infer {
QuantizedSgemvKernel get_quantized_sgemv_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return quantized_sgemv_kernel_cpu;
        case DeviceType::CUDA: return quantized_sgemv_kernel_cuda;
        default: throw std::invalid_argument("Unknown quantized Sgemv backend");
    }
}

}  // namespace mini_infer
