#include "mini_infer/ops/mini_infer_swiglu.h"

#include <cmath>
#include <stdexcept>

namespace mini_infer {
SwiGLUKernel get_swiglu_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return swiglu_kernel_cpu;
        case DeviceType::CUDA: return swiglu_kernel_cuda;
        default: throw std::invalid_argument("Unknown SwiGLU backend");
    }
}

SwiGLULayer::SwiGLULayer(DeviceType device) : device_(device) {
    (void)get_swiglu_kernel(device);  // 构造时拒绝 UNKNOWN，而非计算后才发现。
}

void SwiGLULayer::check(const Tensor& gate, const Tensor& up,
                        const Tensor& output) const {
    detail::check_swiglu_tensors(gate, up, output, device_);
}

void SwiGLULayer::forward(const Tensor& gate, const Tensor& up, Tensor& output,
                          cudaStream_t stream) const {
    check(gate, up, output);
    get_swiglu_kernel(device_)(gate, up, output, stream);
}
}  // namespace mini_infer
