#include "mini_infer/ops/mini_infer_softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mini_infer {
SoftmaxKernel get_softmax_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return softmax_kernel_cpu;
        case DeviceType::CUDA: return softmax_kernel_cuda;
        default: throw std::invalid_argument("Unknown Softmax backend");
    }
}

SoftmaxLayer::SoftmaxLayer(DeviceType device) : device_(device) {
    (void)get_softmax_kernel(device);
}
void SoftmaxLayer::forward(const Tensor& input, Tensor& output, cudaStream_t stream) const {
    get_softmax_kernel(device_)(input, output, stream);
}
}  // namespace mini_infer
