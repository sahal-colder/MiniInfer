#include "mini_infer/ops/mini_infer_add.h"

namespace mini_infer {
ElementwiseAddKernel get_elementwise_add_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return elementwise_add_cpu;
        case DeviceType::CUDA: return elementwise_add_cuda;
        default: throw std::invalid_argument("Unknown Add backend");
    }
}

ElementwiseAdd::ElementwiseAdd(DeviceType device) : device_(device) {
    (void)get_elementwise_add_kernel(device);
}
void ElementwiseAdd::forward(const Tensor& a, const Tensor& b, Tensor& output,
                             cudaStream_t stream) const {
    get_elementwise_add_kernel(device_)(a, b, output, stream);
}
}  // namespace mini_infer
