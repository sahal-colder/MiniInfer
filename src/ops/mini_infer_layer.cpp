#include "mini_infer/ops/mini_infer_layer.h"
#include "mini_infer/ops/mini_infer_kernels.h"
#include <cmath>
#include <utility>
namespace mini_infer {
Layer::Layer(DeviceType device, std::size_t inputs, std::size_t outputs, std::size_t weights)
    : device_(device), inputs_(inputs), outputs_(outputs), weights_(weights) {
    if (device != DeviceType::CPU && device != DeviceType::CUDA)
        throw std::invalid_argument("Layer needs a known device");
}
void Layer::check_tensor(const Tensor& value) const {
    if (value.empty()) {
        throw std::invalid_argument("Layer tensor has no storage");
    }
    if (value.device_type() != device_) {
        throw std::invalid_argument("Layer device mismatch");
    }
    if (value.data_type() != DataType::FLOAT32) {
        throw std::invalid_argument("Layer requires FP32");
    }
}
void Layer::set_input(std::size_t i, const Tensor& value) {
    auto& slot = inputs_.at(i);
    check_tensor(value);
    slot = value;
}
void Layer::set_output(std::size_t i, const Tensor& value) {
    auto& slot = outputs_.at(i);
    check_tensor(value);
    slot = value;
}
void Layer::forward(cudaStream_t stream) {
    check();
    if (device_ == DeviceType::CPU && stream) {
        throw std::invalid_argument("CPU layer does not accept a CUDA stream");
    }
    compute(stream);
}
void Layer::set_weight(std::size_t i, const Tensor& value) {
    auto& slot = weights_.at(i);
    check_tensor(value);
    slot = value;
}
AddLayer::AddLayer(DeviceType device) : Layer(device, 2, 1) {}
void AddLayer::check() const {
    const auto& a = get_input(0);
    const auto& b = get_input(1);
    const auto& y = get_output(0);
    if (a.empty() || b.empty() || y.empty())
        throw std::invalid_argument("Layer tensor has no storage");
    if (a.dims() != b.dims() || a.dims() != y.dims()) {
        throw std::invalid_argument("Add requires equal shapes; no broadcasting");
    }
    if (y.overlaps(a) || y.overlaps(b)) {
        throw std::invalid_argument("Add output must not overlap inputs");
    }
}
void AddLayer::compute(cudaStream_t stream) {
    get_add_kernel(device_type())(get_input(0), get_input(1), output(0), stream);
}
RMSNormLayer::RMSNormLayer(DeviceType device, float eps, RMSNormKernelMode mode)
    : Layer(device, 1, 1, 1), eps_(eps), kernel_mode_(mode) {
    if (!std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument("RMSNorm eps must be finite and positive");
}
void RMSNormLayer::check() const {
    const auto& x = get_input(0);
    const auto& w = get_weight(0);
    const auto& y = get_output(0);
    if (x.empty() || w.empty() || y.empty())
        throw std::invalid_argument("Layer tensor has no storage");
    if (x.dims() != y.dims() || w.dims().size() != 1 || w.dim(0) != x.dims().back()) {
        throw std::invalid_argument("RMSNorm expects x=[...,D], w=[D], y=shape(x)");
    }
    if (y.overlaps(x) || y.overlaps(w)) {
        throw std::invalid_argument("RMSNorm output must not overlap input or weight");
    }
}
void RMSNormLayer::compute(cudaStream_t stream) {
    get_rmsnorm_kernel(device_type(), kernel_mode_)(get_input(0), get_weight(0), output(0), eps_,
                                                    stream);
}
AddKernel get_add_kernel(DeviceType device) {
    switch (device) {
    case DeviceType::CPU:
        return add_kernel_cpu;
    case DeviceType::CUDA:
        return add_kernel_cuda;
    default:
        throw std::invalid_argument("Unknown Add backend");
    }
}
RMSNormKernel get_rmsnorm_kernel(DeviceType device) {
    return get_rmsnorm_kernel(device, RMSNormKernelMode::BLOCK);
}
RMSNormKernel get_rmsnorm_kernel(DeviceType device, RMSNormKernelMode mode) {
    switch (device) {
    case DeviceType::CPU:
        return rmsnorm_kernel_cpu;
    case DeviceType::CUDA:
        switch (mode) {
        case RMSNormKernelMode::WARP:
            return rmsnorm_kernel_cuda_warp;
        case RMSNormKernelMode::BLOCK:
            return rmsnorm_kernel_cuda;
        case RMSNormKernelMode::VECTOR4:
            return rmsnorm_kernel_cuda_vectorized;
        }
        throw std::invalid_argument("Unknown RMSNorm kernel mode");
    default:
        throw std::invalid_argument("Unknown RMSNorm backend");
    }
}
} // namespace mini_infer
