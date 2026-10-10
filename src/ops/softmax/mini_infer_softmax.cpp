#include "mini_infer/ops/softmax/mini_infer_softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mini_infer {
namespace detail {
void check_softmax_tensors(const Tensor& input, const Tensor& output,
                           DeviceType device) {
    for (const Tensor* t : {&input, &output}) {
        if (t->empty() || t->size() == 0 || t->dims().empty() ||
            t->data_type() != DataType::FLOAT32 || t->device_type() != device) {
            throw std::invalid_argument("Softmax requires nonempty FP32 tensors on its backend");
        }
    }
    if (input.dims() != output.dims()) {
        throw std::invalid_argument("Softmax input/output shapes must match");
    }
    if (output.overlaps(input) && output.ptr<float>() != input.ptr<float>()) {
        throw std::invalid_argument("Softmax forbids partially overlapping output");
    }
}

void softmax_rows_cpu(const float* input, float* output, std::size_t rows,
                      int32_t width, int32_t stride) {
    for (std::size_t row = 0; row < rows; ++row) {
        const float* x = input + row * stride;
        float* y = output + row * stride;
        // 第一遍只读。得到最大值后再写，才能安全地做原位 Softmax。
        float max_value = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < width; ++i) max_value = std::max(max_value, x[i]);
        float sum = 0.0f;
        for (int32_t i = 0; i < width; ++i) {
            y[i] = std::exp(x[i] - max_value);
            sum += y[i];
        }
        for (int32_t i = 0; i < width; ++i) y[i] /= sum;
    }
}
}  // namespace detail

void softmax_kernel_cpu(const Tensor& input, Tensor& output, cudaStream_t stream) {
    detail::check_softmax_tensors(input, output, DeviceType::CPU);
    if (stream) throw std::invalid_argument("CPU Softmax does not accept a CUDA stream");
    const int32_t width = input.dims().back();
    detail::softmax_rows_cpu(input.ptr<float>(), output.ptr<float>(),
                             input.size() / width, width, width);
}

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
