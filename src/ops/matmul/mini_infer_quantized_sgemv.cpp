#include "mini_infer/ops/matmul/mini_infer_quantized_sgemv.h"
#include <stdexcept>
#include <cmath>

namespace mini_infer {

void quantized_sgemv_kernel_cpu(const Tensor& input, const QuantizedTensor& weight,
                                Tensor& output, float output_scale, cudaStream_t) {
    detail::check_quantized_sgemv_tensors(input, weight, output, DeviceType::CPU,
                                         output_scale);
    const std::size_t in = static_cast<std::size_t>(weight.values.dim(1));
    const std::size_t out = static_cast<std::size_t>(weight.values.dim(0));
    const float* input_data = input.ptr<float>();
    const int8_t* quantized_values = weight.values.ptr<int8_t>();
    const float* scales = weight.scales.ptr<float>();
    float* output_data = output.ptr<float>();

    for (std::size_t row = 0; row < input.size() / in; ++row) {
        for (std::size_t o = 0; o < out; ++o) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < in; ++i) {
                const std::size_t weight_idx = o * in + i;
                const std::size_t group_idx = weight_idx / weight.group_size;
                // INT8 有符号值必须先转 float；计算仍是 FP32 乘加。
                const float restored_weight = static_cast<float>(quantized_values[weight_idx]) * scales[group_idx];
                sum += input_data[row * in + i] * restored_weight;
            }
            output_data[row * out + o] = sum * output_scale;
        }
    }
}

QuantizedSgemvKernel get_quantized_sgemv_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return quantized_sgemv_kernel_cpu;
        case DeviceType::CUDA: return quantized_sgemv_kernel_cuda;
        default: throw std::invalid_argument("Unknown quantized Sgemv backend");
    }
}

namespace detail {

void check_quantized_sgemv_weight(const QuantizedTensor& weight, DeviceType device) {
    if ((device != DeviceType::CPU && device != DeviceType::CUDA) ||
        weight.group_size == 0 || weight.values.empty() || weight.scales.empty() ||
        weight.values.dims().size() != 2 || weight.scales.dims().size() != 1 ||
        weight.values.data_type() != DataType::INT8 ||
        weight.scales.data_type() != DataType::FLOAT32 ||
        weight.values.device_type() != device || weight.scales.device_type() != device) {
        throw std::invalid_argument("Quantized Sgemv expects INT8 [out,in] and FP32 scales on its backend");
    }
    // 向上取整不能写成 (n + group_size - 1) / group_size：
    // group_size 很大时，加法可能溢出 size_t，导致错误的 scale 数量。
    const std::size_t groups = weight.values.size() / weight.group_size +
                              (weight.values.size() % weight.group_size != 0);
    if (weight.scales.size() != groups) {
        throw std::invalid_argument("Quantized Sgemv scale count does not match groups");
    }
    if (device == DeviceType::CPU) {
        for (std::size_t g = 0; g < groups; ++g) {
            const float scale = weight.scales.ptr<float>()[g];
            if (!std::isfinite(scale) || scale <= 0.0f) {
                throw std::invalid_argument("Quantized Sgemv scales must be finite and positive");
            }
        }
    }
}

void check_quantized_sgemv_tensors(const Tensor& input, const QuantizedTensor& weight,
                                   const Tensor& output, DeviceType device,
                                   float output_scale) {
    check_quantized_sgemv_weight(weight, device);
    if (!std::isfinite(output_scale)) {
        throw std::invalid_argument("Quantized Sgemv output_scale must be finite");
    }
    for (const Tensor* tensor : {&input, &output}) {
        if (tensor->empty() || tensor->dims().empty() ||
            tensor->data_type() != DataType::FLOAT32 || tensor->device_type() != device) {
            throw std::invalid_argument("Quantized Sgemv requires FP32 input/output on its backend");
        }
    }
    if (input.dims().back() != weight.values.dim(1)) {
        throw std::invalid_argument("Quantized Sgemv input width does not match weight");
    }
    std::vector<int32_t> expected = input.dims();
    expected.back() = weight.values.dim(0);
    if (output.dims() != expected) {
        throw std::invalid_argument("Quantized Sgemv output shape mismatch");
    }
    if (output.overlaps(input) || output.overlaps(weight.values) ||
        output.overlaps(weight.scales)) {
        throw std::invalid_argument("Quantized Sgemv output must not overlap its inputs");
    }
}

}  // namespace detail

}  // namespace mini_infer
