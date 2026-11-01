// 第 14 课第 2 步：参数检查；不包含矩阵乘法或设备数据搬运。
#include "mini_infer/ops/mini_infer_quantized_sgemv.h"
#include <cmath>
#include <stdexcept>
namespace mini_infer {
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
