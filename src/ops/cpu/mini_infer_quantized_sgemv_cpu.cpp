// 第 14 课第 1 步：只读 CPU 内积，关注 weight_idx/group_size。
#include "mini_infer/ops/mini_infer_quantized_sgemv.h"

#include <stdexcept>

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

}  // namespace mini_infer
