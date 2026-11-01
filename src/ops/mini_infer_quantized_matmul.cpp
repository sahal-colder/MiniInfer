#include "mini_infer/ops/mini_infer_quantized_matmul.h"

#include <cmath>
#include <stdexcept>

namespace mini_infer {

QuantizedMatmulLayer::QuantizedMatmulLayer(DeviceType device, int32_t out_features,
                                         int32_t in_features, float output_scale)
    : Layer(device, 1, 1),
      out_features_(out_features), in_features_(in_features),
      output_scale_(output_scale) {
    if ((device != DeviceType::CPU && device != DeviceType::CUDA) ||
        out_features <= 0 || in_features <= 0 || !std::isfinite(output_scale)) {
        throw std::invalid_argument("Quantized Matmul needs a valid device, shape and output scale");
    }
}

void QuantizedMatmulLayer::set_quantized_weight(const QuantizedTensor& weight) {
    detail::check_quantized_sgemv_weight(weight, device_type());
    if (weight.values.dims() != std::vector<int32_t>{out_features_, in_features_}) {
        throw std::invalid_argument("Quantized Matmul weight shape must be [out,in]");
    }
    // Tensor 赋值共享两份 Buffer；调用者的局部 weight 析构后参数仍然有效。
    weight_ = weight;
}

void QuantizedMatmulLayer::check() const {
    // Layer 的输入/输出仍是 FP32；量化参数用专用检查同时验证 values/scales。
    detail::check_quantized_sgemv_tensors(get_input(0), weight_, get_output(0),
                                         device_type(), output_scale_);
}

void QuantizedMatmulLayer::compute(cudaStream_t stream) {
    get_quantized_sgemv_kernel(device_type())(get_input(0), weight_, output(0),
                                              output_scale_, stream);
}

}  // namespace mini_infer
