#include "mini_infer/ops/matmul/mini_infer_matmul.h"
#include <stdexcept>
#include "mini_infer/ops/matmul/mini_infer_sgemv.h"
#include <cmath>

namespace mini_infer {

MatmulLayer::MatmulLayer(DeviceType device, int32_t out_features, int32_t in_features)
    : ParamLayer(device, LayerType::MATMUL, "matmul", 1, 1, 1),
      out_features_(out_features), in_features_(in_features) {
    // ParamLayer 的最后三个参数分别表示：1 个输入、1 个输出、1 个权重。
    // 构造层只创建槽位，不分配权重，也不消耗权重文件的任何字节。
    if (out_features <= 0 || in_features <= 0) {
        throw std::invalid_argument("Matmul feature counts must be positive");
    }
}

void MatmulLayer::check() const {
    // 第一层检查：有实际存储、设备和算子一致、数据类型为 FP32。
    // 复用前几课的检查函数，不把文件读取逻辑塞进算子。
    const Tensor& x = get_input(0);
    const Tensor& w = get_weight(0);
    const Tensor& y = get_output(0);
    check_tensor(x);
    check_tensor(w);
    check_tensor(y);

    // 第二层检查：权重行数是输出维度，列数是输入维度。
    // [16,128] 和 [128,16] 元素数相等，但不是同一个算子的权重布局。
    if (w.dims() != std::vector<int32_t>{out_features_, in_features_} ||
        x.dims().back() != in_features_) {
        throw std::invalid_argument(
            "Matmul expects weight=[out,in] and input=[...,in]");
    }
    auto expected_output = x.dims();
    expected_output.back() = out_features_;
    if (y.dims() != expected_output) {
        throw std::invalid_argument("Matmul expects output=[...,out]");
    }

    // 第三层检查：输出不能覆盖输入或权重。
    // 否则写完 y[0] 后，后续行可能读到已被修改的 x 或 w。
    if (y.overlaps(x) || y.overlaps(w)) {
        throw std::invalid_argument("Matmul output must not overlap input or weight");
    }
}

void MatmulLayer::compute(cudaStream_t stream) {
    get_matmul_kernel(device_type())(get_input(0), get_weight(0), output(0), 1.0f, stream);
}

void matmul_kernel_cpu(const Tensor& input, const Tensor& weight, Tensor& output,
                       float scale, cudaStream_t stream) {
    // Matmul 与独立 GEMV 入口共享计算实现。
    sgemv_kernel_cpu(input, weight, output, scale, stream);
}

MatmulKernel get_matmul_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return matmul_kernel_cpu;
        case DeviceType::CUDA: return matmul_kernel_cuda;
        default: throw std::invalid_argument("Unknown Matmul backend");
    }
}

void matmul_kernel_cuda(const Tensor& input, const Tensor& weight, Tensor& output,
                        float scale, cudaStream_t stream) {
    // MatmulLayer/后端选择器复用 CUDA GEMV 的归约实现。
    sgemv_kernel_cuda(input, weight, output, scale, stream);
}

QuantizedMatmulLayer::QuantizedMatmulLayer(DeviceType device, int32_t out_features,
                                         int32_t in_features, float output_scale)
    : Layer(device, LayerType::MATMUL, "quantized_matmul", 1, 1),
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
