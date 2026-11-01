#include "mini_infer/ops/mini_infer_matmul.h"
#include "mini_infer/ops/mini_infer_matmul_backend.h"

#include <stdexcept>

namespace mini_infer {

MatmulLayer::MatmulLayer(DeviceType device, int32_t out_features, int32_t in_features)
    : Layer(device, 1, 1, 1),
      out_features_(out_features), in_features_(in_features) {
    // Layer 的最后三个参数分别表示：1 个输入、1 个输出、1 个权重。
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
    if (x.empty() || w.empty() || y.empty())
        throw std::invalid_argument("Layer tensor has no storage");

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

} // namespace mini_infer
