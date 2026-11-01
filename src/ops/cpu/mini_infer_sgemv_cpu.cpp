// 第 13 课第 1 步：先手写 compute_reference，再读两个 CPU 入口。
// weight=[out,in]，input=[...,in]，output=[...,out]；不涉及模型文件。
#include "mini_infer/ops/mini_infer_sgemv.h"
namespace mini_infer {
namespace detail {
#if defined(MINI_INFER_HAS_ARMADILLO)
// 只在启用数学库时连接到 mini_infer_sgemv_armadillo.cpp。
void compute_sgemv_armadillo(const Tensor&, const Tensor&, Tensor&, float);
#endif
}  // namespace detail
namespace {

// 内部计算函数假定外层已检查。CPU fallback 不需要重复做检查。
void compute_reference(const Tensor& input, const Tensor& weight, Tensor& output,
                       float scale) {
    const std::size_t in = static_cast<std::size_t>(weight.dim(1));
    const std::size_t out = static_cast<std::size_t>(weight.dim(0));
    const std::size_t rows = input.size() / in;
    const float* input_data = input.ptr<float>();
    const float* weight_data = weight.ptr<float>();
    float* output_data = output.ptr<float>();

    // row：第几个输入向量；o：权重第几行；i：向量第几个元素。
    // 例如 W=[2,3] 时，第二行权重从 w[1*3] 开始，而非 w[1*2]。
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t o = 0; o < out; ++o) {
            float sum = 0.0f;  // 每个输出都从零累加，不读取原来的 y。
            for (std::size_t i = 0; i < in; ++i) {
                sum += weight_data[o * in + i] * input_data[row * in + i];
            }
            output_data[row * out + o] = sum * scale;
        }
    }
}

}  // namespace

void sgemv_kernel_cpu_reference(const Tensor& input, const Tensor& weight,
                               Tensor& output, float scale, cudaStream_t) {
    detail::check_sgemv_tensors(input, weight, output, DeviceType::CPU);
    compute_reference(input, weight, output, scale);
}

void sgemv_kernel_cpu(const Tensor& input, const Tensor& weight, Tensor& output,
                     float scale, cudaStream_t) {
    detail::check_sgemv_tensors(input, weight, output, DeviceType::CPU);

#if defined(MINI_INFER_HAS_ARMADILLO)
    detail::compute_sgemv_armadillo(input, weight, output, scale);
#else
    compute_reference(input, weight, output, scale);
#endif
}

}  // namespace mini_infer
