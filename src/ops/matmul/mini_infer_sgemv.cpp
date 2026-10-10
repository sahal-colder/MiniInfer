#include "mini_infer/ops/matmul/mini_infer_sgemv.h"
#include <stdexcept>
#if defined(MINI_INFER_HAS_ARMADILLO)
#include <armadillo>
#endif

namespace mini_infer {

namespace detail {
void check_sgemv_tensors(const Tensor& input, const Tensor& weight,
                        const Tensor& output, DeviceType device) {
    // 元数据检查必须先于指针访问。直接调用后端函数也得到同样保护，
    // 不要求调用者必须先经过 MatmulLayer::check。
    for (const Tensor* tensor : {&input, &weight, &output}) {
        if (tensor->empty() || tensor->dims().empty() || tensor->size() == 0) {
            throw std::invalid_argument("Sgemv requires allocated, nonempty tensors");
        }
        if (tensor->data_type() != DataType::FLOAT32 ||
            tensor->device_type() != device) {
            throw std::invalid_argument("Sgemv requires FP32 tensors on its backend");
        }
    }
    if (weight.dims().size() != 2 || input.dims().back() != weight.dim(1)) {
        throw std::invalid_argument("Sgemv expects weight=[out,in], input=[...,in]");
    }
    std::vector<int32_t> expected_output = input.dims();
    expected_output.back() = weight.dim(0);
    if (output.dims() != expected_output) {
        throw std::invalid_argument("Sgemv expects output=[...,out]");
    }
    // CPU BLAS 与 CUDA 并行计算都可能先写一部分输出，再读另一部分输入。
    // 拒绝内存区间重叠，不能只比较两个 Tensor 对象是否相同。
    if (output.overlaps(input) || output.overlaps(weight)) {
        throw std::invalid_argument("Sgemv output must not overlap input or weight");
    }
}

}  // namespace detail

SgemvKernel get_sgemv_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return sgemv_kernel_cpu;
        case DeviceType::CUDA: return sgemv_kernel_cuda;
        default: throw std::invalid_argument("Unknown Sgemv backend");
    }
}

namespace detail {
#if defined(MINI_INFER_HAS_ARMADILLO)
// 可选 Armadillo 实现在本文件下方，与标量后端共享参数检查。
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

#if defined(MINI_INFER_HAS_ARMADILLO)
namespace detail {
// 调用者已完成 shape/device/重叠检查。局部视图不拥有 Tensor 的 Buffer。
void compute_sgemv_armadillo(const Tensor& input, const Tensor& weight,
                             Tensor& output, float scale) {
    const arma::uword in = static_cast<arma::uword>(weight.dim(1));
    const arma::uword out = static_cast<arma::uword>(weight.dim(0));
    const arma::uword rows = static_cast<arma::uword>(input.size() / in);

    // Armadillo 按列主序访问，不会自动替我们移动/转置原始字节。
    // 因此把行主序 W[out,in] 视作列主序 weight_view[in,out]，
    // 每一列正好对应原 W 的一行；input_view[in,rows] 同理。
    // output_view[out,rows] 的每一列则是一个输入向量的输出。
    //
    // false = 不复制外部内存；true = strict，视图不可改变元素数量。
    // aux_mem 构造函数要求 float*，所以输入处使用 const_cast；把视图
    // 声明为 const，并且只放在表达式右侧，确保不修改 input/weight。
    // 这些局部视图不拥有 Buffer，函数返回后 Tensor 仍负责释放内存。
    const arma::fmat input_view(const_cast<float*>(input.ptr<float>()),
                               in, rows, false, true);
    const arma::fmat weight_view(const_cast<float*>(weight.ptr<float>()),
                                in, out, false, true);
    arma::fmat output_view(output.ptr<float>(), out, rows, false, true);

    // weight_view = W^T，input_view = X^T，因此结果视图为 W * X^T。
    // 对应行主序 Tensor 中 Y = X * W^T。用 .t() 表达转置，而非手动
    // 复制一份转置权重；具体表达式求值仍可能使用库内部临时工作区。
    // 单向量时这也等价于课件的行向量 x * weight_view。
    output_view = (weight_view.t() * input_view) * scale;
}
}  // namespace detail
#endif

}  // namespace mini_infer
