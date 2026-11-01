#include "mini_infer/ops/mini_infer_swiglu.h"
#include <armadillo>

namespace mini_infer {
namespace detail {

// 只在 CMake 启用 Armadillo 时编译；参数检查在外层完成。
void compute_swiglu_armadillo(const Tensor& gate, const Tensor& up, Tensor& output) {
    const auto size = static_cast<arma::uword>(gate.size());
    // false：借用 Tensor 内存；true：strict，禁止改变视图长度。
    // 构造器需要 float*，但 const 视图只读，不通过 const_cast 修改输入。
    const arma::fvec gate_view(const_cast<float*>(gate.ptr<float>()), size, false, true);
    const arma::fvec up_view(const_cast<float*>(up.ptr<float>()), size, false, true);
    arma::fvec output_view(output.ptr<float>(), size, false, true);

    // 对照课件：gate %= sigmoid(gate); output = gate % up;
    // % 是逐元素乘法；* 在 Armadillo 中表示矩阵乘法。
    // 这里先产生独立临时量，保证非原位时 gate 不变，且 gate==up==output
    // 时 up 不会被提前变成 SiLU(gate)。逐元素变换使用稳定 sigmoid 公式。
    arma::fvec sigmoid = arma::exp(-arma::abs(gate_view));
    for (arma::uword i = 0; i < size; ++i) {
        const float e = sigmoid[i];
        sigmoid[i] = gate_view[i] >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    }
    const arma::fvec result = (gate_view % sigmoid) % up_view;
    output_view = result;
}
}  // namespace detail
}  // namespace mini_infer
