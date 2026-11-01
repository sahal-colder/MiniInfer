// 第 13 课第 3 步：仅解释 Armadillo 对已有 Tensor 内存的借用和列主序。
#include "mini_infer/ops/mini_infer_sgemv.h"
#include <armadillo>
namespace mini_infer {
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
}  // namespace mini_infer
