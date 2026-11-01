#include "mini_infer/ops/mini_infer_add.h"

namespace mini_infer {
namespace detail {
void check_elementwise_add_tensors(const Tensor& a, const Tensor& b,
                                   const Tensor& output, DeviceType device) {
    for (const Tensor* t : {&a, &b, &output}) {
        if (t->empty() || t->size() == 0 || t->dims().empty() ||
            t->data_type() != DataType::FLOAT32 || t->device_type() != device) {
            throw std::invalid_argument("Add requires nonempty FP32 tensors on its backend");
        }
    }
    if (a.dims() != b.dims() || a.dims() != output.dims()) {
        throw std::invalid_argument("Add requires equal shapes; no broadcasting");
    }
    if (output.overlaps(a) || output.overlaps(b)) {
        throw std::invalid_argument("Add output must not overlap inputs");
    }
}
}  // namespace detail

void elementwise_add_cpu(const Tensor& a, const Tensor& b, Tensor& output,
                         cudaStream_t stream) {
    detail::check_elementwise_add_tensors(a, b, output, DeviceType::CPU);
    if (stream) throw std::invalid_argument("CPU Add does not accept a CUDA stream");
    const float* lhs = a.ptr<float>();
    const float* rhs = b.ptr<float>();
    float* result = output.ptr<float>();
    // Tensor 连续存储，多维 shape 可展平成 size() 个逐元素加法。
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = lhs[i] + rhs[i];
}

}  // namespace mini_infer
