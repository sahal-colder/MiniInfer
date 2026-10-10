#include "mini_infer/ops/add/mini_infer_add.h"
#include "mini_infer/backend/mini_infer_kernels.h"

namespace mini_infer {
namespace detail {
void check_elementwise_add_tensors(const Tensor& a, const Tensor& b,
                                   const Tensor& output, DeviceType device, bool allow_inplace) {
    for (const Tensor* t : {&a, &b, &output}) {
        if (t->empty() || t->size() == 0 || t->dims().empty() ||
            t->data_type() != DataType::FLOAT32 || t->device_type() != device) {
            throw std::invalid_argument("Add requires nonempty FP32 tensors on its backend");
        }
    }
    if (a.dims() != b.dims() || a.dims() != output.dims()) {
        throw std::invalid_argument("Add requires equal shapes; no broadcasting");
    }
    for (const Tensor* input : {&a, &b}) {
        if (output.overlaps(*input) && (!allow_inplace || output.ptr<float>() != input->ptr<float>()))
            throw std::invalid_argument("Add output has unsupported input overlap");
    }
}
}  // namespace detail

namespace {
void add_cpu(const Tensor& a, const Tensor& b, Tensor& output,
             cudaStream_t stream, bool allow_inplace) {
    detail::check_elementwise_add_tensors(a, b, output, DeviceType::CPU, allow_inplace);
    if (stream) throw std::invalid_argument("CPU Add does not accept a CUDA stream");
    const float* lhs = a.ptr<float>();
    const float* rhs = b.ptr<float>();
    float* result = output.ptr<float>();
    // Tensor 连续存储，多维 shape 可展平成 size() 个逐元素加法。
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = lhs[i] + rhs[i];
}
}  // namespace

void elementwise_add_cpu(const Tensor& a, const Tensor& b, Tensor& output, cudaStream_t stream) {
    add_cpu(a, b, output, stream, false);
}
void add_kernel_cpu(const Tensor& a, const Tensor& b, Tensor& output, cudaStream_t stream) {
    add_cpu(a, b, output, stream, true);
}

ElementwiseAddKernel get_elementwise_add_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return elementwise_add_cpu;
        case DeviceType::CUDA: return elementwise_add_cuda;
        default: throw std::invalid_argument("Unknown Add backend");
    }
}

ElementwiseAdd::ElementwiseAdd(DeviceType device) : device_(device) {
    (void)get_elementwise_add_kernel(device);
}
void ElementwiseAdd::forward(const Tensor& a, const Tensor& b, Tensor& output,
                             cudaStream_t stream) const {
    get_elementwise_add_kernel(device_)(a, b, output, stream);
}
}  // namespace mini_infer
