#include "mini_infer/ops/mini_infer_swiglu.h"

#include <cmath>
#include <stdexcept>

namespace mini_infer {
namespace detail {
#if defined(MINI_INFER_HAS_ARMADILLO)
void compute_swiglu_armadillo(const Tensor&, const Tensor&, Tensor&);
#endif

void check_swiglu_tensors(const Tensor& gate, const Tensor& up,
                          const Tensor& output, DeviceType device) {
    for (const Tensor* tensor : {&gate, &up, &output}) {
        if (tensor->empty() || tensor->size() == 0 || tensor->dims().empty() ||
            tensor->data_type() != DataType::FLOAT32 ||
            tensor->device_type() != device) {
            throw std::invalid_argument("SwiGLU needs nonempty FP32 tensors on its backend");
        }
    }
    if (gate.dims() != up.dims() || gate.dims() != output.dims()) {
        throw std::invalid_argument("SwiGLU requires equal shapes; no broadcasting");
    }
    for (const Tensor* input : {&gate, &up}) {
        // 同形状 + 同首地址意味着完全重合；其他相交区间均禁止。
        if (output.overlaps(*input) && output.ptr<float>() != input->ptr<float>()) {
            throw std::invalid_argument("SwiGLU forbids partially overlapping output");
        }
    }
}
}  // namespace detail

namespace {
void compute_reference(const Tensor& gate, const Tensor& up, Tensor& output) {
    const float* gate_data = gate.ptr<float>();
    const float* up_data = up.ptr<float>();
    float* output_data = output.ptr<float>();
    for (std::size_t i = 0; i < gate.size(); ++i) {
        // 先读两个数，再写 output[i]，因此 output==gate/up 都是安全的。
        const float g = gate_data[i];
        const float u = up_data[i];
        // sigmoid(g)=1/(1+exp(-g))。g<0 时改写为 exp(g)/(1+exp(g))，
        // 避免有限的大负数（如 -100）使 exp(-g) 上溢。
        const float e = std::exp(-std::fabs(g));
        const float sigmoid = g >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
        output_data[i] = (g * sigmoid) * u;
    }
}

void check_cpu_stream(cudaStream_t stream) {
    if (stream) throw std::invalid_argument("CPU SwiGLU does not accept a CUDA stream");
}
}  // namespace

void swiglu_kernel_cpu_reference(const Tensor& gate, const Tensor& up,
                                 Tensor& output, cudaStream_t stream) {
    detail::check_swiglu_tensors(gate, up, output, DeviceType::CPU);
    check_cpu_stream(stream);
    compute_reference(gate, up, output);
}

void swiglu_kernel_cpu(const Tensor& gate, const Tensor& up,
                       Tensor& output, cudaStream_t stream) {
    detail::check_swiglu_tensors(gate, up, output, DeviceType::CPU);
    check_cpu_stream(stream);
#if defined(MINI_INFER_HAS_ARMADILLO)
    detail::compute_swiglu_armadillo(gate, up, output);
#else
    compute_reference(gate, up, output);
#endif
}

}  // namespace mini_infer
