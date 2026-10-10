#include "mini_infer/ops/mlp/mini_infer_swiglu.h"
#include <cmath>
#include <stdexcept>
#if defined(MINI_INFER_HAS_ARMADILLO)
#include <armadillo>
#endif

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

SwiGLUKernel get_swiglu_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return swiglu_kernel_cpu;
        case DeviceType::CUDA: return swiglu_kernel_cuda;
        default: throw std::invalid_argument("Unknown SwiGLU backend");
    }
}

SwiGLULayer::SwiGLULayer(DeviceType device) : device_(device) {
    (void)get_swiglu_kernel(device);  // 构造时拒绝 UNKNOWN，而非计算后才发现。
}

void SwiGLULayer::check(const Tensor& gate, const Tensor& up,
                        const Tensor& output) const {
    detail::check_swiglu_tensors(gate, up, output, device_);
}

void SwiGLULayer::forward(const Tensor& gate, const Tensor& up, Tensor& output,
                          cudaStream_t stream) const {
    check(gate, up, output);
    get_swiglu_kernel(device_)(gate, up, output, stream);
}

#if defined(MINI_INFER_HAS_ARMADILLO)
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
#endif

}  // namespace mini_infer
