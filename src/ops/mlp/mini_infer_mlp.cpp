#include "mini_infer/ops/mlp/mini_infer_mlp.h"

#include <stdexcept>

namespace mini_infer {
namespace {
void check_tensor(const Tensor& tensor, DeviceType device) {
    if (tensor.empty() || tensor.size() == 0 || tensor.dims().empty() ||
        tensor.data_type() != DataType::FLOAT32 || tensor.device_type() != device) {
        throw std::invalid_argument("MLP needs nonempty FP32 tensors on the same device");
    }
}

MLPWeights checked_weights(const MLPWeights& weights) {
    const DeviceType device = weights.norm.device_type();
    if (device != DeviceType::CPU && device != DeviceType::CUDA) {
        throw std::invalid_argument("MLP requires CPU or CUDA weights");
    }
    for (const Tensor* w : {&weights.norm, &weights.w1, &weights.w2, &weights.w3}) {
        check_tensor(*w, device);
    }
    if (weights.norm.dims().size() != 1 || weights.w1.dims().size() != 2) {
        throw std::invalid_argument("MLP expects norm=[dim] and W1=[hidden,dim]");
    }
    const int32_t dim = weights.norm.dim(0);
    const int32_t hidden = weights.w1.dim(0);
    if (weights.w1.dim(1) != dim || weights.w3.dims() != weights.w1.dims() ||
        weights.w2.dims() != std::vector<int32_t>{dim, hidden}) {
        throw std::invalid_argument("MLP expects W1/W3=[hidden,dim], W2=[dim,hidden]");
    }
    return weights;  // 仅复制句柄；输入权重在整个运行期间保持只读。
}
}  // namespace

MLPWorkspace::MLPWorkspace(const std::vector<int32_t>& input_shape,
                           int32_t hidden_dim,
                           std::shared_ptr<DeviceAllocator> allocator) {
    if (!allocator || (allocator->device_type() != DeviceType::CPU &&
                       allocator->device_type() != DeviceType::CUDA) ||
        input_shape.empty() || hidden_dim <= 0) {
        throw std::invalid_argument("MLP workspace needs a shape, hidden size and allocator");
    }
    for (int32_t dim : input_shape) {
        if (dim <= 0) throw std::invalid_argument("MLP workspace dimensions must be positive");
    }
    auto hidden_shape = input_shape;
    hidden_shape.back() = hidden_dim;  // 保留前面的 batch/seq 维，只替换特征维。
    norm_output = Tensor(DataType::FLOAT32, input_shape, allocator);
    w1_output = Tensor(DataType::FLOAT32, hidden_shape, allocator);
    w3_output = Tensor(DataType::FLOAT32, hidden_shape, allocator);
    w2_output = Tensor(DataType::FLOAT32, input_shape, allocator);
    // 四次分配相互独立；中途异常由 Tensor/Buffer 的 RAII 释放已有空间。
}

MLP::MLP(const MLPWeights& weights, float eps)
    : weights_(checked_weights(weights)), dim_(weights_.norm.dim(0)),
      hidden_dim_(weights_.w1.dim(0)),
      norm_layer_(weights_.norm.device_type(), eps),
      w1_layer_(weights_.norm.device_type(), hidden_dim_, dim_),
      w3_layer_(weights_.norm.device_type(), hidden_dim_, dim_),
      swiglu_layer_(weights_.norm.device_type()),
      w2_layer_(weights_.norm.device_type(), dim_, hidden_dim_) {
    norm_layer_.set_weight(0, weights_.norm);
    w1_layer_.set_weight(0, weights_.w1);
    w3_layer_.set_weight(0, weights_.w3);
    w2_layer_.set_weight(0, weights_.w2);
}

void MLP::forward(const Tensor& input, Tensor& output, MLPWorkspace& work,
                   cudaStream_t stream) {
    const DeviceType device = weights_.norm.device_type();
    if (device == DeviceType::CPU && stream) {
        throw std::invalid_argument("CPU MLP does not accept a CUDA stream");
    }
    check_tensor(input, device);
    check_tensor(output, device);
    if (input.dims().back() != dim_ || output.dims() != input.dims()) {
        throw std::invalid_argument("MLP expects input/output=[...,dim]");
    }
    if (output.overlaps(input) && output.ptr<float>() != input.ptr<float>()) {
        throw std::invalid_argument("MLP residual output forbids partial input overlap");
    }

    // 跨算子的检查：单个 Matmul 只知道自己的权重，不能发现 work.w1
    // 意外覆盖 W2 或原始 input 的问题。必须先完成整个流水线的检查。
    Tensor* buffers[] = {&work.norm_output, &work.w1_output,
                         &work.w3_output, &work.w2_output};
    auto hidden_shape = input.dims();
    hidden_shape.back() = hidden_dim_;
    for (int i = 0; i < 4; ++i) {
        const Tensor& buffer = *buffers[i];
        check_tensor(buffer, device);
        const auto& expected = (i == 1 || i == 2) ? hidden_shape : input.dims();
        if (buffer.dims() != expected || buffer.overlaps(input) || buffer.overlaps(output)) {
            throw std::invalid_argument("MLP workspace shape or input/output overlap error");
        }
        for (int j = 0; j < i; ++j) {
            if (buffer.overlaps(*buffers[j])) {
                throw std::invalid_argument("MLP workspace buffers must be independent");
            }
        }
    }
    for (const Tensor* w : {&weights_.norm, &weights_.w1, &weights_.w2, &weights_.w3}) {
        if (output.overlaps(*w)) throw std::invalid_argument("MLP output overlaps weights");
        for (const Tensor* buffer : buffers) {
            if (buffer->overlaps(*w)) throw std::invalid_argument("MLP workspace overlaps weights");
        }
    }

    // 绑定只是共享 Buffer，不复制数值。先连好“计算图”，再统一检查。
    norm_layer_.set_input(0, input);
    norm_layer_.set_output(0, work.norm_output);
    w1_layer_.set_input(0, work.norm_output);
    w1_layer_.set_output(0, work.w1_output);
    w3_layer_.set_input(0, work.norm_output);
    w3_layer_.set_output(0, work.w3_output);
    w2_layer_.set_input(0, work.w1_output);
    w2_layer_.set_output(0, work.w2_output);
    norm_layer_.check();
    w1_layer_.check();
    w3_layer_.check();
    swiglu_layer_.check(work.w1_output, work.w3_output, work.w1_output);
    w2_layer_.check();
    const AddKernel residual_add = get_add_kernel(device);

    // ===== 课件核心六步：初学时可以先只读下面这几行 =====
    norm_layer_.forward(stream);  // 1. x -> ffn_norm_output
    w1_layer_.forward(stream);    // 2. gate = W1 * norm，dim -> hidden_dim
    w3_layer_.forward(stream);    // 3. up   = W3 * norm，dim -> hidden_dim
    swiglu_layer_.forward(work.w1_output, work.w3_output,
                          work.w1_output, stream);  // 4. act 原位写回 gate
    w2_layer_.forward(stream);    // 5. down = W2 * act，hidden_dim -> dim

    // 6. residual = 原始 x + down。旧 AddLayer 的策略禁止所有别名，
    // 本课已验证“仅允许完全原位”，故调用已有逐元素 Add 内核。
    // 每线程只读写同一索引，不会破坏其他索引，支持 input==output。
    residual_add(input, work.w2_output, output, stream);
}
}  // namespace mini_infer
