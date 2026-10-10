#include "mini_infer/ops/embedding/mini_infer_embedding.h"

#include <algorithm>

namespace mini_infer {
namespace {
void check_weight(const Tensor& weight) {
    if (weight.empty() || weight.dims().size() != 2 ||
        weight.data_type() != DataType::FLOAT32 ||
        (weight.device_type() != DeviceType::CPU && weight.device_type() != DeviceType::CUDA)) {
        throw std::invalid_argument("Embedding weight must be a nonempty FP32 [vocab,dim] matrix");
    }
}
}  // namespace

namespace detail {
void check_embedding_tensors(const Tensor& ids, const Tensor& weight,
                             const Tensor& output, DeviceType device) {
    check_weight(weight);
    if (ids.empty() || ids.dims().size() != 1 || ids.data_type() != DataType::INT32 ||
        output.empty() || output.data_type() != DataType::FLOAT32 ||
        output.dims() != std::vector<int32_t>{ids.dim(0), weight.dim(1)} ||
        ids.device_type() != device || weight.device_type() != device || output.device_type() != device) {
        throw std::invalid_argument("Embedding expects INT32 ids=[tokens], FP32 output=[tokens,dim] on one device");
    }
    if (output.overlaps(ids) || output.overlaps(weight)) {
        throw std::invalid_argument("Embedding output must not overwrite ids or weight");
    }
}

void check_embedding_ids(const Tensor& ids, int32_t vocab_size, cudaStream_t stream) {
    // 仅复制 token_num 个整数，不读回整张词表。已有 to_cpu(stream)
    // 在非默认 stream 上是异步的，因此这里必须显式同步后再读主机数据。
    const Tensor host = ids.device_type() == DeviceType::CPU ? ids : ids.to_cpu(stream);
    if (ids.device_type() == DeviceType::CUDA) {
        check_cuda(cudaStreamSynchronize(stream), "Embedding token validation readback");
    }
    for (std::size_t i = 0; i < host.size(); ++i) {
        const int32_t token = host.ptr<int32_t>()[i];
        if (token < 0 || token >= vocab_size) {
            throw std::out_of_range("Embedding token id is outside [0,vocab_size)");
        }
    }
}
}  // namespace detail

void embedding_kernel_cpu(const Tensor& ids, const Tensor& weight,
                          Tensor& output, cudaStream_t stream) {
    detail::check_embedding_tensors(ids, weight, output, DeviceType::CPU);
    if (stream) throw std::invalid_argument("CPU Embedding does not accept a CUDA stream");
    detail::check_embedding_ids(ids, weight.dim(0), stream);
    const int32_t dim = weight.dim(1);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const std::size_t token = static_cast<std::size_t>(ids.ptr<int32_t>()[t]);
        std::copy_n(weight.ptr<float>() + token * dim, dim, output.ptr<float>() + t * dim);
    }
}

EmbeddingKernel get_embedding_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return embedding_kernel_cpu;
        case DeviceType::CUDA: return embedding_kernel_cuda;
        default: throw std::invalid_argument("Unknown Embedding backend");
    }
}

EmbeddingLayer::EmbeddingLayer(const Tensor& weight) : weight_(weight) {
    check_weight(weight_);
}
void EmbeddingLayer::forward(const Tensor& ids, Tensor& output, cudaStream_t stream) const {
    get_embedding_kernel(weight_.device_type())(ids, weight_, output, stream);
}
}  // namespace mini_infer
