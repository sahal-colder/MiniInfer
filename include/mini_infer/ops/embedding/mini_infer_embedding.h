#pragma once

// 第 18 课：Embedding 是查表/搬运，不是矩阵乘法。
#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {

// token_ids: INT32 [token_num]，weight: FP32 [vocab_size,dim]，
// output: FP32 [token_num,dim]。三者须在同一设备，output 不得重叠输入。
//   output[t,i] = weight[token_ids[t],i]
// 全部 id 必须满足 0 <= id < vocab_size，重复 id 合法。
// CUDA 路径先在给定 stream 上读回 id 并同步，验证后才提交查表内核；
// 所以非法 id 会抛出异常，output 保持原值，不留下未初始化的行。
// 这是为了教学接口的明确错误语义，会有小量读回/同步开销。
// 查表内核本身异步；id/weight/output 必须存活至 stream 完成。
using EmbeddingKernel = void (*)(const Tensor&, const Tensor&, Tensor&, cudaStream_t);
EmbeddingKernel get_embedding_kernel(DeviceType device);
void embedding_kernel_cpu(const Tensor& token_ids, const Tensor& weight,
                          Tensor& output, cudaStream_t stream = nullptr);
void embedding_kernel_cuda(const Tensor& token_ids, const Tensor& weight,
                           Tensor& output, cudaStream_t stream = nullptr);

class EmbeddingLayer {
public:
    // Tensor 句柄共享并保活权重 Buffer；不复制整张词表。
    // Tensor::borrow 的外部内存仍需调用者负责保活。
    explicit EmbeddingLayer(const Tensor& weight);
    void forward(const Tensor& token_ids, Tensor& output,
                 cudaStream_t stream = nullptr) const;
private:
    Tensor weight_;
};

namespace detail {
void check_embedding_tensors(const Tensor&, const Tensor&, const Tensor&, DeviceType);
void check_embedding_ids(const Tensor& token_ids, int32_t vocab_size, cudaStream_t stream);
}  // namespace detail
}  // namespace mini_infer
