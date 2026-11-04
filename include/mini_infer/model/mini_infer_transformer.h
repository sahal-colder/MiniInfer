#pragma once

#include "infer/storage/async_kv_pipeline.h"
#include "mini_infer/ops/mini_infer_add.h"
#include "mini_infer/ops/mini_infer_attention.h"
#include "mini_infer/model/mini_infer_model_weights.h"

namespace mini_infer {

// 所有 Transformer 层依次复用一份工作区；层与层之间只保留 hidden 和 KV。
// 工作区不与权重共享；不要把它复制后当成另一个请求的独立内存。
struct TransformerWorkspace {
    Tensor normalized;   // [dim]：attention norm / 最终 norm 的输出
    Tensor query;        // [dim]
    Tensor scores;       // [head_num,seq_len]：只使用当前有效历史
    Tensor attended;     // [dim]：多头注意力拼接后的向量
    Tensor projected;    // [dim]：Wo 投影结果
    Tensor residual;     // [dim]：attention 残差结果
    MLPWorkspace mlp;
    Tensor key, value; // Batched prefill projections, before paged KV scatter.

    TransformerWorkspace(const ModelConfig& config,
                         std::shared_ptr<DeviceAllocator> allocator, int32_t rows = 1);
};

// 一个完整的 pre-norm decoder block：
// x -> RMSNorm -> QKV -> RoPE -> KVCache -> Attention -> Wo -> Add
//   -> RMSNorm -> W1/W3 -> SwiGLU -> W2 -> Add -> x。
// 后半段已由第 17 课的 MLP 封装，包括它自己的 norm 和残差。
class TransformerBlock {
public:
    TransformerBlock(const TransformerWeights& weights, const ModelConfig& config);
    void forward(Tensor& hidden, KVCache& cache, int32_t layer_idx,
                 int32_t position, TransformerWorkspace& work,
                 cudaStream_t stream = nullptr);
    // Separate input/output allows the first block to borrow an embedding row.
    // Input remains read-only unless the caller explicitly uses in-place output.
    void forward(const Tensor& input, Tensor& output, KVCache& cache, int32_t layer_idx,
                 int32_t position, TransformerWorkspace& work,
                 cudaStream_t stream = nullptr);
    void forward_paged(const Tensor& input,Tensor& output,Tensor& key,Tensor& value,
        const float* const* pages,std::size_t tokens_per_page,int32_t position,
        TransformerWorkspace&,cudaStream_t,const int32_t* device_position = nullptr);
    void forward_paged(const Tensor& input, Tensor& output, infer::AsyncKVPipeline::LayerView kv,
                       int32_t position, TransformerWorkspace& work, cudaStream_t stream);
    void prefill(const Tensor& input, Tensor& output, int32_t position,
                 TransformerWorkspace& work, cudaStream_t stream,
                 const float* const* pages, std::size_t tokens_per_page,
                 KVCache* cache = nullptr, int32_t layer = 0);
private:
    int32_t heads_, kv_heads_;
    int32_t head_dim_;               // 每个 Attention 头对应的特征维度（dim / head_num）
    RMSNormLayer norm_;              // Attention 前的归一化算子
    AttentionQKV qkv_;               // Wq, Wk, Wv 线性变换与 RoPE
    MultiHeadAttention attention_;   // QK^T 点积、Softmax、加权 V 算子
    MatmulLayer output_;             // 线性输出投影算子 Wo
    ElementwiseAdd add_;             // 向量按元素相加算子（残差连接）
    MLP mlp_;                        // 包含第 2 个 RMSNorm + SwiGLU + 残差相加的复合子模块
};
}  // namespace mini_infer
