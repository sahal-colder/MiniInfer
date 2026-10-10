#include "mini_infer/model/mini_infer_transformer.h"
#include "mini_infer/ops/rope/mini_infer_rope.h"
#include <stdexcept>

namespace mini_infer {
namespace {
int32_t checked_head_dim(const ModelConfig& c) {
    if (c.dim <= 0 || c.hidden_dim <= 0 || c.head_num <= 0 || c.kv_head_num <= 0 ||
        c.dim % c.head_num || c.head_num % c.kv_head_num || (c.dim / c.head_num) % 2)
        throw std::invalid_argument("Transformer needs valid dimensions and even RoPE heads");
    return c.dim / c.head_num;
}
}
TransformerWorkspace::TransformerWorkspace(
    const ModelConfig& c, std::shared_ptr<DeviceAllocator> allocator)
    : normalized(DataType::FLOAT32, {c.dim}, allocator),
      query(DataType::FLOAT32, {c.dim}, allocator),
      scores(DataType::FLOAT32, {c.head_num, c.seq_len}, allocator),
      attended(DataType::FLOAT32, {c.dim}, allocator),
      projected(DataType::FLOAT32, {c.dim}, allocator),
      residual(DataType::FLOAT32, {c.dim}, allocator),
      mlp({c.dim}, c.hidden_dim, allocator) {}

TransformerBlock::TransformerBlock(const TransformerWeights& weights, const ModelConfig& c)
    : head_dim_(checked_head_dim(c)),
      norm_(weights.attention_norm.device_type()), qkv_(weights.qkv),
      attention_(c.head_num, c.kv_head_num, head_dim_),
      output_(weights.output.device_type(), c.dim, c.dim),
      add_(weights.output.device_type()), mlp_(weights.mlp) {
    if (weights.attention_norm.dims() != std::vector<int32_t>{c.dim} ||
        weights.qkv.query.dims() != std::vector<int32_t>{c.dim, c.dim} ||
        weights.qkv.key.dim(0) != c.kv_head_num * head_dim_ ||
        weights.output.dims() != std::vector<int32_t>{c.dim, c.dim} ||
        weights.mlp.norm.dims() != std::vector<int32_t>{c.dim} ||
        weights.mlp.w1.dims() != std::vector<int32_t>{c.hidden_dim, c.dim})
        throw std::invalid_argument("Transformer weights do not match model config");
    norm_.set_weight(0, weights.attention_norm);
    output_.set_weight(0, weights.output);
}

void TransformerBlock::forward(Tensor& hidden, KVCache& cache, int32_t layer_idx,
                                int32_t position, TransformerWorkspace& work,
                                cudaStream_t stream) {
    // 1. 先归一化。hidden 仍保留原值，稍后的残差连接还需要它。
    norm_.set_input(0, hidden);
    norm_.set_output(0, work.normalized);
    norm_.forward(stream);

    // 2. K/V 直接写入当前层/位置的缓存。只旋转 Q 和这个新 K，然后提交。
    auto kv = qkv_.forward(work.normalized, work.query, cache, layer_idx, position, stream);
    rope_inplace(work.query, kv.first, head_dim_, position, 10000.0f, stream);
    cache.commit(layer_idx, position);

    // 3. Attention 内部完成 QK^T / sqrt(head_dim) -> Softmax -> AV。
    // cache 中只有 [0,position] 的有效前缀，因果掩码由可见长度表达。
    attention_.forward(work.query, cache, layer_idx, position,
                       work.scores, work.attended, stream);
    output_.set_input(0, work.attended);
    output_.set_output(0, work.projected);
    output_.forward(stream);
    add_.forward(hidden, work.projected, work.residual, stream);

    // 4. MLP 包含第二次 RMSNorm 和第二条残差，最终写回 hidden。
    mlp_.forward(work.residual, hidden, work.mlp, stream);
}
}  // namespace mini_infer
