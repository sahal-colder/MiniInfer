#pragma once

#include "mini_infer/ops/embedding/mini_infer_embedding.h"
#include "mini_infer/model/mini_infer_transformer.h"

namespace mini_infer {

// 学习入口：main -> LlamaModel::forward -> TransformerBlock::forward。
// 一个对象对应一条序列。prefill 逐个消费 prompt token，decode 每次消费
// 上一步采样出的 token；两者使用同一个 forward，KVCache 持续累积。
// 权重、激活和 KV 全部在选择的 CPU/CUDA 设备上；只把最终 logits 读回 CPU。
// 当前是同步、单序列 FP32 推理，不支持同一对象并发调用。
class LlamaModel {
public:
    LlamaModel(const std::string& model_path, std::shared_ptr<DeviceAllocator> allocator);
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    const ModelConfig& config() const noexcept { return weights_.config; }
    int32_t position() const noexcept { return position_; }

    // 输入当前 token，返回预测“下一个 token”的 CPU FP32 [vocab_size] logits。
    // 返回值借用模型的工作区，下次 forward 会覆盖；保留时请 clone(cpu_allocator)。
    // 位置自动从 0 增长；越界/非法 token 在计算前拒绝，不改变已有缓存。
    // 计算中途异常后必须 reset 再使用，避免消费只写了一部分层的 KV。
    const Tensor& forward(int32_t token);
    void reset();  // 新请求：清有效长度、复用容量，不重新加载权重。

private:
    ModelWeights weights_;
    KVCache cache_;
    TransformerWorkspace work_;
    EmbeddingLayer embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    RMSNormLayer final_norm_;
    MatmulLayer classifier_;
    Tensor token_host_, token_device_, hidden_, logits_device_, logits_host_;
    int32_t position_ = 0;
    bool failed_ = false;
};
}  // namespace mini_infer
