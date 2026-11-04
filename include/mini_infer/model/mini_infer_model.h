#pragma once

#include "mini_infer/model/mini_infer_transformer.h"
#include "mini_infer/model/mini_infer_model_options.h"

namespace mini_infer {
class PagedModelCache;

// 学习入口：main -> LlamaModel::forward -> TransformerBlock::forward。
// 一个对象对应一条序列。prefill 逐个消费 prompt token，decode 每次消费
// 上一步采样出的 token；prefill 仅在末尾生成 logits，KV 持续累积。
// CUDA 默认使用三级 Paged KV 和跨请求前缀共享，CPU 使用连续 KV。
// 权重和激活在选择的设备上；最终 logits 读回 CPU。
// 当前是同步、单序列 FP32 推理，不支持同一对象并发调用。
class LlamaModel {
public:
    LlamaModel(const std::string& model_path, std::shared_ptr<DeviceAllocator> allocator);
    LlamaModel(const std::string&,std::shared_ptr<DeviceAllocator>,ModelOptions);
    ~LlamaModel();
    bool uses_paged_kv() const noexcept { return bool(paged_); }
    PagedModelStats paged_stats() const;
    const infer::io::LoadStats& weight_io_stats() const noexcept { return weights_.loading; }
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    const ModelConfig& config() const noexcept { return weights_.config; }
    int32_t position() const noexcept { return position_; }

    // 输入当前 token，返回预测“下一个 token”的 CPU FP32 [vocab_size] logits。
    // 返回值借用模型的工作区，下次 forward 会覆盖；保留时请 clone(cpu_allocator)。
    // 位置自动从 0 增长；越界/非法 token 在计算前拒绝，不改变已有缓存。
    // 计算中途异常后必须 reset 再使用，避免消费只写了一部分层的 KV。
    const Tensor& forward(int32_t token);
    // 追加非空提示，返回最后一个 token 的 logits；整批先验证，不重置已有 KV。
    // 中间 token 只推进 decoder，不计算词表投影或把 logits 读回 CPU。
    const Tensor& prefill(const std::vector<int32_t>& tokens);
    void reset();  // 新请求：释放私有尾页，保留有效完整前缀，不重新加载权重。

private:
    const Tensor& run(const int32_t* tokens, std::size_t count);
    ModelWeights weights_;
    std::unique_ptr<KVCache> cache_;
    std::unique_ptr<PagedModelCache> paged_;
    TransformerWorkspace work_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    RMSNormLayer final_norm_;
    MatmulLayer classifier_;
    Tensor hidden_, logits_device_, logits_host_;
    int32_t position_ = 0;
    bool failed_ = false;
};
}  // namespace mini_infer
