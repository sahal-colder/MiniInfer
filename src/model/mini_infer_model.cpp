#include "mini_infer/model/mini_infer_model.h"

#include <stdexcept>

namespace mini_infer {
namespace {
void synchronize(DeviceType device) {
    if (device != DeviceType::CUDA) return;
    const auto error = cudaStreamSynchronize(nullptr);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("model CUDA execution: ") + cudaGetErrorString(error));
}
}

LlamaModel::LlamaModel(const std::string& path, std::shared_ptr<DeviceAllocator> allocator)
    : weights_(load_model_weights(path, allocator)),
      cache_(config(), allocator), work_(config(), allocator),
      embedding_(weights_.embedding), final_norm_(allocator->device_type()),
      classifier_(allocator->device_type(), config().vocab_size, config().dim),
      token_host_(DataType::INT32, {1}, std::make_shared<CPUDeviceAllocator>()),
      token_device_(DataType::INT32, {1}, allocator),
      hidden_(DataType::FLOAT32, {config().dim}, allocator),
      logits_device_(DataType::FLOAT32, {config().vocab_size}, allocator),
      logits_host_(DataType::FLOAT32, {config().vocab_size},
                   std::make_shared<CPUDeviceAllocator>()) {
    for (const auto& weights : weights_.layers)
        blocks_.push_back(std::make_unique<TransformerBlock>(weights, config()));
    final_norm_.set_weight(0, weights_.final_norm);
    final_norm_.set_input(0, hidden_);
    final_norm_.set_output(0, work_.normalized);
    classifier_.set_weight(0, weights_.classifier);
    classifier_.set_input(0, work_.normalized);
    classifier_.set_output(0, logits_device_);
}

const Tensor& LlamaModel::forward(int32_t token) {
    if (failed_) throw std::logic_error("previous forward failed; call reset() first");
    if (token < 0 || token >= config().vocab_size)
        throw std::out_of_range("token id outside model vocabulary");
    if (position_ >= config().seq_len)
        throw std::out_of_range("model context is full; call reset() for a new request");
    try {
        // 1. token -> embedding。view 只改变形状，不搬动 hidden 的内存。
        token_host_.ptr<int32_t>()[0] = token;
        token_device_.copy_from(token_host_);
        Tensor embedding_output = hidden_.view(0, {1, config().dim});
        embedding_.forward(token_device_, embedding_output);

        // 2. N 个 decoder block，共享激活工作区，各自拥有独立权重/KV 段。
        for (int32_t layer = 0; layer < config().layer_num; ++layer)
            blocks_[layer]->forward(hidden_, cache_, layer, position_, work_);

        // 3. 最终 norm + vocabulary projection，输出未归一化的 logits。
        final_norm_.forward();
        classifier_.forward();
        // CUDA 算子按默认 stream 顺序执行，返回 API 前明确完成 GPU 工作。
        synchronize(hidden_.device_type());
        logits_host_.copy_from(logits_device_);
        ++position_;
        return logits_host_;
    } catch (...) {
        failed_ = true;
        // 失败时也先等待已提交工作，防止 reset/析构复用仍被 GPU 使用的内存。
        if (hidden_.device_type() == DeviceType::CUDA) (void)cudaStreamSynchronize(nullptr);
        throw;
    }
}

void LlamaModel::reset() {
    synchronize(hidden_.device_type());
    cache_.reset();
    position_ = 0;
    failed_ = false;
}
}  // namespace mini_infer
