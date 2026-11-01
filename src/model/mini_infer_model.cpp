#include "mini_infer/model/mini_infer_model.h"

#include <algorithm>
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
    return run(&token, 1);
}

const Tensor& LlamaModel::prefill(const std::vector<int32_t>& tokens) {
    return run(tokens.data(), tokens.size());
}

const Tensor& LlamaModel::run(const int32_t* tokens, std::size_t count) {
    if (failed_) throw std::logic_error("previous forward failed; call reset() first");
    if (!count) throw std::invalid_argument("prefill requires at least one token");
    if (count > static_cast<std::size_t>(config().seq_len - position_))
        throw std::out_of_range("model context is full; call reset() for a new request");
    if (std::any_of(tokens, tokens + count,
                    [&](int32_t token) { return token < 0 || token >= config().vocab_size; }))
        throw std::out_of_range("token id outside model vocabulary");
    try {
        cache_.reserve(position_ + static_cast<int32_t>(count));
        for (std::size_t i = 0; i < count; ++i) {
            // 同步输入复制保留 token_host_ 的复用契约；中间 token 不返回 logits。
            token_host_.ptr<int32_t>()[0] = tokens[i];
            token_device_.copy_from(token_host_);
            Tensor embedding_output = hidden_.view(0, {1, config().dim});
            embedding_.forward(token_device_, embedding_output);

            for (int32_t layer = 0; layer < config().layer_num; ++layer)
                blocks_[layer]->forward(hidden_, cache_, layer,
                                         position_ + static_cast<int32_t>(i), work_);
        }

        // 仅提示末尾或当前 decode token 需要词表投影和 CPU logits。
        final_norm_.forward();
        classifier_.forward();
        // CUDA 算子按默认 stream 顺序执行，返回 API 前明确完成 GPU 工作。
        synchronize(hidden_.device_type());
        logits_host_.copy_from(logits_device_);
        position_ += static_cast<int32_t>(count);
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
