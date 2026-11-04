#include "mini_infer/model/mini_infer_model.h"
#include "mini_infer/model/mini_infer_paged_cache.h"

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
    : LlamaModel(path,std::move(allocator),ModelOptions{}) {}
LlamaModel::LlamaModel(const std::string& path,std::shared_ptr<DeviceAllocator> allocator,ModelOptions options)
    : weights_(load_model_weights(path, allocator)),
      work_(config(), allocator),
      final_norm_(allocator->device_type()),
      classifier_(allocator->device_type(), config().vocab_size, config().dim),
      hidden_(DataType::FLOAT32, {config().dim}, allocator),
      logits_device_(DataType::FLOAT32, {config().vocab_size}, allocator),
      logits_host_(DataType::FLOAT32, {config().vocab_size},
                   std::make_shared<CPUDeviceAllocator>()) {
    const bool paged=options.kv_mode==ModelKVMode::PAGED ||
        (options.kv_mode==ModelKVMode::AUTO && allocator->device_type()==DeviceType::CUDA);
    if(paged) paged_=std::make_unique<PagedModelCache>(config(),std::move(options.paged),allocator);
    else cache_=std::make_unique<KVCache>(config(),allocator);
    for (const auto& weights : weights_.layers)
        blocks_.push_back(std::make_unique<TransformerBlock>(weights, config()));
    final_norm_.set_weight(0, weights_.final_norm);
    final_norm_.set_input(0, hidden_);
    final_norm_.set_output(0, work_.normalized);
    classifier_.set_weight(0, weights_.classifier);
    classifier_.set_input(0, work_.normalized);
    classifier_.set_output(0, logits_device_);
}

LlamaModel::~LlamaModel()=default;
PagedModelStats LlamaModel::paged_stats() const {
    if(paged_) return paged_->stats();
    PagedModelStats result;
    if(hidden_.device_type()==DeviceType::CUDA) result.gpu_reserved_bytes=cache_->allocated_bytes();
    else result.host_reserved_bytes=cache_->allocated_bytes();
    return result;
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
        const auto stream=paged_ ? paged_->stream() : nullptr;
        const auto skipped=paged_ && !position_ ? paged_->begin_request(tokens,count) : 0;
        if(cache_) cache_->reserve(position_ + static_cast<int32_t>(count));
        for (std::size_t i = skipped; i < count; ++i) {
            if(paged_) paged_->begin_token(tokens[i],position_+static_cast<int32_t>(i));
            // IDs are validated above. Borrow the immutable row directly; the
            // first block writes to hidden_, never back into embedding weights.
            const auto input = weights_.embedding.view(
                static_cast<std::size_t>(tokens[i]) * config().dim, {config().dim});
            for (int32_t layer = 0; layer < config().layer_num; ++layer) {
                const auto pos=position_+static_cast<int32_t>(i);
                if(paged_) {
                    auto binding=paged_->prepare_layer(layer,pos);
                    blocks_[layer]->forward_paged(layer==0 ? input : hidden_,hidden_,binding.key,binding.value,
                        binding.pages,binding.tokens_per_page,pos,work_,stream);
                    // Current kernels are queued before next-layer I/O is admitted.
                    paged_->prefetch_next(layer+1,layer);
                    binding.finish();
                } else blocks_[layer]->forward(layer==0 ? input : hidden_,hidden_,*cache_,layer,pos,work_);
            }
        }

        // 仅提示末尾或当前 decode token 需要词表投影和 CPU logits。
        final_norm_.forward(stream);
        classifier_.forward(stream);
        // Copy logits on the same stream as the attention and projection work.
        logits_host_.copy_from(logits_device_,stream);
        if(paged_) {
            check_cuda(cudaStreamSynchronize(stream),"Paged logits completion");
            paged_->finish_step();
        }
        position_ += static_cast<int32_t>(count);
        return logits_host_;
    } catch (...) {
        failed_ = true;
        // 失败时也先等待已提交工作，防止 reset/析构复用仍被 GPU 使用的内存。
        if (hidden_.device_type() == DeviceType::CUDA)
            (void)cudaStreamSynchronize(paged_ ? paged_->stream() : nullptr);
        // Retire every accepted prefetch window, preserving the original error.
        if(paged_) {try {paged_->reset(true);}catch(...) {}}
        throw;
    }
}

void LlamaModel::reset() {
    if(paged_) paged_->reset(failed_);
    else { synchronize(hidden_.device_type()); cache_->reset(); }
    position_ = 0;
    failed_ = false;
}
}  // namespace mini_infer
