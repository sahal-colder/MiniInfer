#include "mini_infer/model/mini_infer_model.h"
#include "mini_infer/model/mini_infer_paged_cache.h"
#include "mini_infer/ops/mini_infer_prefill.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

namespace mini_infer {
namespace {
void synchronize(DeviceType device) {
    if (device != DeviceType::CUDA) return;
    const auto error = cudaStreamSynchronize(nullptr);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("model CUDA execution: ") + cudaGetErrorString(error));
}

void retire_paged(PagedModelCache& cache, bool discard_prefix) {
    while (!cache.reset(discard_prefix)) std::this_thread::sleep_for(std::chrono::microseconds(20));
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
      logits_device_(DataType::FLOAT32, {config().vocab_size}, allocator) {
    allocator_ = allocator;
    const bool paged=options.kv_mode==ModelKVMode::PAGED ||
        (options.kv_mode==ModelKVMode::AUTO && allocator->device_type()==DeviceType::CUDA);
    if(paged) paged_=std::make_unique<PagedModelCache>(config(),std::move(options.paged),allocator);
    else cache_=std::make_unique<KVCache>(config(),allocator);
    std::shared_ptr<DeviceAllocator> logits_allocator;
    if (paged_ && paged_->pipeline())
        logits_allocator = std::make_shared<DeviceAllocator>(std::make_shared<infer::PinnedHostAllocator>());
    else
        logits_allocator = std::make_shared<CPUDeviceAllocator>();
    logits_host_ = Tensor(DataType::FLOAT32, {config().vocab_size}, std::move(logits_allocator));
    for (const auto& weights : weights_.layers)
        blocks_.push_back(std::make_unique<TransformerBlock>(weights, config()));
    if (options.enable_cuda_graph) {
        if (!paged_ || !paged_->pipeline())
            throw std::invalid_argument("CUDA Graphs require enable_async_kv_pipeline");
        graphs_ = std::make_unique<CUDAGraphRunner>(config(), blocks_, *paged_->pipeline());
        hidden_ = graphs_->hidden();
    }
    final_norm_.set_weight(0, weights_.final_norm);
    final_norm_.set_input(0, hidden_);
    final_norm_.set_output(0, work_.normalized);
    classifier_.set_weight(0, weights_.classifier);
    classifier_.set_input(0, work_.normalized);
    classifier_.set_output(0, logits_device_);
}

LlamaModel::~LlamaModel() {
    graphs_.reset(); // GraphExec and private scratch precede KV slots and streams.
    paged_.reset(); // Retire streams before destroying tensors referenced by their work.
}
CUDAGraphStats LlamaModel::cuda_graph_stats() const {
    return graphs_ ? graphs_->stats() : CUDAGraphStats{};
}
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
        auto* pipeline = paged_ ? paged_->pipeline() : nullptr;
        const auto stream = paged_ ? paged_->stream() : nullptr;
        const auto skipped=paged_ && !position_ ? paged_->begin_request(tokens,count) : 0;
        if(cache_) cache_->reserve(position_ + static_cast<int32_t>(count));
        const auto remaining=static_cast<int32_t>(count-skipped);
        if (remaining>1) {
            const auto first=position_+static_cast<int32_t>(skipped);
            if (prefill_hidden_.empty() || prefill_hidden_.dim(0)!=remaining ||
                prefill_work_->scores.dim(2)<first+remaining) {
                auto prefill_config=config();
                prefill_config.seq_len=first+remaining;
                prefill_work_=std::make_unique<TransformerWorkspace>(prefill_config,allocator_,remaining);
                prefill_hidden_=Tensor(DataType::FLOAT32,{remaining,config().dim},allocator_);
                prefill_ids_=Tensor(DataType::INT32,{remaining},allocator_);
            }
            if (hidden_.device_type()==DeviceType::CUDA) {
                const auto ids=Tensor::borrow(DataType::INT32,{remaining},const_cast<int32_t*>(tokens+skipped),DeviceType::CPU);
                prefill_ids_.copy_from(ids,stream);
                detail::prefill_embedding_cuda(prefill_ids_,weights_.embedding,prefill_hidden_,stream);
            } else {
                for (int32_t row=0;row<remaining;++row)
                    std::copy_n(weights_.embedding.ptr<float>()+std::size_t(tokens[skipped+row])*config().dim,
                                config().dim,prefill_hidden_.ptr<float>()+std::size_t(row)*config().dim);
            }
            if (paged_) paged_->begin_batch(tokens+skipped,remaining,first);
            if (pipeline) {
                const infer::AsyncKVPipeline::Compute compute =
                    [&](std::size_t layer,infer::AsyncKVPipeline::LayerView kv,cudaStream_t execution) {
                        blocks_[layer]->prefill(prefill_hidden_,prefill_hidden_,first,*prefill_work_,execution,kv.pages,kv.tokens_per_page);
                    };
                while (!pipeline->tick(compute)) std::this_thread::yield();
                paged_->record_async_token();
            } else {
                for (int32_t layer=0;layer<config().layer_num;++layer) {
                    if (paged_) {
                        auto binding=paged_->prepare_layer(layer,first+remaining-1,first);
                        blocks_[layer]->prefill(prefill_hidden_,prefill_hidden_,first,*prefill_work_,stream,binding.pages,binding.tokens_per_page);
                        binding.finish();
                        continue;
                    }
                    blocks_[layer]->prefill(prefill_hidden_,prefill_hidden_,first,*prefill_work_,stream,nullptr,0,cache_.get(),layer);
                }
            }
            hidden_.copy_from(prefill_hidden_.view(std::size_t(remaining-1)*config().dim,{config().dim}),stream);
        }
        for (std::size_t i = remaining>1 ? count : skipped; i < count; ++i) {
            if(paged_) paged_->begin_token(tokens[i],position_+static_cast<int32_t>(i));
            // IDs are validated above. Borrow the immutable row directly; the
            // first block writes to hidden_, never back into embedding weights.
            const auto input = weights_.embedding.view(
                static_cast<std::size_t>(tokens[i]) * config().dim, {config().dim});
            const auto pos = position_ + static_cast<int32_t>(i);
            if (pipeline) {
                if (graphs_) graphs_->prepare(input, pos);
                const infer::AsyncKVPipeline::Compute compute =
                    [&](std::size_t layer, infer::AsyncKVPipeline::LayerView kv, cudaStream_t execution) {
                        if (graphs_) {
                            graphs_->replay(layer, kv, execution);
                            return;
                        }
                        blocks_[layer]->forward_paged(layer == 0 ? input : hidden_, hidden_,
                                                      kv, pos, work_, execution);
                    };
                while (!pipeline->tick(compute))
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                paged_->record_async_token();
                continue;
            }
            for (int32_t layer = 0; layer < config().layer_num; ++layer) {
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
        if (pipeline) {
            pipeline->download(logits_host_.ptr<float>(), logits_device_.ptr<float>(), logits_host_.byte_size());
            while (!pipeline->poll_download())
                std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            logits_host_.copy_from(logits_device_, stream);
            if (paged_) check_cuda(cudaStreamSynchronize(stream), "Paged logits completion");
        }
        if (paged_) paged_->finish_step();
        position_ += static_cast<int32_t>(count);
        return logits_host_;
    } catch (...) {
        failed_ = true;
        // 失败时也先等待已提交工作，防止 reset/析构复用仍被 GPU 使用的内存。
        if (hidden_.device_type() == DeviceType::CUDA && !(paged_ && paged_->pipeline()))
            (void)cudaStreamSynchronize(paged_ ? paged_->stream() : nullptr);
        // Retire every accepted prefetch window, preserving the original error.
        if (paged_) { try { retire_paged(*paged_, true); } catch (...) {} }
        throw;
    }
}

void LlamaModel::reset() {
    if (paged_) retire_paged(*paged_, failed_);
    else { synchronize(hidden_.device_type()); cache_->reset(); }
    position_ = 0;
    failed_ = false;
}
}  // namespace mini_infer
