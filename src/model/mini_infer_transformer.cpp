#include "mini_infer/model/mini_infer_transformer.h"
#include "mini_infer/ops/mini_infer_rope.h"
#include <stdexcept>
#include "infer/storage/paged_attention.h"
#include "mini_infer/ops/mini_infer_softmax.h"
#include "mini_infer/ops/mini_infer_prefill.h"

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
    const ModelConfig& c, std::shared_ptr<DeviceAllocator> allocator, int32_t rows)
    : normalized(DataType::FLOAT32, rows == 1 ? std::vector<int32_t>{c.dim} : std::vector<int32_t>{rows,c.dim}, allocator),
      query(DataType::FLOAT32, normalized.dims(), allocator),
      scores(DataType::FLOAT32, {rows,c.head_num,c.seq_len}, allocator),
      attended(DataType::FLOAT32, normalized.dims(), allocator),
      projected(DataType::FLOAT32, normalized.dims(), allocator),
      residual(DataType::FLOAT32, normalized.dims(), allocator),
      mlp(normalized.dims(), c.hidden_dim, allocator) {
    if (rows == 1) { scores.reshape({c.head_num,c.seq_len}); return; }
    key = Tensor(DataType::FLOAT32, {rows,c.kv_head_num*(c.dim/c.head_num)}, allocator);
    value = Tensor(DataType::FLOAT32, key.dims(), allocator);
}

TransformerBlock::TransformerBlock(const TransformerWeights& weights, const ModelConfig& c)
    : heads_(c.head_num), kv_heads_(c.kv_head_num), head_dim_(checked_head_dim(c)),
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

void TransformerBlock::prefill(const Tensor& input, Tensor& output, int32_t position,
    TransformerWorkspace& work, cudaStream_t stream, const float* const* pages,
    std::size_t tokens_per_page, KVCache* cache, int32_t layer) {
    norm_.set_input(0,input); norm_.set_output(0,work.normalized); norm_.forward(stream);
    qkv_.forward_into(work.normalized,work.query,work.key,work.value,stream);
    if (input.device_type() == DeviceType::CUDA) {
        detail::prefill_rope_cuda(work.query,work.key,work.value,pages,tokens_per_page,position,head_dim_,stream);
        if (cache) cache->append(layer,work.key,work.value,stream);
        const auto history = cache ? cache->history(layer) : std::pair<Tensor,Tensor>{};
        detail::prefill_attention_cuda(work.query,cache ? history.first.ptr<float>() : nullptr,
            cache ? history.second.ptr<float>() : nullptr,pages,tokens_per_page,work.scores,work.attended,
            position,heads_,kv_heads_,stream);
    } else {
        // CPU reference still batches all projections and FFN; only the
        // per-row RoPE/causal attention reference operators need row views.
        for (int32_t row=0;row<input.dim(0);++row) {
            auto q=work.query.view(std::size_t(row)*input.dim(1),{input.dim(1)});
            auto k=work.key.view(std::size_t(row)*work.key.dim(1),{work.key.dim(1)});
            rope_inplace(q,k,head_dim_,position+row);
        }
        cache->append(layer,work.key,work.value);
        for (int32_t row=0;row<input.dim(0);++row) {
            auto q=work.query.view(std::size_t(row)*input.dim(1),{input.dim(1)});
            auto scores=work.scores.view(std::size_t(row)*heads_*work.scores.dim(2),{heads_,work.scores.dim(2)});
            auto attended=work.attended.view(std::size_t(row)*input.dim(1),{input.dim(1)});
            const auto history=cache->history(layer);
            auto keys=history.first.view(0,{position+row+1,work.key.dim(1)});
            auto values=history.second.view(0,keys.dims());
            get_attention_kernel(DeviceType::CPU)(q,keys,values,scores,attended,heads_,kv_heads_,nullptr);
        }
    }
    output_.set_input(0,work.attended); output_.set_output(0,work.projected); output_.forward(stream);
    add_.forward(input,work.projected,work.residual,stream);
    mlp_.forward(work.residual,output,work.mlp,stream);
}

void TransformerBlock::forward(Tensor& hidden, KVCache& cache, int32_t layer_idx,
                                int32_t position, TransformerWorkspace& work,
                                cudaStream_t stream) {
    forward(hidden, hidden, cache, layer_idx, position, work, stream);
}

void TransformerBlock::forward(const Tensor& input, Tensor& output, KVCache& cache,
                                int32_t layer_idx, int32_t position,
                                TransformerWorkspace& work, cudaStream_t stream) {
    // 1. 先归一化。input 仍保留原值，稍后的残差连接还需要它。
    norm_.set_input(0, input);
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
    add_.forward(input, work.projected, work.residual, stream);

    // 4. MLP 包含第二次 RMSNorm 和第二条残差，最终写入独立或原位 output。
    mlp_.forward(work.residual, output, work.mlp, stream);
}
}  // namespace mini_infer
namespace mini_infer {
void TransformerBlock::forward_paged(const Tensor& input,Tensor& output,Tensor& key,Tensor& value,
    const float* const* pages,std::size_t tokens_per_page,int32_t position,
    TransformerWorkspace& work,cudaStream_t stream,const int32_t* device_position) {
    norm_.set_input(0,input);norm_.set_output(0,work.normalized);norm_.forward(stream);
    qkv_.forward_into(work.normalized,work.query,key,value,stream);
    if (device_position) {
        detail::rope_cuda(work.query,key,head_dim_,position,10000.0f,stream,device_position);
        infer::paged_kv_store_f32(key.ptr<float>(),value.ptr<float>(),pages,tokens_per_page,
                                key.dim(0),device_position,stream);
    } else rope_inplace(work.query,key,head_dim_,position,10000.0f,stream);
    const auto heads=work.query.dim(0)/head_dim_,kv_heads=key.dim(0)/head_dim_;
    // Same dot grouping, same CUB softmax, same serial AV accumulation as the
    // contiguous path. Only physical K/V addressing changes.
    infer::paged_attention_qk_f32(work.query.ptr<float>(),pages,work.scores.ptr<float>(),
        position+1,tokens_per_page,heads,kv_heads,head_dim_,work.scores.dim(1),stream,device_position);
    detail::softmax_rows_cuda(work.scores.ptr<float>(),work.scores.ptr<float>(),heads,
        position+1,work.scores.dim(1),stream,device_position);
    infer::paged_attention_av_f32(work.scores.ptr<float>(),pages,work.attended.ptr<float>(),
        position+1,tokens_per_page,heads,kv_heads,head_dim_,work.scores.dim(1),stream,device_position);
    output_.set_input(0,work.attended);output_.set_output(0,work.projected);output_.forward(stream);
    add_.forward(input,work.projected,work.residual,stream);
    mlp_.forward(work.residual,output,work.mlp,stream);
}

void TransformerBlock::forward_paged(const Tensor& input, Tensor& output,
                                     infer::AsyncKVPipeline::LayerView kv, int32_t position,
                                     TransformerWorkspace& work, cudaStream_t stream) {
    const auto width = static_cast<int32_t>(kv.row_elements);
    auto key = Tensor::borrow(DataType::FLOAT32, {width}, kv.key, DeviceType::CUDA);
    auto value = Tensor::borrow(DataType::FLOAT32, {width}, kv.value, DeviceType::CUDA);
    forward_paged(input, output, key, value, kv.pages, kv.tokens_per_page, position, work, stream);
}
}
