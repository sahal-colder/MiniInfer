#include "mini_infer/ops/attention/mini_infer_attention.h"
#include "mini_infer/ops/softmax/mini_infer_softmax.h"

#include <cmath>
#include <limits>

namespace mini_infer {
namespace detail {
void check_attention_tensors(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& scores, const Tensor& output,
                             int32_t heads, int32_t kv_heads, DeviceType device) {
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0) {
        throw std::invalid_argument("Attention needs positive head counts and heads % kv_heads == 0");
    }
    for (const Tensor* t : {&q, &k, &v, &scores, &output}) {
        if (t->empty() || t->size() == 0 || t->data_type() != DataType::FLOAT32 ||
            t->device_type() != device) {
            throw std::invalid_argument("Attention requires nonempty FP32 tensors on its backend");
        }
    }
    if (q.dims().size() != 1 || q.dim(0) % heads != 0 ||
        k.dims().size() != 2 || v.dims() != k.dims() ||
        k.dim(1) != kv_heads * (q.dim(0) / heads) || output.dims() != q.dims() ||
        scores.dims().size() != 2 || scores.dim(0) != heads || scores.dim(1) < k.dim(0)) {
        throw std::invalid_argument("Attention shape mismatch: q/out=[dim], k/v=[length,kv_dim], scores=[heads,capacity]");
    }
    for (const Tensor* dst : {&scores, &output}) {
        for (const Tensor* src : {&q, &k, &v}) {
            if (dst->overlaps(*src)) throw std::invalid_argument("Attention output overlaps an input");
        }
    }
    if (scores.overlaps(output)) throw std::invalid_argument("Attention scores overlap output");
}
}  // namespace detail

void attention_kernel_cpu(const Tensor& query, const Tensor& keys, const Tensor& values,
                          Tensor& scores, Tensor& output, int32_t heads,
                          int32_t kv_heads, cudaStream_t stream) {
    detail::check_attention_tensors(query, keys, values, scores, output, heads, kv_heads, DeviceType::CPU);
    if (stream) throw std::invalid_argument("CPU Attention does not accept a CUDA stream");
    const int32_t head_dim = query.dim(0) / heads;
    const int32_t length = keys.dim(0), kv_dim = keys.dim(1), capacity = scores.dim(1);
    const int32_t kv_mul = heads / kv_heads;
    const float scale = 1.0f / std::sqrt(float(head_dim));

    // 1. QK：各头分别与每个可见 token 的 K 做内积。
    for (int32_t h = 0; h < heads; ++h) {
        const float* q = query.ptr<float>() + std::size_t(h) * head_dim;
        float* score = scores.ptr<float>() + std::size_t(h) * capacity;
        const std::size_t head_offset = std::size_t(h / kv_mul) * head_dim;
        for (int32_t t = 0; t < length; ++t) {
            const float* k = keys.ptr<float>() + std::size_t(t) * kv_dim + head_offset;
            float dot = 0.0f;
            for (int32_t i = 0; i < head_dim; ++i) dot += q[i] * k[i];
            score[t] = dot * scale;
        }
    }
    // 2. 原位 Softmax：这里只计算 length，而不是 score 的预留容量。
    detail::softmax_rows_cpu(scores.ptr<float>(), scores.ptr<float>(), heads, length, capacity);
    // 3. AV：每头每个输出分量沿时间维加权求和，无须实际转置缓存。
    for (int32_t h = 0; h < heads; ++h) {
        const float* score = scores.ptr<float>() + std::size_t(h) * capacity;
        float* out = output.ptr<float>() + std::size_t(h) * head_dim;
        const std::size_t head_offset = std::size_t(h / kv_mul) * head_dim;
        for (int32_t i = 0; i < head_dim; ++i) {
            float sum = 0.0f;
            for (int32_t t = 0; t < length; ++t)
                sum += score[t] * values.ptr<float>()[std::size_t(t) * kv_dim + head_offset + i];
            out[i] = sum;
        }
    }
}

AttentionKernel get_attention_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return attention_kernel_cpu;
        case DeviceType::CUDA: return attention_kernel_cuda;
        default: throw std::invalid_argument("Unknown Attention backend");
    }
}

MultiHeadAttention::MultiHeadAttention(int32_t heads, int32_t kv_heads, int32_t head_dim)
    : head_num_(heads), kv_head_num_(kv_heads), head_dim_(head_dim) {
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || head_dim <= 0 ||
        head_dim > std::numeric_limits<int32_t>::max() / heads) {
        throw std::invalid_argument("Invalid Attention head configuration");
    }
}

void MultiHeadAttention::forward(const Tensor& query, const KVCache& cache,
                                 int32_t layer_idx, int32_t pos, Tensor& scores,
                                 Tensor& output, cudaStream_t stream) const {
    if (pos < 0 || pos >= cache.valid_length(layer_idx)) {
        throw std::out_of_range("Attention token_pos must refer to committed KV history");
    }
    if (cache.kv_dim() != kv_head_num_ * head_dim_ ||
        query.dims() != std::vector<int32_t>{head_num_ * head_dim_}) {
        throw std::invalid_argument("Attention configuration differs from query/cache dimensions");
    }
    // 输出是独立工作区：也不能覆盖被因果截断的较晚历史或其他层历史。
    // 此检查先于数值计算，避免某层计算意外破坏另一层已提交的缓存。
    for (int32_t layer = 0; layer < cache.layer_num(); ++layer) {
        auto [all_keys, all_values] = cache.history(layer);
        for (const Tensor* dst : {&scores, &output}) {
            if (dst->overlaps(all_keys) || dst->overlaps(all_values)) {
                throw std::invalid_argument("Attention workspace must not overwrite committed KV history");
            }
        }
    }
    auto [keys, values] = cache.history(layer_idx);
    keys = keys.view(0, {pos + 1, cache.kv_dim()});
    values = values.view(0, {pos + 1, cache.kv_dim()});
    get_attention_kernel(cache.device_type())(query, keys, values, scores, output,
                                             head_num_, kv_head_num_, stream);
}
}  // namespace mini_infer
