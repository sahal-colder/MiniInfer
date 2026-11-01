#include "mini_infer/ops/mini_infer_attention.h"
#include "mini_infer/ops/mini_infer_softmax.h"

#include <cmath>
#include <limits>

namespace mini_infer {
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
