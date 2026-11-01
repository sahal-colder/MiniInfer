// 第 15 课第 3 步：读取有效前缀、开始新请求时复用已有容量。
#include "mini_infer/tensor/mini_infer_kv_cache.h"
#include "mini_infer/ops/mini_infer_kv_cache_kernel.h"
#include <algorithm>
namespace mini_infer {
std::pair<Tensor, Tensor> KVCache::history(int32_t layer_idx) const {
    const int32_t count = valid_length(layer_idx);
    if (count == 0) return {};
    const std::size_t start = offset(layer_idx, 0);
    return {key_cache_.view(start, {count, kv_dim_}),
            value_cache_.view(start, {count, kv_dim_})};
}

void KVCache::reset(bool zero_storage, cudaStream_t stream) {
    if (zero_storage) {
        if (device_type() == DeviceType::CPU) {
            std::fill_n(key_cache_.ptr<float>(), key_cache_.size(), 0.0f);
            std::fill_n(value_cache_.ptr<float>(), value_cache_.size(), 0.0f);
        } else {
            detail::zero_kv_cache_cuda(key_cache_, value_cache_, stream);
        }
    }
    std::fill(valid_lengths_.begin(), valid_lengths_.end(), 0);
}

}  // namespace mini_infer
