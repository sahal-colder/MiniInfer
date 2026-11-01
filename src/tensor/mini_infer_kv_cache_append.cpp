// 第 15 课第 2 步：验证连续追加范围，复制当前 K/V，成功后增加有效长度。
#include "mini_infer/tensor/mini_infer_kv_cache.h"
#include "mini_infer/ops/mini_infer_kv_cache_kernel.h"
#include <stdexcept>
namespace mini_infer {
void KVCache::check_append_range(int32_t layer_idx, int32_t token_pos,
                                int32_t count) const {
    check_layer(layer_idx);
    if (token_pos != valid_length(layer_idx) || count <= 0) {
        throw std::invalid_argument("KVCache appends require a contiguous valid prefix");
    }
    // count + token_pos 可能溢出，改用剩余容量进行比较。
    if (count > max_seq_len_ - token_pos) {
        throw std::out_of_range("KVCache capacity exceeded");
    }
}

void KVCache::commit(int32_t layer_idx, int32_t token_pos, int32_t token_count) {
    check_append_range(layer_idx, token_pos, token_count);
    reserve(token_pos + token_count);
    valid_lengths_[static_cast<std::size_t>(layer_idx)] += token_count;
}

void KVCache::append(int32_t layer_idx, const Tensor& key, const Tensor& value,
                      cudaStream_t stream) {
    check_layer(layer_idx);
    if (key.empty() || value.empty() || key.dims() != value.dims() ||
        (key.dims().size() != 1 && key.dims().size() != 2) ||
        key.dims().back() != kv_dim_) {
        throw std::invalid_argument("KVCache append expects matching [D] or [T,D]");
    }
    const int32_t count = key.dims().size() == 1 ? 1 : key.dim(0);
    const int32_t pos = valid_length(layer_idx);
    check_append_range(layer_idx, pos, count);
    // 检查整个 backing storage，而非仅检查目标切片，防止缓存内部
    // 历史被当作新输入。直接在缓存上投影时请用 slice + commit。
    if (key.overlaps(key_cache_) || key.overlaps(value_cache_) ||
        value.overlaps(key_cache_) || value.overlaps(value_cache_)) {
        throw std::invalid_argument("KVCache append inputs must be outside the cache");
    }
    reserve(pos + count, stream);
    Tensor key_slot = key_cache_.view(offset(layer_idx, pos), key.dims());
    Tensor value_slot = value_cache_.view(offset(layer_idx, pos), value.dims());
    get_kv_cache_write_kernel(device_type())(key, value, key_slot, value_slot, stream);
    // GPU 情况表示写入已提交；不是 GPU 已完成。后续使用须遵循 stream
    // 顺序。局部视图销毁无妨，缓存本体仍持有两块实际存储。
    commit(layer_idx, pos, count);
}

}  // namespace mini_infer
