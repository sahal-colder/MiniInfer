#include "mini_infer/cache/mini_infer_kv_cache.h"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include "mini_infer/weights/mini_infer_mmap.h"

namespace mini_infer {

KVCache::KVCache(int32_t layer_num, int32_t max_seq_len, int32_t kv_dim,
                 std::shared_ptr<DeviceAllocator> allocator)
    : layer_num_(layer_num), max_seq_len_(max_seq_len), kv_dim_(kv_dim) {
    if (!allocator || (allocator->device_type() != DeviceType::CPU &&
                       allocator->device_type() != DeviceType::CUDA)) {
        throw std::invalid_argument("KVCache requires a CPU/CUDA allocator");
    }
    // 先建立无存储的 Tensor，借用其正维度/乘法/字节溢出检查，确认
    // 两块缓存的总字节数也能表达，然后再申请实际内存。
    key_cache_ = Tensor(DataType::FLOAT32, {layer_num, max_seq_len, kv_dim});
    value_cache_ = Tensor(DataType::FLOAT32, {layer_num, max_seq_len, kv_dim});
    if (key_cache_.byte_size() > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error("KVCache total byte size overflow");
    }
    valid_lengths_.assign(static_cast<std::size_t>(layer_num), 0);
    key_cache_.allocate(allocator);
    // 两个 Buffer 各自保留 allocator 的 shared_ptr；这里复制指针即可。
    // 局部 shared_ptr 析构不会销毁仍被 Buffer 引用的 allocator。
    value_cache_.allocate(allocator);
    // 构造不清零：history 只暴露写过的前缀，未使用的容量不参与计算。
    // 若第二次分配失败，成员 Tensor 的 RAII 自动释放第一次分配。
}

void KVCache::check_layer(int32_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= layer_num_) {
        throw std::out_of_range("KVCache layer index out of range");
    }
}

int32_t KVCache::valid_length(int32_t layer_idx) const {
    check_layer(layer_idx);
    return valid_lengths_[static_cast<std::size_t>(layer_idx)];
}

std::size_t KVCache::offset(int32_t layer_idx, int32_t token_pos) const {
    // 构造时已检查完整 Tensor 的大小；在合法索引范围中，size_t
    // 乘法也安全。不要用课件片段里的 int32_t 计算大型缓存的偏移。
    return (static_cast<std::size_t>(layer_idx) * max_seq_len_ + token_pos) * kv_dim_;
}

std::pair<Tensor, Tensor> KVCache::slice_kv_cache(int32_t layer_idx,
                                                 int32_t token_pos) const {
    check_layer(layer_idx);
    if (token_pos < 0 || token_pos >= max_seq_len_) {
        throw std::out_of_range("KVCache token position out of range");
    }
    const std::size_t start = offset(layer_idx, token_pos);
    return {key_cache_.view(start, {kv_dim_}), value_cache_.view(start, {kv_dim_})};
}

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
    Tensor key_slot = key_cache_.view(offset(layer_idx, pos), key.dims());
    Tensor value_slot = value_cache_.view(offset(layer_idx, pos), value.dims());
    get_kv_cache_write_kernel(device_type())(key, value, key_slot, value_slot, stream);
    // GPU 情况表示写入已提交；不是 GPU 已完成。后续使用须遵循 stream
    // 顺序。局部视图销毁无妨，缓存本体仍持有两块实际存储。
    commit(layer_idx, pos, count);
}

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

namespace {

int32_t kv_dim_from_config(const ModelConfig& config) {
    if (config.dim <= 0 || config.head_num <= 0 || config.kv_head_num <= 0 ||
        config.dim % config.head_num != 0 ||
        config.head_num % config.kv_head_num != 0) {
        throw std::invalid_argument("KVCache requires valid MHA/GQA head dimensions");
    }
    // kv_head_num <= head_num，因此结果 <= dim，不会发生 int32_t 溢出。
    return (config.dim / config.head_num) * config.kv_head_num;
}

}  // namespace

// 委托给上面的构造函数，避免重复分配逻辑。std::move 转移局部 shared_ptr，
// 不移动底层 allocator 对象；两个 Buffer 会分别保留其所有权。
KVCache::KVCache(const ModelConfig& config, std::shared_ptr<DeviceAllocator> allocator)
    : KVCache(config.layer_num, config.seq_len, kv_dim_from_config(config),
               std::move(allocator)) {}

namespace detail {

void check_kv_cache_write(const Tensor& key, const Tensor& value,
                          const Tensor& key_slot, const Tensor& value_slot,
                          DeviceType device) {
    for (const Tensor* tensor : {&key, &value, &key_slot, &value_slot}) {
        if (tensor->empty() || tensor->data_type() != DataType::FLOAT32 ||
            tensor->device_type() != device ||
            (tensor->dims().size() != 1 && tensor->dims().size() != 2) ||
            tensor->dims() != key.dims()) {
            throw std::invalid_argument("KVCache write requires same-shape FP32 tensors");
        }
    }
    // 两个只读输入可以相同；所有写区域须彼此独立且不与输入重叠。
    if (key_slot.overlaps(value_slot) || key_slot.overlaps(key) ||
        key_slot.overlaps(value) || value_slot.overlaps(key) ||
        value_slot.overlaps(value)) {
        throw std::invalid_argument("KVCache write destinations must not overlap");
    }
}

}  // namespace detail

void kv_cache_write_cpu(const Tensor& key, const Tensor& value,
                        Tensor& key_slot, Tensor& value_slot, cudaStream_t) {
    detail::check_kv_cache_write(key, value, key_slot, value_slot, DeviceType::CPU);
    // 保留可观察的标量循环；只写新 token，不搬运前面已经缓存的 K/V。
    for (std::size_t i = 0; i < key.size(); ++i) {
        key_slot.ptr<float>()[i] = key.ptr<float>()[i];
        value_slot.ptr<float>()[i] = value.ptr<float>()[i];
    }
}

KVCacheWriteKernel get_kv_cache_write_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return kv_cache_write_cpu;
        case DeviceType::CUDA: return kv_cache_write_cuda;
        default: throw std::invalid_argument("Unknown KVCache write backend");
    }
}

}  // namespace mini_infer
