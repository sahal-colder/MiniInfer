// 第 15 课第 1 步：分配两块缓存，按 layer/token 算偏移，返回一对 view。
#include "mini_infer/tensor/mini_infer_kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

// 1. 构造 KVCache，申请 K/V 两块大内存
// 2. check_layer()：检查 layer 合不合法
// 3. valid_length()：查询某层已经存了多少 token
// 4. offset()：算某层某 token 在大内存里的位置
// 5. slice_kv_cache()：返回那个位置对应的 K/V view

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
    capacity_bytes_ = 2 * key_cache_.byte_size();
    allocated_tokens_ = std::min(max_seq_len, int32_t(64));
    allocator_ = allocator;
    key_cache_ = Tensor(DataType::FLOAT32, {layer_num, allocated_tokens_, kv_dim});
    value_cache_ = Tensor(DataType::FLOAT32, {layer_num, allocated_tokens_, kv_dim});
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
    return (static_cast<std::size_t>(layer_idx) * allocated_tokens_ + token_pos) * kv_dim_;
}

std::pair<Tensor, Tensor> KVCache::slice_kv_cache(int32_t layer_idx, int32_t token_pos,
                                                  cudaStream_t stream) const {
    check_layer(layer_idx);
    if (token_pos < 0 || token_pos >= max_seq_len_) {
        throw std::out_of_range("KVCache token position out of range");
    }
    reserve(token_pos + 1, stream);
    const std::size_t start = offset(layer_idx, token_pos);
    return {key_cache_.view(start, {kv_dim_}), value_cache_.view(start, {kv_dim_})};
}

void KVCache::reserve(int32_t tokens, cudaStream_t stream) const {
    if (tokens < 0 || tokens > max_seq_len_)
        throw std::out_of_range("KV reservation exceeds maximum context");
    if (tokens <= allocated_tokens_)
        return;
    auto capacity = allocated_tokens_;
    while (capacity < tokens)
        capacity = capacity > max_seq_len_ / 2 ? max_seq_len_ : capacity * 2;
    Tensor keys(DataType::FLOAT32, {layer_num_, capacity, kv_dim_}, allocator_);
    Tensor values(DataType::FLOAT32, {layer_num_, capacity, kv_dim_}, allocator_);
    if (device_type() == DeviceType::CUDA)
        check_cuda(cudaStreamSynchronize(stream), "wait for KV growth");
    CPUDeviceAllocator copier;
    const auto kind =
        device_type() == DeviceType::CUDA ? MemcpyKind::CUDA2CUDA : MemcpyKind::CPU2CPU;
    for (int32_t layer = 0; layer < layer_num_; ++layer) {
        const auto count = static_cast<std::size_t>(valid_lengths_[layer]) * kv_dim_;
        if (!count)
            continue;
        const auto target = static_cast<std::size_t>(layer) * capacity * kv_dim_;
        copier.copy(key_cache_.ptr<float>() + offset(layer, 0), keys.ptr<float>() + target,
                    count * sizeof(float), kind);
        copier.copy(value_cache_.ptr<float>() + offset(layer, 0), values.ptr<float>() + target,
                    count * sizeof(float), kind);
    }
    key_cache_ = std::move(keys);
    value_cache_ = std::move(values);
    allocated_tokens_ = capacity;
}
}  // namespace mini_infer
