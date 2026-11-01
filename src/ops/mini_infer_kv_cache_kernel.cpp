// 后端只搬运当前 K/V，不知道 layer、token_pos 或有效长度。
#include "mini_infer/ops/mini_infer_kv_cache_kernel.h"
#include <stdexcept>
namespace mini_infer {

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
