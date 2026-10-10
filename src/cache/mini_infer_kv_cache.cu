#include "mini_infer/cache/mini_infer_kv_cache.h"

#include <algorithm>

namespace mini_infer {
namespace {

// 写入只涉及当前的一段连续 K/V。每个线程复制若干元素，K/V 一起
// 写回；不需要归约或共享内存，也不需要为整个历史做 concat。
__global__ void write_kv_f32(const float* key, const float* value,
                            float* key_slot, float* value_slot, std::size_t count) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                              + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = start; i < count; i += stride) {
        key_slot[i] = key[i];
        value_slot[i] = value[i];
    }
}

}  // namespace

void kv_cache_write_cuda(const Tensor& key, const Tensor& value,
                         Tensor& key_slot, Tensor& value_slot, cudaStream_t stream) {
    detail::check_kv_cache_write(key, value, key_slot, value_slot, DeviceType::CUDA);
    constexpr unsigned int threads = 256;
    // 限制 grid，长 prefill 用 grid-stride 循环继续处理。除法/余数
    // 向上取整也避开 count + threads - 1 的溢出。
    const std::size_t needed = key.size() / threads + (key.size() % threads != 0);
    const unsigned int blocks = static_cast<unsigned int>(std::min<std::size_t>(needed, 4096));
    write_kv_f32<<<blocks, threads, 0, stream>>>(key.ptr<float>(), value.ptr<float>(),
                                               key_slot.ptr<float>(),
                                               value_slot.ptr<float>(), key.size());
    check_cuda(cudaGetLastError(), "launch KVCache write");
    // 默认/非默认 stream 都只提交 kernel，不暗中同步。
}

namespace detail {

void zero_kv_cache_cuda(Tensor& key_cache, Tensor& value_cache, cudaStream_t stream) {
    // 这里只由 KVCache::reset 调用；两个 Tensor 是构造时创建的完整
    // FP32 缓存。全零字节正好表示 float 的 +0.0f。
    check_cuda(cudaMemsetAsync(key_cache.ptr<float>(), 0, key_cache.byte_size(), stream),
               "clear KVCache keys");
    check_cuda(cudaMemsetAsync(value_cache.ptr<float>(), 0, value_cache.byte_size(), stream),
               "clear KVCache values");
}

}  // namespace detail
}  // namespace mini_infer
