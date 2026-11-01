#pragma once
#include "mini_infer/tensor/mini_infer_tensor.h"
namespace mini_infer {
// 两个后端有一致的签名。低层函数只复制当前 K/V，不更新有效长度。
// source/destination 都要求 [D] 或 [T,D] 且四者形状一致。
using KVCacheWriteKernel = void (*)(const Tensor&, const Tensor&, Tensor&, Tensor&,
                                    cudaStream_t);
KVCacheWriteKernel get_kv_cache_write_kernel(DeviceType device);
void kv_cache_write_cpu(const Tensor& key, const Tensor& value,
                        Tensor& key_slot, Tensor& value_slot, cudaStream_t stream);
void kv_cache_write_cuda(const Tensor& key, const Tensor& value,
                         Tensor& key_slot, Tensor& value_slot, cudaStream_t stream);

namespace detail {
// 共享元数据检查，不解引用 GPU 指针；直接调用后端也受保护。
void check_kv_cache_write(const Tensor& key, const Tensor& value,
                          const Tensor& key_slot, const Tensor& value_slot,
                          DeviceType device);
void zero_kv_cache_cuda(Tensor& key_cache, Tensor& value_cache, cudaStream_t stream);
}  // namespace detail

}  // namespace mini_infer
