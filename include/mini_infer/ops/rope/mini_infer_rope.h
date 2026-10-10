#pragma once

#include "mini_infer/core/mini_infer_tensor.h"

namespace mini_infer {

// Llama 2 / llama2.c 的相邻偶奇配对 RoPE；不是 Hugging Face 的半区配对。
// 每个 head 内 (x[2j],x[2j+1]) 旋转 pos / theta^(2j/head_dim) 弧度。
// Q=[dim]，K=[kv_dim]，只原位旋转当前 token，V 和历史 K 不改动。
// head_dim 必须为偶数；GQA 的 Q/K 头数可以不同，但每头宽度相同。
// 调用顺序：QKV -> rope_inplace(Q,K) -> cache.commit -> attention。
void rope_inplace(Tensor& query, Tensor& key, int32_t head_dim,
                  int32_t position, float theta = 10000.0f,
                  cudaStream_t stream = nullptr);

namespace detail {
// 参数由上面的公开接口统一检查；CUDA 内核异步提交到同一 stream。
void rope_cuda(Tensor& query, Tensor& key, int32_t head_dim,
               int32_t position, float theta, cudaStream_t stream);
}
}  // namespace mini_infer
