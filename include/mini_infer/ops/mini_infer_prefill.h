#pragma once
#include "mini_infer/tensor/mini_infer_tensor.h"

namespace mini_infer::detail {
void prefill_embedding_cuda(const Tensor& ids, const Tensor& weight, Tensor& output, cudaStream_t);
void prefill_rope_cuda(Tensor& query, Tensor& key, const Tensor& value,
                      const float* const* pages, std::size_t tokens_per_page,
                      int32_t position, int32_t head_dim, cudaStream_t);
void prefill_attention_cuda(const Tensor& query, const float* keys, const float* values,
                           const float* const* pages, std::size_t tokens_per_page,
                           Tensor& scores, Tensor& output, int32_t position,
                           int32_t heads, int32_t kv_heads, cudaStream_t);
}
