#pragma once
#include "infer/storage/storage_block.h"
namespace infer {
// FP32 decode reference: one query per head; each device page = [K,V], each
// K/V is [tokens_per_page,kv_heads,head_dim]. Pointers are a DEVICE array.
// All pages must remain GPU-resident under READ leases until stream completes.
void paged_attention_f32(const float* query,const float* const* pages,float* output,
    std::size_t page_count,std::size_t context_tokens,std::size_t tokens_per_page,
    int query_heads,int kv_heads,int head_dim,std::size_t physical_page_bytes,cudaStream_t stream);
}
namespace infer {
// Compatibility stages: same FP32 dot/AV order as native contiguous attention.
// Native supplies its unchanged softmax stage between these calls.
void paged_attention_qk_f32(const float* query,const float* const* pages,float* scores,
    std::size_t context,std::size_t tokens_per_page,int heads,int kv_heads,int dim,int capacity,cudaStream_t stream,
    const int32_t* device_position = nullptr);
void paged_attention_av_f32(const float* scores,const float* const* pages,float* output,
    std::size_t context,std::size_t tokens_per_page,int heads,int kv_heads,int dim,int capacity,cudaStream_t stream,
    const int32_t* device_position = nullptr);
// Store a rotated K and V from static scratch into the changing current row.
void paged_kv_store_f32(const float* key, const float* value, const float* const* pages,
    std::size_t tokens_per_page, int width, const int32_t* device_position, cudaStream_t stream);
}
