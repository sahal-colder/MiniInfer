#include "mini_infer/ops/mini_infer_prefill.h"
#include "mini_infer/ops/mini_infer_softmax.h"
#include <math_constants.h>

namespace mini_infer::detail {
namespace {
__global__ void gather(const int32_t* ids, const float* weights, float* out, int width) {
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        out[std::size_t(blockIdx.x) * width + i] = weights[std::size_t(ids[blockIdx.x]) * width + i];
}

__global__ void rope_store(float* q, float* k, const float* v, const float* const* pages,
                           int page_tokens, int start, int dim, int kv_dim, int head_dim) {
    const auto token = blockIdx.x;
    const auto position = start + token;
    for (int i = threadIdx.x * 2; i < dim; i += blockDim.x * 2) {
        const float angle = position / powf(10000.f, float(i % head_dim) / head_dim);
        const float c = cosf(angle), s = sinf(angle);
        auto* query = q + std::size_t(token) * dim + i;
        const float a = query[0], b = query[1];
        query[0] = a * c - b * s;
        query[1] = a * s + b * c;
        if (i >= kv_dim) continue;
        auto* key = k + std::size_t(token) * kv_dim + i;
        const float ka = key[0], kb = key[1];
        key[0] = ka * c - kb * s;
        key[1] = ka * s + kb * c;
        if (!pages) continue;
        auto* dest = const_cast<float*>(pages[position / page_tokens]) + (position % page_tokens) * kv_dim + i;
        dest[0] = key[0]; dest[1] = key[1];
        dest[page_tokens * kv_dim] = v[std::size_t(token) * kv_dim + i];
        dest[page_tokens * kv_dim + 1] = v[std::size_t(token) * kv_dim + i + 1];
    }
}

__device__ float dot(const float* q, const float* k, int width) {
    float result = 0;
    int i = 0;
    if ((reinterpret_cast<std::uintptr_t>(q) % 16 == 0) &&
        (reinterpret_cast<std::uintptr_t>(k) % 16 == 0)) {
        for (; i <= width - 4; i += 4) {
            const auto a = *reinterpret_cast<const float4*>(q + i);
            const auto b = *reinterpret_cast<const float4*>(k + i);
            result += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
    }
    for (; i < width; ++i) result += q[i] * k[i];
    return result;
}

__global__ void qk(const float* query, const float* keys, const float* const* pages, float* scores,
                    int page_tokens, int start, int heads, int kv_heads, int head_dim, int stride) {
    const int token = blockIdx.x / heads, head = blockIdx.x % heads;
    const int kv_dim = kv_heads * head_dim, offset = head / (heads / kv_heads) * head_dim;
    const auto* q = query + (std::size_t(token) * heads + head) * head_dim;
    auto* score = scores + std::size_t(blockIdx.x) * stride;
    for (int t = threadIdx.x; t < stride; t += blockDim.x) {
        if (t > start + token) { score[t] = -CUDART_INF_F; continue; }
        const auto* key = pages ? pages[t / page_tokens] + (t % page_tokens) * kv_dim : keys + std::size_t(t) * kv_dim;
        score[t] = dot(q, key + offset, head_dim) * rsqrtf(float(head_dim));
    }
}

__global__ void av(const float* scores, const float* values, const float* const* pages, float* output,
                    int page_tokens, int start, int heads, int kv_heads, int head_dim, int stride) {
    const int token = blockIdx.x / heads, head = blockIdx.x % heads;
    const int kv_dim = kv_heads * head_dim, offset = head / (heads / kv_heads) * head_dim;
    const auto* score = scores + std::size_t(blockIdx.x) * stride;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float result = 0;
        for (int t = 0; t <= start + token; ++t) {
            const auto* value = pages ? pages[t / page_tokens] + (page_tokens + t % page_tokens) * kv_dim : values + std::size_t(t) * kv_dim;
            result += score[t] * value[offset + i];
        }
        output[std::size_t(blockIdx.x) * head_dim + i] = result;
    }
}
}
void prefill_embedding_cuda(const Tensor& ids, const Tensor& weight, Tensor& output, cudaStream_t stream) {
    gather<<<ids.size(), 128, 0, stream>>>(ids.ptr<int32_t>(), weight.ptr<float>(), output.ptr<float>(), weight.dim(1));
    check_cuda(cudaGetLastError(), "prefill embedding");
}
void prefill_rope_cuda(Tensor& query, Tensor& key, const Tensor& value,
                      const float* const* pages, std::size_t tokens_per_page,
                      int32_t position, int32_t head_dim, cudaStream_t stream) {
    rope_store<<<query.dim(0), 128, 0, stream>>>(query.ptr<float>(), key.ptr<float>(), value.ptr<float>(),
        pages, tokens_per_page, position, query.dim(1), key.dim(1), head_dim);
    check_cuda(cudaGetLastError(), "prefill RoPE and KV store");
}
void prefill_attention_cuda(const Tensor& query, const float* keys, const float* values,
                           const float* const* pages, std::size_t tokens_per_page,
                           Tensor& scores, Tensor& output, int32_t position,
                           int32_t heads, int32_t kv_heads, cudaStream_t stream) {
    const auto rows = query.dim(0) * heads;
    const auto stride = scores.dims().back();
    qk<<<rows, 128, 0, stream>>>(query.ptr<float>(), keys, pages, scores.ptr<float>(),
        tokens_per_page, position, heads, kv_heads, query.dim(1) / heads, stride);
    check_cuda(cudaGetLastError(), "prefill QK");
    softmax_rows_cuda(scores.ptr<float>(), scores.ptr<float>(), rows, position + query.dim(0), stride, stream);
    av<<<rows, 128, 0, stream>>>(scores.ptr<float>(), values, pages, output.ptr<float>(),
        tokens_per_page, position, heads, kv_heads, query.dim(1) / heads, stride);
    check_cuda(cudaGetLastError(), "prefill AV");
}
}
