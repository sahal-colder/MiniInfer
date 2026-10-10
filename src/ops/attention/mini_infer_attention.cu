#include "mini_infer/ops/attention/mini_infer_attention.h"
#include "mini_infer/ops/softmax/mini_infer_softmax.h"

#include <algorithm>
#include <cstdint>

namespace mini_infer {
namespace {
constexpr unsigned int threads = 128;

__device__ float head_dot(const float* q, const float* k, int32_t size) {
    float sum = 0.0f;
    int32_t i = 0;
    // float4 要求两个实际地址都 16 字节对齐，不能只检查 cudaMalloc 首址。
    // head_dim/kv_dim 不整齐时，后面的头或 token 行可能不对齐。
    if ((reinterpret_cast<std::uintptr_t>(q) % alignof(float4) == 0) &&
        (reinterpret_cast<std::uintptr_t>(k) % alignof(float4) == 0)) {
        for (; i <= size - 4; i += 4) {
            const float4 a = *reinterpret_cast<const float4*>(q + i);
            const float4 b = *reinterpret_cast<const float4*>(k + i);
            sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
    }
    // 非对齐行走完整标量循环；对齐行也需要处理不足四个数的尾部。
    for (; i < size; ++i) sum += q[i] * k[i];
    return sum;
}

// 一个 block 处理一个 head，每个线程负责一个或多个历史 token 的内积。
__global__ void query_key_f32(const float* query, const float* keys, float* scores,
                              int32_t heads, int32_t kv_mul, int32_t head_dim,
                              int32_t length, int32_t kv_dim, int32_t capacity) {
    for (std::size_t h = blockIdx.x; h < std::size_t(heads); h += gridDim.x) {
        const float* q = query + h * head_dim;
        float* score = scores + h * capacity;
        const std::size_t head_offset = (h / kv_mul) * head_dim;
        const float scale = rsqrtf(float(head_dim));
        for (std::size_t t = threadIdx.x; t < std::size_t(length); t += blockDim.x) {
            const float* k = keys + t * kv_dim + head_offset;
            score[t] = head_dot(q, k, head_dim) * scale;
        }
    }
}

// 每线程负责头内一个或多个分量；沿时间维读取 Softmax 概率和对应 V。
__global__ void attention_value_f32(const float* scores, const float* values, float* output,
                                    int32_t heads, int32_t kv_mul, int32_t head_dim,
                                    int32_t length, int32_t kv_dim, int32_t capacity) {
    for (std::size_t h = blockIdx.x; h < std::size_t(heads); h += gridDim.x) {
        const float* score = scores + h * capacity;
        const std::size_t head_offset = (h / kv_mul) * head_dim;
        float* out = output + h * head_dim;
        for (std::size_t i = threadIdx.x; i < std::size_t(head_dim); i += blockDim.x) {
            float sum = 0.0f;
            for (int32_t t = 0; t < length; ++t)
                sum += score[t] * values[std::size_t(t) * kv_dim + head_offset + i];
            out[i] = sum;
        }
    }
}
}  // namespace

void attention_kernel_cuda(const Tensor& query, const Tensor& keys, const Tensor& values,
                           Tensor& scores, Tensor& output, int32_t heads,
                           int32_t kv_heads, cudaStream_t stream) {
    detail::check_attention_tensors(query, keys, values, scores, output, heads, kv_heads, DeviceType::CUDA);
    const int32_t head_dim = query.dim(0) / heads;
    const int32_t length = keys.dim(0), kv_dim = keys.dim(1), capacity = scores.dim(1);
    const int32_t kv_mul = heads / kv_heads;
    const unsigned int blocks = static_cast<unsigned int>(std::min(heads, 65535));
    // 三次 launch 都使用同一 stream，后一阶段会等待前一阶段完成。
    // 用阶段边界代替一个巨大内核，让第 18 课每一步都能独立对照 CPU。
    query_key_f32<<<blocks, threads, 0, stream>>>(query.ptr<float>(), keys.ptr<float>(),
        scores.ptr<float>(), heads, kv_mul, head_dim, length, kv_dim, capacity);
    check_cuda(cudaGetLastError(), "Attention QK kernel launch");
    detail::softmax_rows_cuda(scores.ptr<float>(), scores.ptr<float>(), heads, length, capacity, stream);
    attention_value_f32<<<blocks, threads, 0, stream>>>(scores.ptr<float>(), values.ptr<float>(),
        output.ptr<float>(), heads, kv_mul, head_dim, length, kv_dim, capacity);
    check_cuda(cudaGetLastError(), "Attention AV kernel launch");
}
}  // namespace mini_infer
