#include "mini_infer/ops/rope/mini_infer_rope.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mini_infer::detail {
namespace {
__global__ void rotate_pairs(float* q, float* k, std::size_t q_pairs,
                             std::size_t k_pairs, int head_dim, int pos, float theta) {
    // 一个线程负责一个偶奇对，同步旋转 Q 以及（若存在）对应的 K。
    for (std::size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
         pair < q_pairs; pair += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t i = 2 * pair;
        const float angle = pos / powf(theta, static_cast<float>(i % head_dim) / head_dim);
        const float c = cosf(angle), s = sinf(angle);
        const float qa = q[i], qb = q[i + 1];
        q[i] = qa * c - qb * s;
        q[i + 1] = qa * s + qb * c;
        if (pair < k_pairs) {
            const float ka = k[i], kb = k[i + 1];
            k[i] = ka * c - kb * s;
            k[i + 1] = ka * s + kb * c;
        }
    }
}
}
void rope_cuda(Tensor& query, Tensor& key, int32_t head_dim,
               int32_t position, float theta, cudaStream_t stream) {
    const auto pairs = query.size() / 2;
    const int blocks = static_cast<int>(std::min<std::size_t>((pairs + 255) / 256, 65535));
    rotate_pairs<<<blocks, 256, 0, stream>>>(query.ptr<float>(), key.ptr<float>(),
                                            pairs, key.size() / 2, head_dim, position, theta);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("RoPE launch: ") + cudaGetErrorString(error));
    }
}
}  // namespace mini_infer::detail
