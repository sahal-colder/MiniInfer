#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mini_infer {
namespace {
constexpr int threads_per_block = 128;
// 一个线程只计算自己负责的列；返回局部和，块归约由外层 kernel 完成。
__device__ float thread_dot(const float* input_row, const float* weight_row,
                            std::size_t in) {
    float local_sum = 0.0f;
    // cudaMalloc 的首地址对齐，不代表 W 的每一行都对齐：
    // 例如 in=5 时下一行偏移 20 B，无法安全解引用 float4*。
    // Tensor::borrow 也允许只按 float 对齐的外部地址，必须两端都检查。
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(input_row) % alignof(float4) == 0 &&
        reinterpret_cast<std::uintptr_t>(weight_row) % alignof(float4) == 0;
    if (aligned) {
        const std::size_t packs = in / 4;
        const float4* input_packs = reinterpret_cast<const float4*>(input_row);
        const float4* weight_packs = reinterpret_cast<const float4*>(weight_row);
        // 线程 tid 处理第 tid、tid+128、... 个四元素包。
        for (std::size_t p = threadIdx.x; p < packs; p += blockDim.x) {
            const float4 input_pack = input_packs[p];
            const float4 weight_pack = weight_packs[p];
            local_sum += input_pack.x * weight_pack.x + input_pack.y * weight_pack.y
                       + input_pack.z * weight_pack.z + input_pack.w * weight_pack.w;
        }
        // 不足四个元素的尾部只能标量读取，不能读到下一行。
        for (std::size_t i = packs * 4 + threadIdx.x; i < in; i += blockDim.x) {
            local_sum += input_row[i] * weight_row[i];
        }
    } else {
        for (std::size_t i = threadIdx.x; i < in; i += blockDim.x) {
            local_sum += input_row[i] * weight_row[i];
        }
    }

    return local_sum;
}

}  // namespace
}  // namespace mini_infer
