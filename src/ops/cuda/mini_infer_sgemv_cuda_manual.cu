#include <cuda_runtime.h>
// 第 13 课 CUDA 进阶：读完 mini_infer_sgemv.cu 后，再单独学习 manual。
#include "mini_infer/ops/mini_infer_sgemv.h"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include "ops/cuda/sgemv_common.cuh"

namespace mini_infer {
namespace {

__device__ float warp_reduce_sum(float value) {
    // 所有 32 个 lane 参加；最后仅 lane 0 得到整个 warp 的和。
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void sgemv_manual(const float* input, const float* weight, float* output,
                             std::size_t in, std::size_t out, std::size_t total,
                             float scale) {
    __shared__ float warp_sums[4];
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;
    for (std::size_t index = blockIdx.x; index < total; index += gridDim.x) {
        const float local_sum = thread_dot(input + (index / out) * in,
                                            weight + (index % out) * in, in);
        const float warp_sum = warp_reduce_sum(local_sum);
        if (lane == 0) warp_sums[warp] = warp_sum;
        __syncthreads();  // 四个 warp 写完后，warp 0 才能读取。
        if (warp == 0) {
            const float sum = warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
            if (lane == 0) output[index] = sum * scale;
        }
        __syncthreads();  // 保护下一轮对 warp_sums 的复用。
    }
}

}  // namespace

void sgemv_kernel_cuda_manual(const Tensor& input, const Tensor& weight, Tensor& output,
                              float scale, cudaStream_t stream) {
    detail::check_sgemv_tensors(input, weight, output, DeviceType::CUDA);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(output.size(), 65535));
    sgemv_manual<<<blocks, threads_per_block, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(),
        static_cast<std::size_t>(weight.dim(1)),
        static_cast<std::size_t>(weight.dim(0)), output.size(), scale);
    check_cuda(cudaGetLastError(), "Sgemv manual launch");
}

}  // namespace mini_infer
