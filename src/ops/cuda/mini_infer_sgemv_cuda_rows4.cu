#include <cuda_runtime.h>
// 第 13 课 CUDA 进阶：读完 mini_infer_sgemv.cu 后，再单独学习 rows4。
#include "mini_infer/ops/mini_infer_sgemv.h"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include "ops/cuda/sgemv_common.cuh"

namespace mini_infer {
namespace {

__global__ void sgemv_rows4(const float* input, const float* weight, float* output,
                            std::size_t in, std::size_t out, std::size_t total,
                            float scale) {
    using BlockReduce = cub::BlockReduce<float, threads_per_block>;
    __shared__ BlockReduce::TempStorage reduce_storage;
    const std::size_t groups = total / 4 + (total % 4 != 0);
    for (std::size_t group = blockIdx.x; group < groups; group += gridDim.x) {
        for (int local_row = 0; local_row < 4; ++local_row) {
            const std::size_t index = group * 4 + local_row;
            if (index >= total) break;  // 全 block 一致退出，不跳过部分线程的归约。
            const float local_sum = thread_dot(input + (index / out) * in,
                                                weight + (index % out) * in, in);
            const float sum = BlockReduce(reduce_storage).Sum(local_sum);
            if (threadIdx.x == 0) output[index] = sum * scale;
            __syncthreads();
        }
    }
}

}  // namespace

void sgemv_kernel_cuda_rows4(const Tensor& input, const Tensor& weight, Tensor& output,
                             float scale, cudaStream_t stream) {
    detail::check_sgemv_tensors(input, weight, output, DeviceType::CUDA);
    const std::size_t groups = output.size() / 4 + (output.size() % 4 != 0);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(groups, 65535));
    sgemv_rows4<<<blocks, threads_per_block, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(),
        static_cast<std::size_t>(weight.dim(1)),
        static_cast<std::size_t>(weight.dim(0)), output.size(), scale);
    check_cuda(cudaGetLastError(), "Sgemv rows4 launch");
}

}  // namespace mini_infer
