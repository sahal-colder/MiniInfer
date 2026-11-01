#include <cuda_runtime.h>
#include "mini_infer/ops/mini_infer_sgemv.h"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include "ops/cuda/sgemv_common.cuh"

namespace mini_infer {
namespace {

// 主线只读这个 kernel：一个 block 算一个输出，固定 128 线程使用 CUB。
// 不必先理解模板的模式组合，再猜本次实际启动了哪一种归约。
__global__ void sgemv_f32(const float* input, const float* weight, float* output,
                          std::size_t in, std::size_t out, std::size_t total,
                          float scale) {

    // 输入矩阵：[B,K]，其中列数 K=in，B 是样本数（Batch size）。
    // 权重矩阵：[N,K]，其中行数 N=out，列数 K=in。
    // 输出矩阵：[B,N]，即 [B,out]。   
    using BlockReduce = cub::BlockReduce<float, threads_per_block>;
    __shared__ BlockReduce::TempStorage reduce_storage;
    for (std::size_t index = blockIdx.x; index < total; index += gridDim.x) {
        // index/out 是输入向量编号；index%out 是权重行编号。
        const float* input_row = input + (index / out) * in;
        const float* weight_row = weight + (index % out) * in;
        const float local_sum = thread_dot(input_row, weight_row, in);
        const float sum = BlockReduce(reduce_storage).Sum(local_sum);
        if (threadIdx.x == 0) output[index] = sum * scale;
        // grid-stride 的下一轮会复用同一块共享内存。
        __syncthreads();
    }
}
}  // namespace

void sgemv_kernel_cuda(const Tensor& input, const Tensor& weight, Tensor& output,
                      float scale, cudaStream_t stream) {
    detail::check_sgemv_tensors(input, weight, output, DeviceType::CUDA);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(output.size(), 65535));
    sgemv_f32<<<blocks, threads_per_block, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(),
        static_cast<std::size_t>(weight.dim(1)),
        static_cast<std::size_t>(weight.dim(0)), output.size(), scale);
    // 此处检查提交错误；调用者同步 stream 时检查执行错误。
    check_cuda(cudaGetLastError(), "Sgemv kernel launch");
}
}  // namespace mini_infer
