#include "mini_infer/ops/softmax/mini_infer_softmax.h"

#include <cub/block/block_reduce.cuh>
#include <math_constants.h>  // CUDART_INF_F：归约最大值时使用的负无穷初值。
#include <algorithm>

namespace mini_infer {
namespace {
constexpr unsigned int threads = 128;

// 一个 block 处理一行。每个线程先做局部 max/sum，再用 CUB 合并。
// width 可以大于 blockDim.x；尾部线程仍参与归约和全部同步。
__global__ void softmax_f32(const float* input, float* output, std::size_t rows,
                            int32_t width, int32_t stride) {
    using Reduce = cub::BlockReduce<float, threads>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ float row_max;
    __shared__ float row_sum;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x = input + row * stride;
        float* y = output + row * stride;
        float local_max = -CUDART_INF_F;
        for (std::size_t i = threadIdx.x; i < std::size_t(width); i += blockDim.x)
            local_max = fmaxf(local_max, x[i]);
        const float maximum = Reduce(storage).Reduce(local_max, cub::Max());
        if (threadIdx.x == 0) row_max = maximum;
        __syncthreads();  // 广播最大值；也保证所有线程读完原始输入。

        float local_sum = 0.0f;
        for (std::size_t i = threadIdx.x; i < std::size_t(width); i += blockDim.x) {
            const float e = expf(x[i] - row_max);
            y[i] = e;
            local_sum += e;
        }
        const float sum = Reduce(storage).Sum(local_sum);
        if (threadIdx.x == 0) row_sum = sum;
        __syncthreads();
        for (std::size_t i = threadIdx.x; i < std::size_t(width); i += blockDim.x)
            y[i] /= row_sum;
        __syncthreads();  // 下一行复用共享内存之前，完成本行的读取。
    }
}
}  // namespace

namespace detail {
void softmax_rows_cuda(const float* input, float* output, std::size_t rows,
                       int32_t width, int32_t stride, cudaStream_t stream) {
    const unsigned int blocks = static_cast<unsigned int>(std::min<std::size_t>(rows, 65535));
    softmax_f32<<<blocks, threads, 0, stream>>>(input, output, rows, width, stride);
    check_cuda(cudaGetLastError(), "Softmax kernel launch");
}
}  // namespace detail

void softmax_kernel_cuda(const Tensor& input, Tensor& output, cudaStream_t stream) {
    detail::check_softmax_tensors(input, output, DeviceType::CUDA);
    const int32_t width = input.dims().back();
    detail::softmax_rows_cuda(input.ptr<float>(), output.ptr<float>(),
                              input.size() / width, width, width, stream);
}
}  // namespace mini_infer
