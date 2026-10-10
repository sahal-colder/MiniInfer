#include "mini_infer/ops/matmul/mini_infer_quantized_sgemv.h"

#include <cub/block/block_reduce.cuh>

#include <algorithm>

namespace mini_infer {
namespace {

// 固定 128 线程；rows_per_block 是普通整数 1 或 4，逐行执行同一公式。
// 补齐课件片段省略的归约、写回、尾行边界、stream 和 output_scale。
__global__ void quantized_sgemv_f32int8(const float* input, const int8_t* weight,
                                        const float* scales, std::size_t group_size,
                                        float* output, std::size_t in, std::size_t out,
                                        std::size_t total, float output_scale, int rows_per_block) {
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ BlockReduce::TempStorage reduce_storage;
    const std::size_t groups = total / rows_per_block + (total % rows_per_block != 0);

    for (std::size_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const std::size_t start_row = group * rows_per_block;
        // 这个 for 在 block 内一致执行；不能让部分线程提前退出归约。
        for (int local_row = 0; local_row < rows_per_block; ++local_row) {
            const std::size_t index = start_row + local_row;
            if (index >= total) break;  // 最后一个 block 可能不足四行。
            const std::size_t input_row = index / out;
            const std::size_t weight_row = index % out;
            float local_sum = 0.0f;

            // 与 FP32 版本相同：线程 tid 处理 tid、tid+128、... 列。
            // 每个权重用自己的 group scale，在寄存器中恢复一个 float，
            // 随即参与乘加；不分配 out*in 个 FP32 权重的中间 Tensor。
            for (std::size_t i = threadIdx.x; i < in; i += blockDim.x) {
                const std::size_t weight_idx = weight_row * in + i;
                const std::size_t group_idx = weight_idx / group_size;
                const float restored_weight = float(weight[weight_idx]) * scales[group_idx];
                local_sum += input[input_row * in + i] * restored_weight;
            }
            const float sum = BlockReduce(reduce_storage).Sum(local_sum);
            if (threadIdx.x == 0) output[index] = sum * output_scale;
            // 同一个 block 接下来可能处理下一行，必须等所有线程离开
            // 本轮归约，再覆盖共享 TempStorage。这也保护 grid-stride 复用。
            __syncthreads();
        }
    }
}

void launch(const Tensor& input, const QuantizedTensor& weight, Tensor& output,
            float output_scale, cudaStream_t stream, int rows_per_block) {
    detail::check_quantized_sgemv_tensors(input, weight, output, DeviceType::CUDA,
                                         output_scale);
    constexpr int threads = 128;
    const std::size_t total = output.size();
    const std::size_t groups = total / rows_per_block + (total % rows_per_block != 0);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(groups, 65535));
    quantized_sgemv_f32int8<<<blocks, threads, 0, stream>>>(
        input.ptr<float>(), weight.values.ptr<int8_t>(), weight.scales.ptr<float>(),
        weight.group_size, output.ptr<float>(), std::size_t(weight.values.dim(1)),
        std::size_t(weight.values.dim(0)), total, output_scale, rows_per_block);
    check_cuda(cudaGetLastError(), "Quantized Sgemv kernel launch");
}

}  // namespace

void quantized_sgemv_kernel_cuda(const Tensor& input, const QuantizedTensor& weight,
                                 Tensor& output, float output_scale, cudaStream_t stream) {
    launch(input, weight, output, output_scale, stream, 1);
}

void quantized_sgemv_kernel_cuda_rows4(const Tensor& input,
                                       const QuantizedTensor& weight, Tensor& output,
                                       float output_scale, cudaStream_t stream) {
    launch(input, weight, output, output_scale, stream, 4);
}

}  // namespace mini_infer
