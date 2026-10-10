#include "mini_infer/ops/matmul/mini_infer_sgemv.h"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
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
