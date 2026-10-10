#include "mini_infer/ops/add/mini_infer_add.h"
#include "mini_infer/backend/mini_infer_kernels.h"

#include <algorithm>

namespace mini_infer {
namespace {
constexpr unsigned int threads = 32;  // 课件的 1024 个元素 -> 32 块，每块 32 线程。

__global__ void add_f32(const float* a, const float* b, float* output, std::size_t size) {
    // 课件 tid = threadIdx.x + blockDim.x * blockIdx.x。
    // block 0 负责 0～31，block 1 负责 32～63；所有线程并行执行。
    // 网格达到预设上限时，每线程按 stride 继续处理下一批元素。
    const std::size_t stride = std::size_t(blockDim.x) * gridDim.x;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        output[i] = a[i] + b[i];
    }
    // 尾部 i>=size 的线程不访问内存；各元素独立，不需要共享内存/同步。
}
void add_cuda(const Tensor& a, const Tensor& b, Tensor& output,
              cudaStream_t stream, bool allow_inplace) {
    detail::check_elementwise_add_tensors(a, b, output, DeviceType::CUDA, allow_inplace);
    // 已拒绝空 Tensor。用 size-1 计算向上取整，避免 size+threads-1 溢出。
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(1 + (a.size() - 1) / threads, 65535));
    add_f32<<<blocks, threads, 0, stream>>>(a.ptr<float>(), b.ptr<float>(),
                                          output.ptr<float>(), a.size());
    check_cuda(cudaGetLastError(), "Elementwise Add kernel launch");
}
}  // namespace

void elementwise_add_cuda(const Tensor& a, const Tensor& b, Tensor& output, cudaStream_t stream) {
    add_cuda(a, b, output, stream, false);
}
void add_kernel_cuda(const Tensor& a, const Tensor& b, Tensor& output, cudaStream_t stream) {
    add_cuda(a, b, output, stream, true);
}
}  // namespace mini_infer
