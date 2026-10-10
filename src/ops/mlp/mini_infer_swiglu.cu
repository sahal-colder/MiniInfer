#include "mini_infer/ops/mlp/mini_infer_swiglu.h"

#include <algorithm>
#include <cmath>

namespace mini_infer {
namespace {
constexpr unsigned int threads = 128;  // 与课件一致。

__device__ float swiglu_value(float gate, float up) {
    const float e = expf(-fabsf(gate));
    const float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return (gate * sigmoid) * up;
}

// 主线：没有跨元素通信，无归约，一个线程独立完成一对元素。
// 不使用 __restrict__：接口明确允许 output 与输入完全重合。
__global__ void swiglu_f32(const float* gate, const float* up, float* output,
                            std::size_t size) {
    const std::size_t stride = std::size_t(blockDim.x) * gridDim.x;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const float g = gate[i];
        const float u = up[i];
        output[i] = swiglu_value(g, u);
    }
}

// 课件对照：一块动态共享内存切成两段，各容纳 blockDim.x 个 float。
// 实际不需要共享内存，但保留此版本帮助理解 tid / idx / smem 的对应。
__global__ void swiglu_shared_f32(const float* gate, const float* up, float* output,
                                 std::size_t size) {
    extern __shared__ float shared[];
    float* smem_gate = shared;
    float* smem_up = shared + blockDim.x;
    const unsigned int tid = threadIdx.x;
    const std::size_t stride = std::size_t(blockDim.x) * gridDim.x;
    // base 的循环条件对整个 block 相同，尾部线程也参加所有 barrier。
    // 不照抄“越界线程提前 return，然后其他线程 __syncthreads”的写法。
    for (std::size_t base = std::size_t(blockIdx.x) * blockDim.x;
         base < size; base += stride) {
        const std::size_t i = base + tid;
        smem_gate[tid] = i < size ? gate[i] : 0.0f;
        smem_up[tid] = i < size ? up[i] : 0.0f;
        __syncthreads();
        if (i < size) output[i] = swiglu_value(smem_gate[tid], smem_up[tid]);
        // 下一轮复用 shared 前完成本轮读取，保持清楚的同步边界。
        __syncthreads();
    }
}

unsigned int block_count(std::size_t size) {
    // 用 size-1 再除避免 size+threads-1 溢出；外层已拒绝空 Tensor。
    return static_cast<unsigned int>(
        std::min<std::size_t>(1 + (size - 1) / threads, 65535));
}
}  // namespace

void swiglu_kernel_cuda(const Tensor& gate, const Tensor& up, Tensor& output,
                        cudaStream_t stream) {
    detail::check_swiglu_tensors(gate, up, output, DeviceType::CUDA);
    swiglu_f32<<<block_count(gate.size()), threads, 0, stream>>>(
        gate.ptr<float>(), up.ptr<float>(), output.ptr<float>(), gate.size());
    check_cuda(cudaGetLastError(), "SwiGLU kernel launch");
}

void swiglu_kernel_cuda_shared(const Tensor& gate, const Tensor& up, Tensor& output,
                               cudaStream_t stream) {
    detail::check_swiglu_tensors(gate, up, output, DeviceType::CUDA);
    const std::size_t shared_bytes = 2 * threads * sizeof(float);
    swiglu_shared_f32<<<block_count(gate.size()), threads, shared_bytes, stream>>>(
        gate.ptr<float>(), up.ptr<float>(), output.ptr<float>(), gate.size());
    check_cuda(cudaGetLastError(), "SwiGLU shared kernel launch");
}
}  // namespace mini_infer
