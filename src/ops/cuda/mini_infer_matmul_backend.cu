#include <cuda_runtime.h>
#include "mini_infer/ops/mini_infer_matmul_backend.h"
#include "mini_infer/ops/mini_infer_sgemv.h"

namespace mini_infer {
void matmul_kernel_cuda(const Tensor& input, const Tensor& weight, Tensor& output,
                        float scale, cudaStream_t stream) {
    // 旧 MatmulLayer/后端选择器继续可用，实际 CUDA 归约放在第 13 课。
    sgemv_kernel_cuda(input, weight, output, scale, stream);
}
}  // namespace mini_infer
