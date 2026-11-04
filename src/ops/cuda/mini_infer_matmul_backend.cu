#include <cuda_runtime.h>
#include "mini_infer/ops/mini_infer_matmul_backend.h"
#include "mini_infer/ops/mini_infer_sgemv.h"

namespace mini_infer {
namespace {
// A tile of weights is reused by 16 prompt rows instead of reloading it per GEMV.
__global__ void gemm(const float* x, const float* w, float* y, int m, int n, int k, float scale) {
    __shared__ float a[16][16], b[16][16];
    const int row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0;
    for (int first = 0; first < k; first += 16) {
        a[threadIdx.y][threadIdx.x] = row < m && first + threadIdx.x < k ? x[std::size_t(row) * k + first + threadIdx.x] : 0;
        b[threadIdx.y][threadIdx.x] = col < n && first + threadIdx.y < k ? w[std::size_t(col) * k + first + threadIdx.y] : 0;
        __syncthreads();
        for (int i = 0; i < 16; ++i) sum += a[threadIdx.y][i] * b[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) y[std::size_t(row) * n + col] = sum * scale;
}
}
void matmul_kernel_cuda(const Tensor& input, const Tensor& weight, Tensor& output,
                        float scale, cudaStream_t stream) {
    const auto rows = input.size() / weight.dim(1);
    if (rows > 1) {
        gemm<<<dim3((weight.dim(0) + 15) / 16, (rows + 15) / 16), dim3(16,16), 0, stream>>>(
            input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(), rows, weight.dim(0), weight.dim(1), scale);
        check_cuda(cudaGetLastError(), "prefill GEMM");
        return;
    }
    // 旧 MatmulLayer/后端选择器继续可用，实际 CUDA 归约放在第 13 课。
    sgemv_kernel_cuda(input, weight, output, scale, stream);
}
}  // namespace mini_infer
