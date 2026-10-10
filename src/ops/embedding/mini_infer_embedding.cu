#include "mini_infer/ops/embedding/mini_infer_embedding.h"

#include <algorithm>

namespace mini_infer {
namespace {
constexpr unsigned int threads = 128;

// 一个 block 搬运一个 token 的整行向量，各线程分担 dim 个元素。
// token_idx 是输入序列的位置 0,1,2；token_ids[token_idx] 才是词表 id。
__global__ void embedding_f32(const int32_t* ids, const float* weight, float* output,
                              std::size_t token_num, int32_t vocab_size, int32_t dim) {
    for (std::size_t t = blockIdx.x; t < token_num; t += gridDim.x) {
        const int32_t token = ids[t];
        // 外层已检查 id。这里保留界限保护，避免设备读取越界地址。
        if (token < 0 || token >= vocab_size) continue;
        const float* row = weight + std::size_t(token) * dim;
        float* result = output + t * dim;
        for (std::size_t i = threadIdx.x; i < std::size_t(dim); i += blockDim.x)
            result[i] = row[i];
    }
}
}  // namespace

void embedding_kernel_cuda(const Tensor& ids, const Tensor& weight,
                           Tensor& output, cudaStream_t stream) {
    detail::check_embedding_tensors(ids, weight, output, DeviceType::CUDA);
    detail::check_embedding_ids(ids, weight.dim(0), stream);
    const unsigned int blocks = static_cast<unsigned int>(std::min<std::size_t>(ids.size(), 65535));
    embedding_f32<<<blocks, threads, 0, stream>>>(ids.ptr<int32_t>(), weight.ptr<float>(),
        output.ptr<float>(), ids.size(), weight.dim(0), weight.dim(1));
    check_cuda(cudaGetLastError(), "Embedding kernel launch");
}
}  // namespace mini_infer
