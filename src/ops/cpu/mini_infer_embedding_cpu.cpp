#include "mini_infer/ops/mini_infer_embedding.h"

#include <algorithm>

namespace mini_infer {
void embedding_kernel_cpu(const Tensor& ids, const Tensor& weight,
                          Tensor& output, cudaStream_t stream) {
    detail::check_embedding_tensors(ids, weight, output, DeviceType::CPU);
    if (stream) throw std::invalid_argument("CPU Embedding does not accept a CUDA stream");
    detail::check_embedding_ids(ids, weight.dim(0), stream);
    const int32_t dim = weight.dim(1);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const std::size_t token = static_cast<std::size_t>(ids.ptr<int32_t>()[t]);
        std::copy_n(weight.ptr<float>() + token * dim, dim, output.ptr<float>() + t * dim);
    }
}

}  // namespace mini_infer
