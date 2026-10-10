#include "mini_infer/ops/rope/mini_infer_rope.h"

#include <cmath>
#include <stdexcept>

namespace mini_infer {
void rope_inplace(Tensor& query, Tensor& key, int32_t head_dim,
                  int32_t position, float theta, cudaStream_t stream) {
    const auto device = query.device_type();
    if (head_dim <= 0 || head_dim % 2 || position < 0 ||
        !std::isfinite(theta) || theta <= 0 ||
        (device != DeviceType::CPU && device != DeviceType::CUDA) ||
        (device == DeviceType::CPU && stream)) {
        throw std::invalid_argument("RoPE needs even head_dim, valid position/theta/device");
    }
    for (const Tensor* x : {&query, &key}) {
        if (x->empty() || x->data_type() != DataType::FLOAT32 ||
            x->device_type() != device || x->dims().size() != 1 ||
            x->dim(0) % head_dim != 0) {
            throw std::invalid_argument("RoPE expects FP32 Q/K vectors of whole heads");
        }
    }
    if (query.overlaps(key) || query.dim(0) % key.dim(0) != 0) {
        throw std::invalid_argument("RoPE Q/K must be independent with valid GQA widths");
    }
    if (device == DeviceType::CUDA) {
        detail::rope_cuda(query, key, head_dim, position, theta, stream);
        return;
    }
    // 先读出一对，再同时写回；不能写完 x[i] 后才用它算 x[i+1]。
    for (Tensor* tensor : {&query, &key}) {
        float* x = tensor->ptr<float>();
        for (std::size_t i = 0; i < tensor->size(); i += 2) {
            const float exponent = static_cast<float>(i % head_dim) / head_dim;
            const float angle = position / std::pow(theta, exponent);
            const float c = std::cos(angle), s = std::sin(angle);
            const float a = x[i], b = x[i + 1];
            x[i] = a * c - b * s;
            x[i + 1] = a * s + b * c;
        }
    }
}
}  // namespace mini_infer
