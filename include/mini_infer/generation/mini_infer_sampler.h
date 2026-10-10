#pragma once

#include "mini_infer/core/mini_infer_tensor.h"
#include <random>

namespace mini_infer {
// logits 已由模型读回 CPU，采样只处理一个词表向量。
// temperature=0：贪心；>0：temperature softmax 后随机采样。
// top_k=0：保留全词表；>0：仅在得分最高的 k 个 token 中采样。
class Sampler {
public:
    Sampler(float temperature = 0.0f, int32_t top_k = 0, uint32_t seed = 42);
    int32_t sample(const Tensor& logits);
private:
    float temperature_;
    int32_t top_k_;
    std::mt19937 random_;
};
}  // namespace mini_infer
