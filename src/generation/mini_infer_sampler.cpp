#include "mini_infer/generation/mini_infer_sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mini_infer {
Sampler::Sampler(float temperature, int32_t top_k, uint32_t seed)
    : temperature_(temperature), top_k_(top_k), random_(seed) {
    if (!std::isfinite(temperature) || temperature < 0 || top_k < 0)
        throw std::invalid_argument("sampling needs finite temperature >= 0 and top_k >= 0");
}

int32_t Sampler::sample(const Tensor& logits) {
    if (logits.empty() || logits.device_type() != DeviceType::CPU ||
        logits.data_type() != DataType::FLOAT32 || logits.dims().size() != 1)
        throw std::invalid_argument("sampler expects CPU FP32 logits=[vocab_size]");
    const float* values = logits.ptr<float>();
    const int32_t count = logits.dim(0);
    for (int32_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) throw std::runtime_error("model produced non-finite logits");
    const auto best = static_cast<int32_t>(std::max_element(values, values + count) - values);
    if (temperature_ == 0 || top_k_ == 1) return best;

    std::vector<int32_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    if (top_k_ > 0 && top_k_ < count) {
        std::partial_sort(ids.begin(), ids.begin() + top_k_, ids.end(), [&](int32_t a, int32_t b) {
            return values[a] == values[b] ? a < b : values[a] > values[b];
        });
        ids.resize(top_k_);
    }
    // 先减最大值再除温度，使用 double 避免极小温度导致 float 溢出。
    std::vector<double> probabilities;
    double sum = 0;
    for (int32_t id : ids) {
        const double p = std::exp((static_cast<double>(values[id]) - values[best]) / temperature_);
        probabilities.push_back(p);
        sum += p;
    }
    const double target = std::uniform_real_distribution<double>(0.0, sum)(random_);
    double cumulative = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        cumulative += probabilities[i];
        if (target < cumulative) return ids[i];
    }
    return ids.back(); // 浮点累计的尾部保护
}
}  // namespace mini_infer
