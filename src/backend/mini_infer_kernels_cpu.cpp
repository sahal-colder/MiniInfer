#include "mini_infer/backend/mini_infer_kernels.h"
#include <cmath>

namespace mini_infer {

void rmsnorm_kernel_cpu(const Tensor& x, const Tensor& w, Tensor& y, float eps, cudaStream_t) {
    // EXERCISE_BEGIN: rmsnorm_cpu
    const auto* xp = x.ptr<float>();
    const auto* wp = w.ptr<float>();
    auto* yp = y.ptr<float>();
    const std::size_t hidden = std::size_t(x.dims().back());
    const std::size_t rows = x.size() / hidden;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto offset = row * hidden;
        float sum = 0.0f;
        for (std::size_t j = 0; j < hidden; ++j) {
            const float v = xp[offset + j];
            sum += v * v;
        }
        const float scale = 1.0f / std::sqrt(sum / float(hidden) + eps);
        for (std::size_t j = 0; j < hidden; ++j)
            yp[offset + j] = xp[offset + j] * scale * wp[j];
    }
    // EXERCISE_END: rmsnorm_cpu
}

}  // namespace mini_infer
