#include "test_support.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/cache/mini_infer_kv_cache.h"
#include "mini_infer/ops/matmul/mini_infer_sgemv.h"
#include "kv_test_support.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 只供拓展测试使用的单头标量 attention；不属于 KVCache 实现。
std::vector<double> attention(const Tensor& query, const Tensor& keys, const Tensor& values) {
    const int32_t tokens = keys.dim(0), dim = keys.dim(1);
    std::vector<double> scores(static_cast<std::size_t>(tokens));
    for (int32_t t = 0; t < tokens; ++t) {
        for (int32_t d = 0; d < dim; ++d)
            scores[t] += static_cast<double>(query.index<float>(d)) *
                         keys.index<float>(static_cast<std::size_t>(t) * dim + d);
        scores[t] /= std::sqrt(static_cast<double>(dim));
    }
    const double max_score = *std::max_element(scores.begin(), scores.end());
    double denominator = 0;
    for (double& score : scores) { score = std::exp(score - max_score); denominator += score; }
    std::vector<double> output(static_cast<std::size_t>(dim));
    for (int32_t t = 0; t < tokens; ++t)
        for (int32_t d = 0; d < dim; ++d)
            output[d] += scores[t] / denominator *
                         values.index<float>(static_cast<std::size_t>(t) * dim + d);
    return output;
}

// 当前 token 只投影一次，和每步重算整个前缀的数学结果比较。
void verify_projection_reuse(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    constexpr int32_t steps = 4, dim = 3;
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    KVCache cache(1, steps, dim, allocator);
    Tensor tokens = input({steps, dim}, -3);
    Tensor wk = input({dim, dim}, -4), wv = input({dim, dim}, -2);
    // 缩小权重以使示例 softmax 分数温和，便于观察各历史 token 的贡献。
    for (std::size_t i = 0; i < wk.size(); ++i) {
        wk.index<float>(i) *= 0.05f; wv.index<float>(i) *= 0.1f;
    }
    Tensor dtokens = on_device(tokens, allocator);
    Tensor dwk = on_device(wk, allocator), dwv = on_device(wv, allocator);
    int cached_projection_rows = 0, recomputed_projection_rows = 0;
    double max_error = 0;
    for (int32_t pos = 0; pos < steps; ++pos) {
        Tensor current = dtokens.view(static_cast<std::size_t>(pos) * dim, {dim});
        std::pair<Tensor, Tensor> slot = cache.slice_kv_cache(0, pos);
        // 复用第 13 课 Sgemv，把当前 K/V 直接写到缓存，没有临时拼接。
        get_sgemv_kernel(allocator->device_type())(current, dwk, slot.first, 1, stream);
        get_sgemv_kernel(allocator->device_type())(current, dwv, slot.second, 1, stream);
        cache.commit(0, pos);
        wait(allocator->device_type(), stream);
        cached_projection_rows += 2;

        // 独立对照：每一步重算所有已有 token 的 K/V，并使用标量参考。
        Tensor prefix = tokens.view(0, {pos + 1, dim});
        Tensor reference_k(DataType::FLOAT32, {pos + 1, dim}, cpu);
        Tensor reference_v(DataType::FLOAT32, {pos + 1, dim}, cpu);
        sgemv_kernel_cpu_reference(prefix, wk, reference_k, 1, nullptr);
        sgemv_kernel_cpu_reference(prefix, wv, reference_v, 1, nullptr);
        recomputed_projection_rows += 2 * (pos + 1);
        std::pair<Tensor, Tensor> history = cache.history(0);
        Tensor cached_k = history.first.to_cpu(), cached_v = history.second.to_cpu();
        Tensor query = tokens.view(static_cast<std::size_t>(pos) * dim, {dim});
        std::vector<double> result = attention(query, cached_k, cached_v);
        std::vector<double> reference = attention(query, reference_k, reference_v);
        for (std::size_t i = 0; i < result.size(); ++i) {
            const double error = std::abs(result[i] - reference[i]);
            require(std::isfinite(result[i]) && error < 1e-5, "cached attention differs");
            max_error = std::max(max_error, error);
        }
    }
    require(cached_projection_rows == 8 && recomputed_projection_rows == 20,
            "projection reuse accounting incorrect");
    std::cout << "[" << (allocator->device_type() == DeviceType::CPU ? "CPU" : "CUDA")
              << "] four steps: cached K/V rows=" << cached_projection_rows
              << ", recomputed rows=" << recomputed_projection_rows
              << ", attention max error=" << max_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        verify_projection_reuse(std::make_shared<CPUDeviceAllocator>(), nullptr);
        if (use_cuda) {
            TestStream stream;
            verify_projection_reuse(std::make_shared<CUDADeviceAllocator>(), stream.get());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
