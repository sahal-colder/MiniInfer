#include "mini_infer/loader/mini_infer_mlp_weight_loader.h"
#include "mini_infer/model/mini_infer_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
struct TempDirectory {
    std::filesystem::path path;
    TempDirectory() {
        char name[] = "/tmp/infer-async-kv-test-XXXXXX";
        if (!mkdtemp(name)) throw std::runtime_error("create KV test directory");
        path = name;
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::vector<float> logits(const mini_infer::Tensor& tensor) {
    return {tensor.ptr<float>(), tensor.ptr<float>() + tensor.size()};
}

TEST(AsyncKVModelTest, MatchesSynchronousPagedLogitsWithSpillsAndPrefixReuse) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP() << "CUDA required";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    TempDirectory directory;
    const auto model_path = directory.path / "model.bin";
    const mini_infer::ModelConfig config{8, 16, 5, 2, 1, 16, 12};
    {
        std::ofstream file(model_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&config), sizeof(config));
        const auto count = mini_infer::make_mlp_weight_layout(config).end +
                           config.dim + config.seq_len * (config.dim / config.head_num);
        for (std::size_t i = 0; i < count; ++i) {
            const float value = float(int(i % 37) - 18) / 100;
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        ASSERT_TRUE(file.good());
    }

    auto allocator = std::make_shared<mini_infer::CUDADeviceAllocator>();
    mini_infer::ModelOptions options;
    options.kv_mode = mini_infer::ModelKVMode::PAGED;
    options.paged.ssd_directory = directory.path.string();
    options.paged.ssd_direct = false;
    options.paged.ssd_io.mode = infer::FileIOMode::POSIX;
    options.paged.tokens_per_page = 4;
    options.paged.host_bytes = config.layer_num * options.paged.page_bytes;
    mini_infer::LlamaModel synchronous(model_path.string(), allocator, options);
    options.paged.enable_async_kv_pipeline = true;
    mini_infer::LlamaModel asynchronous(model_path.string(), allocator, options);

    const std::vector<int32_t> prompt{2, 4, 0, 8, 10, 12, 6, 9, 1};
    const auto expected_prefill = logits(synchronous.prefill(prompt));
    const auto expected_decode = logits(synchronous.forward(3));
    EXPECT_EQ(logits(asynchronous.prefill(prompt)), expected_prefill);
    EXPECT_EQ(logits(asynchronous.forward(3)), expected_decode);
    const auto miss = asynchronous.paged_stats();
    EXPECT_GT(miss.ssd_spilled_pages, 0u);
    EXPECT_GT(miss.ssd_read_bytes, 0u);
    EXPECT_LE(miss.gpu_reserved_bytes, miss.gpu_limit_bytes);

    asynchronous.reset();
    // Prefix hits may leave one row, selecting GEMV instead of the miss GEMM.
    // Require bounded FP32 error and exactly the same greedy output token.
    const auto compare = [](const auto& expected,const auto& actual) {
        ASSERT_EQ(expected.size(),actual.size());
        float maximum=0;
        for (std::size_t i=0;i<expected.size();++i) {
            ASSERT_TRUE(std::isfinite(actual[i]));
            maximum=std::max(maximum,std::abs(expected[i]-actual[i]));
            EXPECT_NEAR(expected[i],actual[i],1e-6f);
        }
        EXPECT_EQ(std::max_element(expected.begin(),expected.end())-expected.begin(),
                  std::max_element(actual.begin(),actual.end())-actual.begin());
        std::cout << "[prefix GEMM/GEMV max_abs_logit_error] " << maximum << '\n';
    };
    compare(expected_prefill,logits(asynchronous.prefill(prompt)));
    EXPECT_EQ(asynchronous.paged_stats().last_prefix_hit_tokens, 8u);
    compare(expected_decode,logits(asynchronous.forward(3)));

    // Batched projections/KV scatter must agree with independent sequential
    // decode, including appending a batch after an existing partial page.
    options.kv_mode=mini_infer::ModelKVMode::CONTIGUOUS;
    options.paged.enable_async_kv_pipeline=false;
    mini_infer::LlamaModel sequential(model_path.string(),allocator,options);
    for (const auto token:prompt) sequential.forward(token);
    const auto expected=logits(sequential.forward(3));
    options.kv_mode=mini_infer::ModelKVMode::PAGED;
    options.paged.enable_async_kv_pipeline=true;
    options.paged.resident_layers=0;
    options.paged.host_bytes=20*options.paged.page_bytes;
    mini_infer::LlamaModel resident(model_path.string(),allocator,options);
    resident.prefill({2,4,0});
    resident.prefill({8,10,12,6,9,1});
    compare(expected,logits(resident.forward(3)));
    const auto before=resident.paged_stats();
    resident.forward(2);
    const auto after=resident.paged_stats();
    EXPECT_EQ(after.host_read_bytes,before.host_read_bytes);
    EXPECT_EQ(after.host_written_bytes,before.host_written_bytes);
    resident.reset();
    compare(expected_prefill,logits(resident.prefill(prompt)));

    // The raw all-layer size fits this budget, but rounded resident slots do
    // not. Two rounded streaming slots fit and must remain usable.
    options.paged.gpu_bytes=5*(3*65536+3*sizeof(void*));
    mini_infer::LlamaModel bounded(model_path.string(),allocator,options);
    compare(expected_prefill,logits(bounded.prefill(prompt)));
    EXPECT_LE(bounded.paged_stats().gpu_reserved_bytes,options.paged.gpu_bytes);

    options.kv_mode=mini_infer::ModelKVMode::CONTIGUOUS;
    options.paged.enable_async_kv_pipeline=false;
    auto cpu=std::make_shared<mini_infer::CPUDeviceAllocator>();
    mini_infer::LlamaModel cpu_batch(model_path.string(),cpu,options);
    mini_infer::LlamaModel cpu_steps(model_path.string(),cpu,options);
    for (const auto token:prompt) cpu_steps.forward(token);
    compare(logits(cpu_steps.forward(3)),[&] {
        cpu_batch.prefill(prompt);
        return logits(cpu_batch.forward(3));
    }());
}
} // namespace
