#include "mini_infer/loader/mini_infer_mlp_weight_loader.h"
#include "mini_infer/model/mini_infer_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
class CUDAGraphModelTest : public ::testing::Test {
  protected:
    std::filesystem::path directory;
    const mini_infer::ModelConfig config{32, 64, 5, 4, 2, 48, 40};

    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP() << "CUDA required";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        char path[] = "/tmp/infer-graph-test-XXXXXX";
        ASSERT_NE(mkdtemp(path), nullptr);
        directory = path;
        std::ofstream file(directory / "model.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(&config), sizeof(config));
        const auto weights = mini_infer::make_mlp_weight_layout(config).end +
                             config.dim + config.seq_len * (config.dim / config.head_num);
        for (std::size_t i = 0; i < weights; ++i) {
            const float value = float(int(i % 37) - 18) / 100;
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        ASSERT_TRUE(file.good());
    }
    void TearDown() override {
        std::error_code error;
        if (!directory.empty()) std::filesystem::remove_all(directory, error);
    }
    mini_infer::ModelOptions options() const {
        mini_infer::ModelOptions options;
        options.kv_mode = mini_infer::ModelKVMode::PAGED;
        options.paged.enable_async_kv_pipeline = true;
        options.paged.tokens_per_page = 4;
        options.paged.ssd_directory = directory.string();
        options.paged.ssd_direct = false;
        options.paged.ssd_io.mode = infer::FileIOMode::POSIX;
        options.paged.host_bytes = config.layer_num * options.paged.page_bytes;
        return options;
    }
};

TEST_F(CUDAGraphModelTest, MatchesEagerAcrossPagesTokensResetAndPrefixReuse) {
    auto allocator = std::make_shared<mini_infer::CUDADeviceAllocator>();
    auto settings = options();
    mini_infer::LlamaModel eager((directory / "model.bin").string(), allocator, settings);
    settings.enable_cuda_graph = true;
    mini_infer::LlamaModel graph((directory / "model.bin").string(), allocator, settings);
    const auto initial = graph.cuda_graph_stats();
    EXPECT_EQ(initial.graphs, 2u * config.layer_num);
    EXPECT_GT(initial.workspace_bytes, 0u);
    const std::vector<int32_t> prompt{2, 4, 0, 8, 10, 12, 6, 9, 1};
    float max_error = 0;
    const auto compare = [&](const mini_infer::Tensor& expected, const mini_infer::Tensor& actual) {
        for (std::size_t i = 0; i < expected.size(); ++i) {
            ASSERT_TRUE(std::isfinite(actual.ptr<float>()[i]));
            max_error = std::max(max_error, std::abs(actual.ptr<float>()[i] - expected.ptr<float>()[i]));
            EXPECT_NEAR(actual.ptr<float>()[i], expected.ptr<float>()[i], 1e-5f);
        }
    };
    for (int request = 0; request < 2; ++request) {
        compare(eager.prefill(prompt), graph.prefill(prompt));
        if (request) EXPECT_EQ(graph.paged_stats().last_prefix_hit_tokens, 8u);
        for (int32_t position = prompt.size(); position < config.seq_len; ++position) {
            const auto token = (position * 7 + request) % config.vocab_size;
            compare(eager.forward(token), graph.forward(token));
        }
        EXPECT_THROW(graph.forward(0), std::out_of_range);
        EXPECT_GT(graph.paged_stats().ssd_read_bytes, 0u);
        EXPECT_EQ(graph.cuda_graph_stats().workspace_bytes, initial.workspace_bytes);
        EXPECT_EQ(graph.cuda_graph_stats().graphs, initial.graphs);
        eager.reset();
        graph.reset();
    }
    EXPECT_GT(graph.cuda_graph_stats().replays, 0u);
    std::cout << "[CUDA Graph precision] max_abs_logit_error=" << max_error
              << " graphs=" << initial.graphs << " shared_workspace_bytes=" << initial.workspace_bytes << '\n';
}

TEST_F(CUDAGraphModelTest, ReportsPairedEndToEndDecodeTPOT) {
    auto allocator = std::make_shared<mini_infer::CUDADeviceAllocator>();
    auto settings = options();
    settings.paged.prefix_cache = false;
    mini_infer::LlamaModel eager((directory / "model.bin").string(), allocator, settings);
    settings.enable_cuda_graph = true;
    mini_infer::LlamaModel graph((directory / "model.bin").string(), allocator, settings);
    const auto measure = [](mini_infer::LlamaModel& model) {
        model.reset();
        model.prefill({2, 4, 0, 8});
        const auto start = std::chrono::steady_clock::now();
        for (int32_t i = 0; i < 24; ++i) model.forward((i * 7) % model.config().vocab_size);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 24;
    };
    measure(eager);
    measure(graph);
    std::vector<double> eager_us, graph_us;
    for (int pair = 0; pair < 5; ++pair) {
        if (pair % 2 == 0) {
            eager_us.push_back(measure(eager));
            graph_us.push_back(measure(graph));
        } else {
            graph_us.push_back(measure(graph));
            eager_us.push_back(measure(eager));
        }
        std::cout << "[CUDA Graph TPOT pair " << pair + 1 << "] eager_us=" << eager_us.back()
                  << " graph_us=" << graph_us.back() << '\n';
    }
    std::sort(eager_us.begin(), eager_us.end());
    std::sort(graph_us.begin(), graph_us.end());
    std::cout << "[CUDA Graph TPOT median, 5 pairs, 24 tokens/pair, includes KV I/O and logits D2H] eager_us="
              << eager_us[2] << " graph_us=" << graph_us[2] << " speedup=" << eager_us[2] / graph_us[2] << '\n';
    EXPECT_EQ(graph.cuda_graph_stats().graphs, 2u * config.layer_num);
    EXPECT_GT(graph.cuda_graph_stats().replays, 0u);
}

TEST_F(CUDAGraphModelTest, RejectsGraphWithoutFixedSlotPipeline) {
    auto settings = options();
    settings.paged.enable_async_kv_pipeline = false;
    settings.enable_cuda_graph = true;
    EXPECT_THROW(mini_infer::LlamaModel((directory / "model.bin").string(),
        std::make_shared<mini_infer::CUDADeviceAllocator>(), settings), std::invalid_argument);
}
} // namespace
