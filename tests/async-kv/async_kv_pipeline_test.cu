#include "infer/storage/async_kv_pipeline.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace {
using infer::AsyncKVPipeline;
using namespace std::chrono_literals;

constexpr std::size_t layers = 7;
constexpr std::size_t tokens = 4;
constexpr std::size_t width = 4;
constexpr std::size_t page_words = 2 * tokens * width;
constexpr std::size_t max_pages = 3;
constexpr AsyncKVPipeline::Config config{
    layers, max_pages, page_words * sizeof(float), tokens, width * sizeof(float), 0};

template <class Poll>
void poll_until(Poll poll) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!poll()) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("timed out advancing async KV");
        std::this_thread::yield();
    }
}

__global__ void update_rows(AsyncKVPipeline::LayerView view, std::size_t layer,
                            std::size_t position) {
    const auto col = threadIdx.x;
    float previous_key = 0;
    float previous_value = 0;
    if (position) {
        const auto previous = position - 1;
        const auto* page = view.pages[previous / tokens];
        previous_key = page[(previous % tokens) * width + col];
        previous_value = page[tokens * width + (previous % tokens) * width + col];
    }
    const auto* current = view.pages[position / tokens];
    const auto row = (position % tokens) * width;
    view.key[col] = current[row + col] + previous_key + float(layer + 1);
    view.value[col] = current[tokens * width + row + col] + previous_value - float(layer + 1);
}

class AsyncKVPipelineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA required";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};

TEST_F(AsyncKVPipelineTest, AlternatesTwoStableSlotsAndWritesEveryLayerAcrossPages) {
    std::array<std::array<float, page_words * max_pages>, layers> canonical;
    for (std::size_t layer = 0; layer < layers; ++layer)
        canonical[layer].fill(float(10 * (layer + 1)));

    AsyncKVPipeline pipeline(
        config,
        [&](const infer::LayerRead& job) {
            std::memcpy(job.destination, canonical[job.layer].data(), job.pages * config.page_bytes);
        },
        [&](std::size_t layer, std::size_t position, const std::byte* key, const std::byte* value) {
            auto* page = canonical[layer].data() + position / tokens * page_words;
            const auto row = position % tokens * width;
            std::memcpy(page + row, key, config.row_bytes);
            std::memcpy(page + tokens * width + row, value, config.row_bytes);
        });

    std::array<const float* const*, 2> tables{};
    for (std::size_t position = 0; position < 10; ++position) {
        std::size_t next_layer = 0;
        const AsyncKVPipeline::Compute compute =
            [&](std::size_t layer, AsyncKVPipeline::LayerView view, cudaStream_t stream) {
                EXPECT_EQ(layer, next_layer++);
                auto& expected_table = tables[layer % 2];
                if (!expected_table) expected_table = view.pages;
                EXPECT_EQ(view.pages, expected_table);
                update_rows<<<1, width, 0, stream>>>(view, layer, position);
                infer::check_cuda(cudaGetLastError(), "launch KV regression kernel");
            };
        pipeline.begin_token(position);
        poll_until([&] { return pipeline.tick(compute); });
        ASSERT_EQ(next_layer, layers);
        ASSERT_NE(tables[0], tables[1]);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto* page = canonical[layer].data() + position / tokens * page_words;
            const auto row = position % tokens * width;
            for (std::size_t col = 0; col < width; ++col) {
                EXPECT_EQ(page[row + col], float((position + 1) * (layer + 1) * 11));
                EXPECT_EQ(page[tokens * width + row + col], float((position + 1) * (layer + 1) * 9));
            }
        }
    }
}

TEST_F(AsyncKVPipelineTest, RetiresAfterPartialComputeSubmissionAndRestarts) {
    AsyncKVPipeline pipeline(config, [](const infer::LayerRead& job) {
        std::memset(job.destination, 0, job.pages * config.page_bytes);
    }, [](std::size_t, std::size_t, const std::byte*, const std::byte*) {});

    pipeline.begin_token(0);
    const AsyncKVPipeline::Compute fail = [](std::size_t, AsyncKVPipeline::LayerView view,
                                             cudaStream_t stream) {
        infer::check_cuda(cudaMemsetAsync(view.key, 0, config.row_bytes, stream), "queue partial layer");
        throw std::runtime_error("compute submission failed");
    };
    EXPECT_THROW(poll_until([&] { return pipeline.tick(fail); }), std::runtime_error);
    EXPECT_THROW(pipeline.tick(fail), std::logic_error);
    pipeline.begin_retire();
    poll_until([&] { return pipeline.poll_retired(); });

    pipeline.begin_token(0);
    const AsyncKVPipeline::Compute compute =
        [](std::size_t layer, AsyncKVPipeline::LayerView view, cudaStream_t stream) {
            update_rows<<<1, width, 0, stream>>>(view, layer, 0);
            infer::check_cuda(cudaGetLastError(), "launch after retirement");
        };
    EXPECT_NO_THROW(poll_until([&] { return pipeline.tick(compute); }));
}
TEST_F(AsyncKVPipelineTest, FailedReadDoesNotReachComputeAndRetiresPendingIO) {
    AsyncKVPipeline pipeline(config, [](const infer::LayerRead& job) {
        if (job.layer == 1) throw std::domain_error("KV storage read failed");
        std::memset(job.destination, 0, job.pages * config.page_bytes);
    }, [](std::size_t, std::size_t, const std::byte*, const std::byte*) {});

    pipeline.begin_token(0);
    const AsyncKVPipeline::Compute compute =
        [](std::size_t layer, AsyncKVPipeline::LayerView, cudaStream_t) {
            EXPECT_EQ(layer, 0u);
        };
    EXPECT_THROW(poll_until([&] { return pipeline.tick(compute); }), std::domain_error);
    pipeline.begin_retire();
    EXPECT_NO_THROW(poll_until([&] { return pipeline.poll_retired(); }));
}
TEST_F(AsyncKVPipelineTest, ResidentDownloadAndRetirementPreserveAdmissionBoundaries) {
    auto resident = config;
    resident.gpu_resident = true;
    std::size_t reads = 0, writes = 0;
    AsyncKVPipeline pipeline(resident,
        [&](const infer::LayerRead&) { ++reads; },
        [&](std::size_t, std::size_t, const std::byte* key, const std::byte* value) {
            for (std::size_t i = 0; i < config.row_bytes; ++i) {
                EXPECT_EQ(key[i], std::byte{0});
                EXPECT_EQ(value[i], std::byte{0});
            }
            ++writes;
        });
    const AsyncKVPipeline::Compute compute =
        [&](std::size_t, AsyncKVPipeline::LayerView view, cudaStream_t stream) {
            infer::check_cuda(cudaMemsetAsync(view.key, 0, config.row_bytes, stream), "clear key");
            infer::check_cuda(cudaMemsetAsync(view.value, 0, config.row_bytes, stream), "clear value");
        };
    pipeline.begin_token(0);
    EXPECT_THROW(pipeline.begin_token(0), std::logic_error);
    ASSERT_TRUE(pipeline.tick(compute));
    EXPECT_EQ(reads, 0u);
    pipeline.flush_resident();
    EXPECT_EQ(writes, layers);
    pipeline.flush_resident();
    EXPECT_EQ(writes, layers);

    void* output = nullptr;
    ASSERT_EQ(cudaMallocHost(&output, config.row_bytes), cudaSuccess);
    const auto release = [](void* pointer) { cudaFreeHost(pointer); };
    std::unique_ptr<void, decltype(release)> pinned(output, release);
    pipeline.download(output, pipeline.graph_slot(0).key, config.row_bytes);
    EXPECT_THROW(pipeline.begin_token(1), std::logic_error);
    poll_until([&] { return pipeline.poll_download(); });
    EXPECT_TRUE(pipeline.poll_download());
    pipeline.begin_retire();
    pipeline.begin_retire();
    EXPECT_THROW(pipeline.begin_token(1), std::logic_error);
    poll_until([&] { return pipeline.poll_retired(); });
    EXPECT_THROW(pipeline.poll_retired(), std::logic_error);
    pipeline.begin_token(0);
    EXPECT_TRUE(pipeline.tick(compute));
}

} // namespace
