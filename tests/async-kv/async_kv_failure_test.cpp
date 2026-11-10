#include "infer/storage/async_kv_pipeline.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <thread>

namespace {
using infer::AsyncKVPipeline;
constexpr AsyncKVPipeline::Config config{3, 2, 128, 4, 16, 0};
enum class Fault { None, Create, Record, Wait, Copy };
Fault fault = Fault::None;
int calls = 0;

bool fail(Fault operation) {
    return fault == operation && --calls == 0;
}

template <class Poll> void poll_until(Poll poll) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!poll()) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("pipeline did not retire");
        std::this_thread::yield();
    }
}

void read(const infer::LayerRead& job) {
    std::memset(job.destination, 0, job.pages * config.page_bytes);
}
void write(std::size_t, std::size_t, const std::byte*, const std::byte*) {}
void compute(std::size_t, AsyncKVPipeline::LayerView view, cudaStream_t stream) {
    infer::check_cuda(cudaMemsetAsync(view.key, 0, config.row_bytes, stream), "test key");
    infer::check_cuda(cudaMemsetAsync(view.value, 0, config.row_bytes, stream), "test value");
}

class PipelineFailure : public testing::Test {
  protected:
    void SetUp() override {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
    void TearDown() override { fault = Fault::None; }
};

class ResidentFlushFailure : public PipelineFailure, public testing::WithParamInterface<Fault> {};

TEST_P(ResidentFlushFailure, RequiresRetirementAfterPartialSubmission) {
    auto resident = config;
    resident.gpu_resident = true;
    AsyncKVPipeline pipeline(resident, read, write);
    pipeline.begin_token(0);
    ASSERT_TRUE(pipeline.tick(compute));
    fault = GetParam();
    // Fail the final fence, after D2H work has already been accepted.
    calls = fault == Fault::Wait ? 1 : 2;
    EXPECT_THROW(pipeline.flush_resident(), std::exception);
    fault = Fault::None;
    EXPECT_THROW(pipeline.begin_token(1), std::logic_error);
    pipeline.begin_retire();
    poll_until([&] { return pipeline.poll_retired(); });
    pipeline.begin_token(0);
    EXPECT_TRUE(pipeline.tick(compute));
    EXPECT_NO_THROW(pipeline.flush_resident());
}
INSTANTIATE_TEST_SUITE_P(CUDA, ResidentFlushFailure,
                        testing::Values(Fault::Record, Fault::Wait, Fault::Copy));

TEST_F(PipelineFailure, FailedResidentCallbackRequiresRetirement) {
    auto resident = config;
    resident.gpu_resident = true;
    bool fail_write = true;
    AsyncKVPipeline pipeline(resident, read,
        [&](std::size_t, std::size_t, const std::byte*, const std::byte*) {
            if (fail_write) throw std::runtime_error("writeback failed");
        });
    pipeline.begin_token(0);
    ASSERT_TRUE(pipeline.tick(compute));
    EXPECT_THROW(pipeline.flush_resident(), std::runtime_error);
    EXPECT_THROW(pipeline.begin_token(1), std::logic_error);
    pipeline.begin_retire();
    poll_until([&] { return pipeline.poll_retired(); });
    fail_write = false;
    pipeline.begin_token(0);
    ASSERT_TRUE(pipeline.tick(compute));
    EXPECT_NO_THROW(pipeline.flush_resident());
}

TEST_F(PipelineFailure, PartialConstructionAndRetirementFenceCanBeRetried) {
    fault = Fault::Create;
    calls = 3;
    EXPECT_THROW(AsyncKVPipeline(config, read, write), std::exception);
    fault = Fault::None;
    AsyncKVPipeline pipeline(config, read, write);
    pipeline.begin_token(0);
    fault = Fault::Record;
    calls = 2;
    EXPECT_THROW(pipeline.begin_retire(), std::exception);
    fault = Fault::None;
    EXPECT_THROW(pipeline.begin_token(0), std::logic_error);
    pipeline.begin_retire();
    pipeline.begin_retire();
    poll_until([&] { return pipeline.poll_retired(); });
    pipeline.begin_token(0);
    poll_until([&] { return pipeline.tick(compute); });
}

TEST_F(PipelineFailure, DestructionDrainsWorkerBeforeDestroyingCallbacks) {
    std::promise<void> entered, release;
    const auto gate = release.get_future().share();
    auto owner = std::make_shared<std::atomic<unsigned>>(0);
    const std::weak_ptr<std::atomic<unsigned>> lifetime = owner;
    auto pipeline = std::make_unique<AsyncKVPipeline>(config,
        [owner, gate, &entered](const infer::LayerRead& job) {
            if (owner->fetch_add(1) == 0) entered.set_value();
            gate.wait();
            read(job);
        }, write);
    pipeline->begin_token(0);
    entered.get_future().wait();
    owner.reset();
    auto destroyed = std::async(std::launch::async, [p = std::move(pipeline)]() mutable { p.reset(); });
    EXPECT_FALSE(lifetime.expired());
    release.set_value();
    destroyed.get();
    EXPECT_TRUE(lifetime.expired());
}
} // namespace

extern "C" cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    if (fail(Fault::Create)) return cudaErrorMemoryAllocation;
    return __real_cudaEventCreateWithFlags(event, flags);
}
extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (fail(Fault::Record)) return cudaErrorUnknown;
    return __real_cudaEventRecord(event, stream);
}
extern "C" cudaError_t __real_cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
extern "C" cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    if (fail(Fault::Wait)) return cudaErrorUnknown;
    return __real_cudaStreamWaitEvent(stream, event, flags);
}
extern "C" cudaError_t __real_cudaMemcpy2DAsync(void*, size_t, const void*, size_t,
                                               size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                               size_t spitch, size_t width, size_t height,
                                               cudaMemcpyKind kind, cudaStream_t stream) {
    if (fail(Fault::Copy)) return cudaErrorUnknown;
    return __real_cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}
