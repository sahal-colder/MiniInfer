#include "infer/cuda/cuda_memory_pool.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <thread>

namespace {
__global__ void fill(float* data, float value) { data[threadIdx.x] = value; }

class CUDAGraphPoolTest : public ::testing::Test {
  protected:
    cudaStream_t stream = nullptr;
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP() << "CUDA required";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    }
    void TearDown() override { if (stream) cudaStreamDestroy(stream); }
};

TEST_F(CUDAGraphPoolTest, SharesPrivateAddressesAcrossGraphsAndSurvivesEagerTrim) {
    infer::CachingCUDAAllocatorConfig config;
    config.max_cached_bytes = 0;
    infer::CachingCUDADeviceAllocator pool(config);
    pool.reserve_graph_workspace(4096);
    std::array<cudaGraph_t, 2> graphs{};
    std::array<cudaGraphExec_t, 2> execs{};
    std::array<float*, 2> pointers{};
    for (std::size_t i = 0; i < graphs.size(); ++i) {
        infer::GraphCaptureScope capture(pool, i + 1, stream);
        pointers[i] = static_cast<float*>(pool.allocate(1024 * (i + 1), stream));
        fill<<<1, 256, 0, stream>>>(pointers[i], float(i + 3));
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        pool.release(pointers[i], stream);
        graphs[i] = capture.finish();
        ASSERT_EQ(cudaGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0), cudaSuccess);
    }
    EXPECT_EQ(pointers[0], pointers[1]);
    EXPECT_EQ(pool.graph_workspace_used(1), 1024u);
    EXPECT_EQ(pool.graph_workspace_used(2), 2048u);
    EXPECT_EQ(pool.stats().graph_peak_used_bytes, 2048u);
    EXPECT_EQ(pool.stats().graph_count, 2u);
    EXPECT_THROW(pool.release_graph_workspace(), std::runtime_error);
    EXPECT_THROW(pool.release(pointers[0], stream), std::invalid_argument);

    auto* eager = pool.allocate(1024, stream);
    EXPECT_NE(eager, pointers[0]);
    pool.release(eager, stream);
    pool.trim();
    EXPECT_EQ(pool.stats().graph_reserved_bytes, 4096u);
    for (std::size_t i = 0; i < execs.size(); ++i) {
        ASSERT_EQ(cudaGraphLaunch(execs[i], stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::array<float, 256> output{};
        ASSERT_EQ(cudaMemcpy(output.data(), pointers[i], sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
        for (const auto value : output) EXPECT_EQ(value, float(i + 3));
        ASSERT_EQ(cudaGraphExecDestroy(execs[i]), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graphs[i]), cudaSuccess);
        pool.forget_graph(i + 1);
    }
    pool.release_graph_workspace();
    EXPECT_EQ(pool.stats().reserved_bytes, 0u);
}

TEST_F(CUDAGraphPoolTest, ScopeAbortsExceptionsAndRejectsForeignRequests) {
    infer::CachingCUDADeviceAllocator pool;
    pool.reserve_graph_workspace(4096);
    EXPECT_THROW(( [&] {
        infer::GraphCaptureScope capture(pool, 1, stream);
        pool.allocate(256, stream); // Abort also discards outstanding host leases.
        throw std::runtime_error("capture failed");
    }()), std::runtime_error);
    EXPECT_EQ(pool.stats().graph_count, 0u);
    cudaStreamCaptureStatus status{};
    ASSERT_EQ(cudaStreamIsCapturing(stream, &status), cudaSuccess);
    EXPECT_EQ(status, cudaStreamCaptureStatusNone);
    {
        infer::GraphCaptureScope capture(pool, 1, stream);
        std::thread foreign([&] {
            EXPECT_THROW(pool.allocate(256, stream), std::runtime_error);
            EXPECT_THROW(pool.trim(), std::runtime_error);
        });
        foreign.join();
        auto* ptr = pool.allocate(256, stream);
        pool.release(ptr, stream);
        const auto graph = capture.finish();
        EXPECT_THROW(capture.finish(), std::logic_error);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        pool.forget_graph(1);
    }
    // A successful CUDA end with a missing host release also rolls back safely.
    EXPECT_THROW(( [&] {
        infer::GraphCaptureScope capture(pool, 2, stream);
        pool.allocate(256, stream);
        capture.finish();
    }()), std::runtime_error);
    EXPECT_EQ(pool.stats().graph_count, 0u);
    pool.release_graph_workspace();
}

TEST_F(CUDAGraphPoolTest, DoesNotFreePrivateStorageWhenAllocatorDiesBeforeGraph) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    float* data = nullptr;
    {
        infer::CachingCUDADeviceAllocator pool;
        pool.reserve_graph_workspace(1024);
        infer::GraphCaptureScope capture(pool, 1, stream);
        data = static_cast<float*>(pool.allocate(1024, stream));
        fill<<<1, 256, 0, stream>>>(data, 9);
        pool.release(data, stream);
        graph = capture.finish();
        ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
        // Deliberately violate the normal ownership order. The allocator must
        // retain graph storage rather than invalidate a still-live executable.
    }
    ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    float output = 0;
    ASSERT_EQ(cudaMemcpy(&output, data, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(output, 9);
    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    // The invalid ownership order orphaned the reservation; reclaim it only
    // after destroying the graph. Production runners use the normal order.
    ASSERT_EQ(cudaFree(data), cudaSuccess);
}

TEST_F(CUDAGraphPoolTest, RecoversAfterCUDAInvalidatesCapture) {
    infer::CachingCUDADeviceAllocator pool;
    pool.reserve_graph_workspace(1024);
    {
        infer::GraphCaptureScope capture(pool, 1, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaErrorStreamCaptureUnsupported);
        EXPECT_THROW(capture.finish(), std::runtime_error);
    }
    cudaGetLastError();
    EXPECT_EQ(pool.stats().graph_count, 0u);
    {
        infer::GraphCaptureScope capture(pool, 1, stream);
        const auto graph = capture.finish();
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        pool.forget_graph(1);
    }
    pool.release_graph_workspace();
}
TEST_F(CUDAGraphPoolTest, SplitsFineSizeClassesWithinBudgetAndCoalescesAtTrim) {
    infer::CachingCUDAAllocatorConfig config;
    config.max_reserved_bytes=2*1024*1024;
    config.max_cached_bytes=config.max_reserved_bytes;
    infer::CachingCUDADeviceAllocator pool(config);
    pool.register_stream(stream);
    const std::array<std::size_t,4> sizes{65792,131328,262400,524544};
    std::array<void*,4> pointers{};
    for (std::size_t i=0;i<sizes.size();++i) {
        pointers[i]=pool.allocate(sizes[i],stream);
        fill<<<1,32,0,stream>>>(static_cast<float*>(pointers[i]),float(i+1));
    }
    const auto active=pool.stats();
    EXPECT_LE(active.reserved_bytes,config.max_reserved_bytes);
    EXPECT_LT(active.active_capacity_bytes-active.active_requested_bytes,active.active_capacity_bytes/50);
    for (std::size_t i=0;i<sizes.size();++i) {
        std::array<float,32> values{};
        ASSERT_EQ(cudaMemcpyAsync(values.data(),pointers[i],sizeof(values),cudaMemcpyDeviceToHost,stream),cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream),cudaSuccess);
        for (const auto value:values) EXPECT_EQ(value,float(i+1));
        pool.release(pointers[i],stream);
    }
    const auto allocations=pool.stats().upstream_allocations;
    for (const auto size:sizes) {
        auto* pointer=pool.allocate(size,stream);
        pool.release(pointer,stream);
    }
    EXPECT_EQ(pool.stats().upstream_allocations,allocations);
    pool.trim();
    EXPECT_EQ(pool.stats().reserved_bytes,0u);
    EXPECT_EQ(pool.stats().active_blocks,0u);
    pool.unregister_stream(stream);
}
} // namespace
