#include "infer/cuda/cuda_memory_pool.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <new>
#include <thread>
#include <vector>

namespace {
thread_local int fail_new = -1;
std::atomic<int> mallocs{0}, frees{0}, live{0};
std::atomic<int> fail_malloc{0}, fail_free{0}, fail_record{0}, fail_create{0}, fail_sync{0};
std::atomic<bool> hold_queries{false};
bool fail(std::atomic<int>& remaining) {
    if (!remaining) return false;
    --remaining;
    return true;
}
}

void* operator new(std::size_t size) {
    if (fail_new == 0) { fail_new = -1; throw std::bad_alloc(); }
    if (fail_new > 0) --fail_new;
    if (auto* pointer = std::malloc(size ? size : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t size) {
    ++mallocs;
    if (fail(fail_malloc)) { hold_queries = false; return cudaErrorMemoryAllocation; }
    const auto pending = fail_new;
    fail_new = -1; // Fault only pool metadata, not driver-internal host allocation.
    const auto result = __real_cudaMalloc(pointer, size);
    fail_new = pending;
    if (result == cudaSuccess) ++live;
    return result;
}
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
    ++frees;
    if (fail(fail_free)) return cudaErrorUnknown;
    const auto result = __real_cudaFree(pointer);
    if (result == cudaSuccess && pointer) --live;
    return result;
}
extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (fail(fail_record)) return cudaErrorUnknown;
    return __real_cudaEventRecord(event, stream);
}
extern "C" cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    if (fail(fail_create)) return cudaErrorUnknown;
    return __real_cudaEventCreateWithFlags(event, flags);
}
extern "C" cudaError_t __real_cudaEventQuery(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventQuery(cudaEvent_t event) {
    return hold_queries ? cudaErrorNotReady : __real_cudaEventQuery(event);
}
extern "C" cudaError_t __real_cudaDeviceSynchronize();
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    if (fail(fail_sync)) return cudaErrorUnknown;
    return __real_cudaDeviceSynchronize();
}

namespace {
constexpr std::size_t MiB = 1 << 20;
class CUDAAllocatorContract : public testing::Test {
  protected:
    cudaStream_t first = nullptr, second = nullptr;
    void SetUp() override {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && !count)) GTEST_SKIP() << cudaGetErrorString(status);
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(live.load(), 0);
        mallocs = frees = 0;
    }
    void TearDown() override {
        fail_new = -1;
        fail_malloc = fail_free = fail_record = fail_create = fail_sync = 0;
        hold_queries = false;
        if (first) cudaStreamDestroy(first);
        if (second) cudaStreamDestroy(second);
        EXPECT_EQ(live.load(), 0);
    }
    void balanced(const infer::CachingCUDADeviceAllocator& pool) {
        const auto stats = pool.stats();
        EXPECT_EQ(stats.reserved_bytes, stats.active_capacity_bytes + stats.pending_bytes +
                  stats.ready_bytes + stats.graph_reserved_bytes);
        EXPECT_EQ(stats.segments + stats.splits - stats.merges,
                  stats.active_blocks + stats.pending_blocks + stats.ready_blocks);
    }
};

TEST_F(CUDAAllocatorContract, DestructionRetainsActiveSegmentWhenCompletionIsUnknown) {
    auto pool = std::make_unique<infer::CachingCUDADeviceAllocator>();
    auto* pointer = pool->allocate(2 * MiB, first);
    ASSERT_EQ(cudaMemsetAsync(pointer, 0, 2 * MiB, first), cudaSuccess);
    fail_sync = 1;
    pool.reset();
    EXPECT_EQ(frees.load(), 0);
    // The injected failure did not damage the CUDA context. Reclaim the
    // intentionally retained segment only after confirming completion.
    ASSERT_EQ(cudaStreamSynchronize(first), cudaSuccess);
    if (!frees) EXPECT_EQ(cudaFree(pointer), cudaSuccess);
}

TEST_F(CUDAAllocatorContract, DefaultRoundingAndExplicitLargeGranularityAreDifferentContracts) {
    for (bool explicit_granularity : {false, true}) {
        infer::CachingCUDAAllocatorConfig config;
        if (explicit_granularity) config.large_allocation_granularity = 2 * MiB;
        infer::CachingCUDADeviceAllocator pool(config);
        auto* pointer = pool.allocate(17 * MiB + 1, first);
        EXPECT_EQ(pool.stats().active_capacity_bytes, explicit_granularity ? 18 * MiB : 17 * MiB + MiB / 4);
        balanced(pool);
        pool.release(pointer, first);
        pool.trim();
        EXPECT_EQ(pool.stats().reserved_bytes, 0);
    }
}

TEST_F(CUDAAllocatorContract, DefaultSlabSplitsAndCoalescesBeforeSingleRootFree) {
    infer::CachingCUDADeviceAllocator pool;
    auto* pointer = pool.allocate(256, first);
    EXPECT_EQ(pool.stats().reserved_bytes, 2 * MiB);
    EXPECT_EQ(pool.stats().active_capacity_bytes, 256);
    EXPECT_EQ(pool.stats().ready_bytes, 2 * MiB - 256);
    EXPECT_EQ(pool.stats().splits, 1);
    pool.release(pointer, first);
    pool.flush();
    ASSERT_EQ(cudaStreamSynchronize(first), cudaSuccess);
    pool.collect();
    EXPECT_EQ(pool.stats().ready_bytes, 2 * MiB);
    EXPECT_EQ(pool.stats().ready_blocks, 1);
    balanced(pool);
    pool.trim();
    EXPECT_EQ(frees.load(), 1);
}

TEST_F(CUDAAllocatorContract, MetadataFailureRetainsOwnedReadySlabAndTrimReclaimsIt) {
    bool saw_retained_slab = false;
    for (int position = 0; position < 12; ++position) {
        infer::CachingCUDADeviceAllocator pool;
        pool.register_stream(first);
        bool failed = false;
        void* pointer = nullptr;
        fail_new = position;
        try { pointer = pool.allocate(256, first); }
        catch (const std::bad_alloc&) { failed = true; }
        fail_new = -1;
        if (failed) {
            EXPECT_EQ(pool.stats().active_blocks, 0);
            if (pool.stats().reserved_bytes) {
                saw_retained_slab = true;
                EXPECT_EQ(pool.stats().reserved_bytes, 2 * MiB);
                EXPECT_EQ(pool.stats().ready_bytes, 2 * MiB);
                EXPECT_EQ(live.load(), 1);
                EXPECT_GT(mallocs.load(), 0);
            }
        }
        if (pointer) pool.release(pointer, first);
        balanced(pool);
        pool.trim();
        EXPECT_EQ(pool.stats().reserved_bytes, 0);
        EXPECT_EQ(live.load(), 0);
    }
    EXPECT_TRUE(saw_retained_slab);
}

TEST_F(CUDAAllocatorContract, SplitMetadataFailureLeavesTheExistingSlabUnchanged) {
    infer::CachingCUDADeviceAllocator pool;
    auto* anchor = pool.allocate(MiB, first);
    const auto before = pool.stats();
    bool failed = false;
    fail_new = 0;
    try { (void)pool.allocate(256, first); }
    catch (const std::bad_alloc&) { failed = true; }
    fail_new = -1;
    ASSERT_TRUE(failed);
    EXPECT_EQ(pool.stats().active_capacity_bytes, before.active_capacity_bytes);
    EXPECT_EQ(pool.stats().ready_bytes, before.ready_bytes);
    EXPECT_EQ(pool.stats().splits, before.splits);
    EXPECT_EQ(live.load(), 1);
    auto* retry = pool.allocate(256, first);
    pool.release(retry, first);
    pool.release(anchor, first);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, OomActuallyCallsUpstreamTrimsAndRetriesOnce) {
    infer::CachingCUDADeviceAllocator pool;
    auto* cached = pool.allocate(MiB, first);
    pool.release(cached, first);
    const auto attempts = mallocs.load(), releases = frees.load();
    fail_malloc = 1;
    auto* pointer = pool.allocate(3 * MiB, second); // Cannot fit anywhere in the existing 2 MiB slab.
    EXPECT_EQ(fail_malloc.load(), 0);
    EXPECT_EQ(mallocs.load() - attempts, 2);
    EXPECT_EQ(frees.load() - releases, 1);
    EXPECT_EQ(pool.stats().oom_retries, 1);
    EXPECT_EQ(pool.stats().reserved_bytes, 3 * MiB);
    balanced(pool);
    pool.release(pointer, second);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, OomRecoveryCanReuseNewlyCompletedTailWithoutSecondMalloc) {
    infer::CachingCUDADeviceAllocator pool;
    auto* anchor = pool.allocate(MiB, first);
    auto* tail = pool.allocate(MiB, first);
    hold_queries = true;
    pool.release(tail, first);
    const auto attempts = mallocs.load();
    fail_malloc = 1;
    auto* retry = pool.allocate(MiB, second);
    EXPECT_EQ(fail_malloc.load(), 0);
    EXPECT_EQ(mallocs.load() - attempts, 1);
    EXPECT_EQ(pool.stats().oom_retries, 1);
    EXPECT_EQ(retry, tail);
    EXPECT_EQ(pool.stats().active_blocks, 2);
    balanced(pool);
    pool.release(anchor, first);
    pool.release(retry, second);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, BothUpstreamOomAttemptsFailWithoutLosingBookkeeping) {
    infer::CachingCUDADeviceAllocator pool;
    fail_malloc = 2;
    EXPECT_THROW(pool.allocate(MiB, first), std::bad_alloc);
    EXPECT_EQ(fail_malloc.load(), 0);
    EXPECT_EQ(mallocs.load(), 2);
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
    EXPECT_EQ(pool.stats().active_blocks, 0);
    balanced(pool);
    auto* retry = pool.allocate(MiB, first);
    pool.release(retry, first);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, FailedFreeKeepsSegmentTraceableUntilRetry) {
    infer::CachingCUDADeviceAllocator pool;
    auto* pointer = pool.allocate(256, first);
    pool.release(pointer, first);
    const auto reservation = pool.stats().reserved_bytes;
    fail_free = 1;
    EXPECT_THROW(pool.trim(), std::runtime_error);
    EXPECT_EQ(fail_free.load(), 0);
    EXPECT_EQ(live.load(), 1);
    EXPECT_EQ(pool.stats().reserved_bytes, reservation);
    EXPECT_EQ(pool.stats().ready_bytes, reservation);
    EXPECT_EQ(pool.stats().ready_blocks, 1);
    balanced(pool);
    auto* reused = pool.allocate(256, first);
    EXPECT_EQ(reused, pointer);
    pool.release(reused, first);
    pool.trim();
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
}

TEST_F(CUDAAllocatorContract, EventCreationFailureDoesNotAcquireDeviceStorage) {
    infer::CachingCUDADeviceAllocator pool;
    fail_create = 1;
    EXPECT_THROW(pool.allocate(256, first), std::runtime_error);
    EXPECT_EQ(fail_create.load(), 0);
    EXPECT_EQ(mallocs.load(), 0);
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
    auto* retry = pool.allocate(256, first);
    pool.release(retry, first);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, FailedRetirementEventKeepsCallerOwnership) {
    infer::CachingCUDADeviceAllocator pool;
    auto* pointer = pool.allocate(256, first);
    fail_record = 1;
    EXPECT_THROW(pool.release(pointer, first), std::runtime_error);
    EXPECT_EQ(fail_record.load(), 0);
    EXPECT_EQ(pool.stats().active_blocks, 1);
    EXPECT_EQ(pool.stats().pending_blocks, 0);
    pool.release(pointer, first);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, BatchedFlushFailureKeepsThePendingAllocationUntilRetry) {
    infer::CachingCUDAAllocatorConfig config;
    config.event_batch_size = 16;
    infer::CachingCUDADeviceAllocator pool(config);
    auto* pointer = pool.allocate(256, first);
    pool.release(pointer, first);
    fail_record = 1;
    EXPECT_THROW(pool.flush(), std::runtime_error);
    EXPECT_EQ(fail_record.load(), 0);
    EXPECT_EQ(pool.stats().pending_blocks, 1);
    EXPECT_EQ(live.load(), 1);
    balanced(pool);
    pool.trim();
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
}

TEST_F(CUDAAllocatorContract, IncompleteEventCannotBeReusedByAnotherStream) {
    infer::CachingCUDADeviceAllocator pool;
    auto* anchor = pool.allocate(MiB, first);
    auto* pending = pool.allocate(MiB, first);
    hold_queries = true;
    pool.release(pending, first);
    auto* other = pool.allocate(MiB, second);
    EXPECT_NE(other, pending);
    EXPECT_GE(pool.stats().pending_bytes, MiB);
    EXPECT_EQ(pool.stats().active_blocks, 2);
    hold_queries = false;
    pool.release(anchor, first);
    pool.release(other, second);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, RealPendingDmaKeepsItsTailUntilTheProducerEventCompletes) {
    infer::CachingCUDADeviceAllocator pool;
    auto* anchor = pool.allocate(MiB, first);
    auto* pending = pool.allocate(MiB, first);
    auto* spare = pool.allocate(2 * MiB, second);
    pool.release(spare, second);
    pool.flush();
    ASSERT_EQ(cudaStreamSynchronize(second), cudaSuccess);
    pool.collect(); // Prepare enough ready storage before blocking a CUDA callback.
    struct Gate {
        cudaStream_t stream;
        std::atomic<bool> ready{false}, release{false};
    } gate{first};
    const std::unique_ptr<Gate, void (*)(Gate*)> release(&gate, [](Gate* state) {
        state->release = true;
        cudaStreamSynchronize(state->stream);
    });
    ASSERT_EQ(cudaLaunchHostFunc(first, [](void* pointer) {
        auto& state = *static_cast<Gate*>(pointer);
        state.ready = true;
        while (!state.release) std::this_thread::yield();
    }, &gate), cudaSuccess);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.ready && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    ASSERT_TRUE(gate.ready);
    ASSERT_EQ(cudaMemsetAsync(pending, 42, MiB, first), cudaSuccess);
    pool.release(pending, first);
    pool.flush();
    const auto attempts = mallocs.load();
    auto* other = pool.allocate(MiB, second);
    EXPECT_NE(other, pending);
    EXPECT_EQ(mallocs.load(), attempts);
    EXPECT_GE(pool.stats().pending_bytes, MiB);
    gate.release = true;
    ASSERT_EQ(cudaStreamSynchronize(first), cudaSuccess);
    pool.collect();
    std::vector<unsigned char> data(MiB);
    ASSERT_EQ(cudaMemcpy(data.data(), pending, data.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](auto value) { return value == 42; }));
    pool.release(anchor, first);
    pool.release(other, second);
    pool.trim();
}

TEST_F(CUDAAllocatorContract, DestructorOwnsActiveAndPendingSegmentsWithoutFreeingInteriorPointers) {
    {
        infer::CachingCUDADeviceAllocator pool;
        (void)pool.allocate(MiB, first);
        auto* pending = pool.allocate(MiB, first);
        pool.release(pending, first);
        EXPECT_EQ(live.load(), 1);
    }
    EXPECT_EQ(frees.load(), 1);
    EXPECT_EQ(live.load(), 0);
}

TEST_F(CUDAAllocatorContract, GraphWorkspaceAllocationAndFreeFailuresRemainRetryable) {
    infer::CachingCUDADeviceAllocator pool;
    fail_malloc = 1;
    EXPECT_THROW(pool.reserve_graph_workspace(4096), std::bad_alloc);
    EXPECT_EQ(fail_malloc.load(), 0);
    EXPECT_EQ(pool.stats().graph_reserved_bytes, 0);
    pool.reserve_graph_workspace(4096);
    fail_free = 1;
    EXPECT_THROW(pool.release_graph_workspace(), std::runtime_error);
    EXPECT_EQ(pool.stats().graph_reserved_bytes, 4096);
    EXPECT_EQ(live.load(), 1);
    pool.release_graph_workspace();
    EXPECT_EQ(pool.stats().graph_reserved_bytes, 0);
}

TEST_F(CUDAAllocatorContract, GraphMetadataFailuresDoNotPublishWorkspaceOrCapture) {
    infer::CachingCUDADeviceAllocator pool;
    bool failed = false;
    fail_new = 0;
    try { pool.reserve_graph_workspace(4096); }
    catch (const std::bad_alloc&) { failed = true; }
    fail_new = -1;
    ASSERT_TRUE(failed);
    EXPECT_EQ(live.load(), 0);
    EXPECT_EQ(pool.stats().graph_reserved_bytes, 0);
    pool.reserve_graph_workspace(4096);
    pool.register_stream(first);
    failed = false;
    fail_new = 0;
    try { pool.begin_graph_capture(1, first); }
    catch (const std::bad_alloc&) { failed = true; }
    fail_new = -1;
    EXPECT_TRUE(failed);
    EXPECT_EQ(pool.stats().graph_count, 0);
    pool.release_graph_workspace();
}
}
