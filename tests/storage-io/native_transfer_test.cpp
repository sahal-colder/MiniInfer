#include "infer/storage/file_io.h"
#include "../../integrations/vllm/native_transfer.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <sys/uio.h>
#include <unistd.h>

namespace {
enum class Fault { None, Read, Copy, Record };
std::atomic<Fault> fault{Fault::None};
std::atomic<bool> unknown{false};
std::atomic<int> copies{0}, synchronizations{0}, destroyed_events{0};

bool fail(Fault value) {
    auto expected = value;
    return fault.compare_exchange_strong(expected, Fault::None);
}
}

extern "C" CUresult __real_cuMemcpyAsync(CUdeviceptr, CUdeviceptr, std::size_t, CUstream);
extern "C" CUresult __wrap_cuMemcpyAsync(CUdeviceptr to, CUdeviceptr from, std::size_t size,
                                         CUstream stream) {
    ++copies;
    if (fail(Fault::Copy)) return CUDA_ERROR_INVALID_VALUE;
    return __real_cuMemcpyAsync(to, from, size, stream);
}
extern "C" CUresult __real_cuEventRecord(CUevent, CUstream);
extern "C" CUresult __wrap_cuEventRecord(CUevent event, CUstream stream) {
    if (fail(Fault::Record)) return CUDA_ERROR_INVALID_VALUE;
    return __real_cuEventRecord(event, stream);
}
extern "C" CUresult __real_cuStreamSynchronize(CUstream);
extern "C" CUresult __wrap_cuStreamSynchronize(CUstream stream) {
    ++synchronizations;
    if (unknown) return CUDA_ERROR_UNKNOWN;
    return __real_cuStreamSynchronize(stream);
}
extern "C" CUresult __real_cuEventDestroy_v2(CUevent);
extern "C" CUresult __wrap_cuEventDestroy_v2(CUevent event) {
    ++destroyed_events;
    return __real_cuEventDestroy_v2(event);
}
extern "C" ssize_t __real_pread(int, void*, std::size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* data, std::size_t size, off_t offset) {
    if (fail(Fault::Read)) { errno = EIO; return -1; }
    return __real_pread(fd, data, size, offset);
}
extern "C" ssize_t __real_preadv(int, const iovec*, int, off_t);
extern "C" ssize_t __wrap_preadv(int fd, const iovec* parts, int count, off_t offset) {
    if (fail(Fault::Read)) { errno = EIO; return -1; }
    return __real_preadv(fd, parts, count, offset);
}

namespace {
constexpr std::size_t bytes = 4096, blocks = 8, slots = 4;

class NativeTransferTest : public testing::Test {
  protected:
    infer::FileIO io{{infer::FileIOMode::POSIX, 4}};
    int fd = -1;
    void* gpu = nullptr;
    void* host[2]{};
    cudaStream_t streams[2]{};
    std::unique_ptr<HostIOQueue> queue;
    std::unique_ptr<NativeTransfer> transfer;
    std::atomic<int> reads{0};
    std::atomic<bool> gate_second{false}, entered{false};
    std::promise<void> release;
    std::shared_future<void> gate = release.get_future().share();

    void SetUp() override {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && !count)) GTEST_SKIP() << cudaGetErrorString(status);
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&gpu, blocks * bytes), cudaSuccess);
        for (int i = 0; i < 2; ++i) {
            ASSERT_EQ(cudaHostAlloc(&host[i], slots * bytes, cudaHostAllocDefault), cudaSuccess);
            ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cudaSuccess);
        }
        char path[] = "/tmp/miniinfer-native-fault-XXXXXX";
        fd = mkstemp(path);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(unlink(path), 0);
        std::array<unsigned char, blocks * bytes> data{};
        for (std::size_t i = 0; i < blocks; ++i)
            std::memset(data.data() + i * bytes, int(17 + i), bytes);
        ASSERT_EQ(pwrite(fd, data.data(), data.size(), 0), ssize_t(data.size()));
        ASSERT_EQ(cudaMemcpy(gpu, data.data(), data.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto work = [this](bool write, const auto& disk, const auto& pointers) {
            if (!write) {
                ++reads;
                if (gate_second && disk.front() == 2) {
                    entered = true;
                    gate.wait();
                }
            }
            std::vector<infer::FileIORequest> requests;
            for (std::size_t i = 0; i < disk.size(); ++i)
                requests.push_back({pointers[i], bytes, disk[i] * bytes});
            if (write) { io.write_many(fd, requests); return; }
            io.read_many(fd, requests);
        };
        queue = std::make_unique<HostIOQueue>(work, bytes);
        transfer = std::make_unique<NativeTransfer>(work, *queue, bytes, slots, blocks, blocks,
            std::vector<CUdeviceptr>{reinterpret_cast<CUdeviceptr>(gpu)},
            std::vector<std::size_t>{bytes}, host[1], host[0], streams[1], streams[0], 64, true);
        fault = Fault::None;
        unknown = false;
        copies = synchronizations = destroyed_events = 0;
    }
    void TearDown() override {
        fault = Fault::None;
        unknown = false;
        gate_second = false;
        if (gate.wait_for(std::chrono::seconds(0)) != std::future_status::ready) release.set_value();
        transfer.reset();
        queue.reset();
        for (int i = 0; i < 2; ++i) {
            if (streams[i]) cudaStreamDestroy(streams[i]);
            if (host[i]) cudaFreeHost(host[i]);
        }
        if (gpu) cudaFree(gpu);
        if (fd >= 0) close(fd);
    }
    void load_one() {
        const std::size_t index = 0;
        NativeTransfer::Result result;
        transfer->execute(false, &index, &index, 1, result);
        EXPECT_EQ(result.blocks, 1);
        ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
        std::array<unsigned char, bytes> result_bytes{};
        ASSERT_EQ(cudaMemcpy(result_bytes.data(), gpu, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::all_of(result_bytes.begin(), result_bytes.end(), [](auto value) { return value == 17; }));
    }
};

TEST_F(NativeTransferTest, RealFileRoundTripAndReadAheadPreserveNeighbors) {
    const std::array<std::size_t, 5> source{{0, 1, 2, 3, 4}}, target{{3, 4, 5, 6, 7}};
    NativeTransfer::Result saved, loaded;
    transfer->execute(true, source.data(), source.data(), source.size(), saved);
    ASSERT_EQ(saved.blocks, source.size());
    std::array<unsigned char, 5 * bytes> stored{};
    ASSERT_EQ(pread(fd, stored.data(), stored.size(), 0), ssize_t(stored.size()));
    for (std::size_t block = 0; block < source.size(); ++block)
        for (std::size_t byte = 0; byte < bytes; ++byte)
            ASSERT_EQ(stored[block * bytes + byte], 17 + block) << "stored block " << block;
    ASSERT_EQ(cudaMemset(gpu, 0, blocks * bytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    transfer->execute(false, target.data(), source.data(), source.size(), loaded);
    ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
    EXPECT_EQ(loaded.blocks, source.size());
    EXPECT_EQ(loaded.prefetched_blocks, 3);
    std::array<unsigned char, blocks * bytes> output{};
    ASSERT_EQ(cudaMemcpy(output.data(), gpu, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::size_t block = 0; block < blocks; ++block)
        for (std::size_t byte = 0; byte < bytes; ++byte)
            ASSERT_EQ(output[block * bytes + byte], block < 3 ? 0 : 17 + block - 3) << "loaded block " << block;
    EXPECT_EQ(synchronizations.load(), 0); // Normal execution keeps its event-based completion contract.
}

TEST_F(NativeTransferTest, ReadFailureIsConsumedAndNextRequestWorks) {
    const std::size_t index = 0;
    NativeTransfer::Result result;
    fault = Fault::Read;
    EXPECT_THROW(transfer->execute(false, &index, &index, 1, result), std::runtime_error);
    EXPECT_EQ(fault.load(), Fault::None);
    EXPECT_EQ(result.blocks, 0);
    EXPECT_EQ(copies.load(), 0);
    EXPECT_GT(synchronizations.load(), 0);
    load_one();
}

TEST_F(NativeTransferTest, RejectedDmaIsFencedAndNextRequestWorks) {
    const std::size_t index = 0;
    NativeTransfer::Result result;
    fault = Fault::Copy;
    EXPECT_THROW(transfer->execute(false, &index, &index, 1, result), std::runtime_error);
    EXPECT_EQ(fault.load(), Fault::None);
    EXPECT_EQ(copies.load(), 1);
    EXPECT_GT(synchronizations.load(), 0);
    load_one();
}

TEST_F(NativeTransferTest, FailedFenceDrainsAcceptedReadAheadBeforeReturningBuffers) {
    const std::array<std::size_t, 4> indices{{0, 1, 2, 3}};
    gate_second = true;
    fault = Fault::Record;
    auto task = std::async(std::launch::async, [&] {
        NativeTransfer::Result result;
        EXPECT_THROW(transfer->execute(false, indices.data(), indices.data(), indices.size(), result), std::runtime_error);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool accepted = entered;
    const bool held = task.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    release.set_value();
    task.get();
    ASSERT_TRUE(accepted);
    ASSERT_TRUE(held);
    EXPECT_EQ(reads.load(), 2);
    EXPECT_GT(synchronizations.load(), 0);
    gate_second = false;
    load_one();
}

TEST_F(NativeTransferTest, UnknownCompletionBlocksReuseUntilRecoveryIsConfirmed) {
    const std::size_t index = 0;
    NativeTransfer::Result result;
    fault = Fault::Record;
    unknown = true;
    EXPECT_THROW(transfer->execute(false, &index, &index, 1, result), std::runtime_error);
    ASSERT_EQ(copies.load(), 1); // A real DMA was accepted before the failed fence.
    const auto prior_reads = reads.load();
    EXPECT_THROW(transfer->execute(false, &index, &index, 1, result), std::runtime_error);
    EXPECT_EQ(copies.load(), 1);
    EXPECT_EQ(reads.load(), prior_reads);
    unknown = false;
    load_one();
}

TEST_F(NativeTransferTest, DestructionWaitsBeforeDestroyingTransferFences) {
    load_one();
    synchronizations = destroyed_events = 0;
    transfer.reset();
    EXPECT_EQ(synchronizations.load(), 2);
    EXPECT_EQ(destroyed_events.load(), 4);
    cudaPointerAttributes attributes{};
    for (auto* pointer : host) {
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, pointer), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    }
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, gpu), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
}
}
