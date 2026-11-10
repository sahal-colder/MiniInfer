#include "infer/storage/ssd_device.h"
#include <spdk/bdev.h>
#include <spdk/env.h>
#include <spdk/event.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
struct Completion {
    spdk_bdev_io_completion_cb callback = nullptr;
    void* argument = nullptr;
};
std::array<Completion, 4096> completions;
std::atomic<unsigned> submitted{0}, accepted{0}, completed{0}, exhausted{0}, queued{0};
std::atomic<unsigned> inflight{0}, unsafe_frees{0}, stops{0};

void completion(spdk_bdev_io* io, bool success, void* argument) {
    auto& saved = *static_cast<Completion*>(argument);
    ++completed;
    --inflight;
    saved.callback(io, success, saved.argument);
}
}

extern "C" int __real_spdk_bdev_write(spdk_bdev_desc*, spdk_io_channel*, void*, uint64_t,
                                      uint64_t, spdk_bdev_io_completion_cb, void*);
extern "C" int __real_spdk_bdev_read(spdk_bdev_desc*, spdk_io_channel*, void*, uint64_t,
                                     uint64_t, spdk_bdev_io_completion_cb, void*);

namespace {
int submit(bool write, spdk_bdev_desc* desc, spdk_io_channel* channel, void* data,
           uint64_t offset, uint64_t length, spdk_bdev_io_completion_cb callback, void* argument) {
    const auto index = submitted++;
    if (index >= completions.size()) std::terminate();
    auto& saved = completions[index];
    saved = {callback, argument};
    ++inflight;
    const auto result = (write ? __real_spdk_bdev_write : __real_spdk_bdev_read)(
        desc, channel, data, offset, length, completion, &saved);
    if (!result) ++accepted;
    else {
        --inflight;
        if (result == -ENOMEM) ++exhausted;
    }
    return result;
}
}
extern "C" int __wrap_spdk_bdev_write(spdk_bdev_desc* desc, spdk_io_channel* channel,
        void* data, uint64_t offset, uint64_t length, spdk_bdev_io_completion_cb callback, void* argument) {
    return submit(true, desc, channel, data, offset, length, callback, argument);
}
extern "C" int __wrap_spdk_bdev_read(spdk_bdev_desc* desc, spdk_io_channel* channel,
        void* data, uint64_t offset, uint64_t length, spdk_bdev_io_completion_cb callback, void* argument) {
    return submit(false, desc, channel, data, offset, length, callback, argument);
}
extern "C" int __real_spdk_bdev_queue_io_wait(spdk_bdev*, spdk_io_channel*, spdk_bdev_io_wait_entry*);
extern "C" int __wrap_spdk_bdev_queue_io_wait(spdk_bdev* bdev, spdk_io_channel* channel,
                                              spdk_bdev_io_wait_entry* entry) {
    const auto result = __real_spdk_bdev_queue_io_wait(bdev, channel, entry);
    if (!result) ++queued;
    return result;
}
extern "C" void __real_spdk_dma_free(void*);
extern "C" void __wrap_spdk_dma_free(void* data) {
    if (inflight) ++unsafe_frees;
    __real_spdk_dma_free(data);
}
extern "C" void __real_spdk_app_stop(int);
extern "C" void __wrap_spdk_app_stop(int result) {
    ++stops;
    __real_spdk_app_stop(result);
}

namespace {
infer::SpdkBdevConfig config(const char* file) {
    infer::SpdkBdevConfig result;
    result.json_config = file;
    result.bdev_name = "Malloc0";
    result.capacity_bytes = 32 << 20;
    result.no_huge = result.no_pci = true;
    result.io_buffer_bytes = 512 << 10;
    result.dma_budget_bytes = 16 << 20;
    return result;
}

TEST(SpdkLifecycle, RealBdevIoExhaustionWaitsRetriesAndDrainsBeforeFree) {
    auto device = infer::open_spdk_device(config(INFER_SPDK_LOWIO_JSON));
    const auto initial = accepted.load();
    ASSERT_EQ(inflight.load(), 0);
    constexpr std::size_t bytes = 4096, count = 128;
    std::vector<unsigned char> input(bytes * count), output(bytes * count);
    std::vector<infer::FileIORequest> writes, reads;
    for (std::size_t i = 0; i < count; ++i) {
        std::memset(input.data() + i * bytes, int(i + 1), bytes);
        writes.push_back({input.data() + i * bytes, bytes, i * bytes});
        reads.push_back({output.data() + i * bytes, bytes, i * bytes});
    }
    device->write_many(writes);
    device->read_many(reads);
    EXPECT_EQ(input, output);
    EXPECT_GT(exhausted.load(), 0); // Observes the actual SDK return, never injects ENOMEM.
    EXPECT_EQ(queued.load(), exhausted.load());
    EXPECT_EQ(accepted.load() - initial, 2 * count);
    EXPECT_EQ(completed.load(), accepted.load());
    EXPECT_EQ(inflight.load(), 0);
    device->write(input.data(), bytes, 0);
    device->read(output.data(), bytes, 0);
    EXPECT_EQ(std::memcmp(input.data(), output.data(), bytes), 0);
    device.reset();
    EXPECT_EQ(stops.load(), 1);
    EXPECT_EQ(unsafe_frees.load(), 0);
    EXPECT_EQ(completed.load(), accepted.load());
    std::cout << "accepted=" << accepted << " completed=" << completed
              << " initial=" << initial << " sdk_enomem=" << exhausted << " queued=" << queued << '\n';
}

TEST(SpdkLifecycle, ZeroIoCacheIsRejectedBeforeAcceptingUserIo) {
    EXPECT_THROW(infer::open_spdk_device(config(INFER_SPDK_ZEROIO_JSON)), std::invalid_argument);
    EXPECT_EQ(stops.load(), 1);
    EXPECT_EQ(inflight.load(), 0);
    EXPECT_EQ(unsafe_frees.load(), 0);
}

TEST(SpdkLifecycle, DmaLeaseRetainsArenaUntilLastOwnerAndBudgetFailureCanRecover) {
    const auto options = config(INFER_SPDK_MALLOC_JSON);
    auto device = infer::open_spdk_device(options);
    const auto before = device->stats().dma_reserved_bytes;
    EXPECT_THROW(device->allocate_dma_buffer(options.dma_budget_bytes), std::bad_alloc);
    EXPECT_EQ(device->stats().dma_reserved_bytes, before);
    auto buffer = device->allocate_dma_buffer(8192);
    ASSERT_TRUE(buffer);
    EXPECT_TRUE(device->owns_dma_range(buffer->data(), buffer->size()));
    EXPECT_EQ(device->stats().dma_reserved_bytes, before + buffer->size());
    std::memset(buffer->data(), 42, buffer->size());
    device->write(buffer->data(), buffer->size(), 0);
    std::memset(buffer->data(), 0, buffer->size());
    device->read(buffer->data(), buffer->size(), 0);
    EXPECT_TRUE(std::all_of(static_cast<unsigned char*>(buffer->data()),
        static_cast<unsigned char*>(buffer->data()) + buffer->size(), [](auto value) { return value == 42; }));
    device.reset();
    EXPECT_EQ(stops.load(), 0); // The DMA lease, rather than the public handle, owns the arena.
    std::memset(buffer->data(), 17, buffer->size());
    buffer.reset();
    EXPECT_EQ(stops.load(), 1);
    EXPECT_EQ(unsafe_frees.load(), 0);
    EXPECT_EQ(inflight.load(), 0);
    EXPECT_EQ(completed.load(), accepted.load());
    EXPECT_THROW(infer::open_spdk_device(options), std::runtime_error);
}
}
