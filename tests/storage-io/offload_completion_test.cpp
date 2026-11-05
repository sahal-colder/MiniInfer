#include "infer/storage/kv_offload_cache.h"

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <sys/uio.h>
#include <unistd.h>

namespace {
std::atomic<bool> fail_writes{false};

class OffloadCompletion : public testing::TestWithParam<bool> {};

TEST_P(OffloadCompletion, FailedDemotionsPreserveSourcesAndAllowRetry) {
    infer::KVOffloadConfig config{4096, 8, 5, "/tmp"};
    config.overlap_writeback = GetParam();
    config.ssd_verification = infer::SSDVerification::ON_READ;
    infer::KVOffloadCache cache(config);
    std::array<unsigned char, 4096> original{}, replacement{}, output{};
    original.fill(11);
    replacement.fill(23);
    for (std::size_t slot = 0; slot < 4; ++slot)
        cache.write(slot, original.data());

    fail_writes = true;
    EXPECT_THROW(cache.write_many({4, 5, 6, 7}, {replacement.data(), replacement.data(),
                                                 replacement.data(), replacement.data()}),
                 std::runtime_error);
    fail_writes = false;

    // Overlap may commit the foreground write into the already free CPU slot.
    EXPECT_EQ(cache.contains(4), GetParam());
    for (auto slot : {5u, 6u, 7u})
        EXPECT_FALSE(cache.contains(slot));
    for (std::size_t slot = 0; slot < 4; ++slot) {
        cache.read(slot, output.data());
        EXPECT_EQ(output, original);
    }
    EXPECT_EQ(cache.stats().ssd_reserved_bytes, 0u);
    EXPECT_EQ(cache.stats().cpu_pending_cleanup_bytes, 0u);
    EXPECT_EQ(cache.stats().ssd_pending_cleanup_bytes, 0u);

    cache.write_many({4, 5, 6, 7}, {replacement.data(), replacement.data(), replacement.data(),
                                    replacement.data()});
    for (std::size_t slot = 4; slot < 8; ++slot) {
        cache.read(slot, output.data());
        EXPECT_EQ(output, replacement);
    }
    for (std::size_t slot = 0; slot < 4; ++slot)
        EXPECT_TRUE(cache.erase(slot));
}

TEST_P(OffloadCompletion, MultipleWavesRetainContentsAndBoundCapacity) {
    infer::KVOffloadConfig config{4096, 32, 8, "/tmp"};
    config.overlap_writeback = GetParam();
    config.ssd_verification = infer::SSDVerification::ON_READ;
    infer::KVOffloadCache cache(config);
    std::array<unsigned char, 4096> first{}, second{}, output{};
    first.fill(37);
    second.fill(79);
    for (std::size_t group = 0; group < 32; group += 8) {
        std::vector<std::size_t> slots;
        std::vector<void*> inputs;
        for (std::size_t i = 0; i < 8; ++i) {
            slots.push_back(group + i);
            inputs.push_back(i % 2 ? second.data() : first.data());
        }
        cache.write_many(slots, inputs);
    }
    EXPECT_GT(cache.stats().demotions, 0u);
    for (std::size_t slot = 0; slot < 32; ++slot) {
        cache.read(slot, output.data());
        EXPECT_EQ(output, slot % 2 ? second : first);
    }
    EXPECT_LE(cache.stats().cpu_reserved_bytes, 8u * 4096);
    EXPECT_EQ(cache.stats().cpu_pending_cleanup_bytes, 0u);
    EXPECT_EQ(cache.stats().ssd_pending_cleanup_bytes, 0u);
    for (std::size_t slot = 0; slot < 32; ++slot)
        EXPECT_TRUE(cache.erase(slot));
    EXPECT_EQ(cache.stats().cpu_reserved_bytes, 0u);
    EXPECT_EQ(cache.stats().ssd_reserved_bytes, 0u);
}

INSTANTIATE_TEST_SUITE_P(BatchAndOverlap, OffloadCompletion, testing::Bool());
} // namespace

// Inject only the syscall error; successful operations use the real file backend.
extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);
extern "C" ssize_t __real_pwritev(int, const iovec*, int, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void* data, size_t size, off_t offset) {
    if (fail_writes) {
        errno = ENOSPC;
        return -1;
    }
    return __real_pwrite(fd, data, size, offset);
}
extern "C" ssize_t __wrap_pwritev(int fd, const iovec* parts, int count, off_t offset) {
    if (fail_writes) {
        errno = ENOSPC;
        return -1;
    }
    return __real_pwritev(fd, parts, count, offset);
}
