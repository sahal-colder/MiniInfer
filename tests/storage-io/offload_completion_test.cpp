#include "infer/storage/kv_offload_cache.h"
#include "infer/storage/tiered_storage.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
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

TEST(OffloadCache, VariableResizePreservesAcknowledgedDataOnDemotionFailure) {
    infer::KVOffloadConfig config{8192, 3, 1, "/tmp"};
    config.variable_payloads = true;
    config.retain_ssd_copy = true;
    config.ssd_verification = infer::SSDVerification::ON_READ;
    infer::KVOffloadCache cache(config);
    std::array<unsigned char, 8192> first{}, replacement{}, output{};
    first.fill(11);
    replacement.fill(23);
    cache.write_many({0, 1}, {first.data(), first.data()}, {4096, 4096});
    const auto original = cache.metadata(0);

    fail_writes = true;
    EXPECT_THROW(cache.write(0, replacement.data()), std::runtime_error);
    fail_writes = false;
    EXPECT_EQ(cache.metadata(0).id, original.id);
    EXPECT_EQ(cache.metadata(0).size, 4096u);
    for (auto slot : {0u, 1u}) {
        output.fill(0);
        EXPECT_EQ(cache.read(slot, output.data(), output.size()), 4096u);
        EXPECT_TRUE(std::equal(output.begin(), output.begin() + 4096, first.begin()));
    }

    cache.write(0, replacement.data());
    EXPECT_NE(cache.metadata(0).id, original.id);
    EXPECT_EQ(cache.metadata(0).size, replacement.size());
    EXPECT_EQ(cache.read(0, output.data(), output.size()), replacement.size());
    EXPECT_EQ(output, replacement);
    EXPECT_FALSE(cache.prefetch_if_space_async(1).get());
    ASSERT_TRUE(cache.prefetch_async(1).get());
    EXPECT_TRUE(cache.metadata(1).ssd_copy);
    EXPECT_EQ(cache.read(1, output.data(), output.size()), 4096u);
    EXPECT_TRUE(std::equal(output.begin(), output.begin() + 4096, first.begin()));
    EXPECT_EQ(cache.metadata(0).location, infer::StorageLocation::SSD);
    EXPECT_EQ(cache.stats().prefetch_inflight, 0u);
    EXPECT_EQ(cache.stats().cpu_pending_cleanup_bytes, 0u);
    EXPECT_EQ(cache.stats().ssd_pending_cleanup_bytes, 0u);
}

TEST(OffloadCache, ResidentListTracksPromotionEraseAndReadAccounting) {
    infer::KVOffloadConfig config{4096, 4, 2, "/tmp"};
    config.promote_hits = 0;
    config.ssd_verification = infer::SSDVerification::ON_READ;
    infer::KVOffloadCache cache(config);
    std::array<unsigned char, 4096> data{}, output{};
    data.fill(37);
    cache.write(0, data.data());
    cache.write(1, data.data());
    cache.read(0, output.data());
    cache.write(2, data.data());
    EXPECT_EQ(cache.metadata(1).location, infer::StorageLocation::SSD);
    cache.read(1, output.data());
    EXPECT_EQ(cache.stats().cpu_hits, 1u);
    EXPECT_EQ(cache.stats().ssd_hits, 1u);
    ASSERT_TRUE(cache.prefetch_async(1).get());
    EXPECT_EQ(cache.metadata(0).location, infer::StorageLocation::SSD);
    EXPECT_EQ(cache.metadata(1).location, infer::StorageLocation::CPU);
    cache.read(1, output.data());
    EXPECT_EQ(cache.stats().cpu_hits, 2u);
    EXPECT_EQ(cache.stats().prefetch_hits, 1u);
    ASSERT_TRUE(cache.erase(1));
    cache.write(3, data.data());
    EXPECT_FALSE(cache.prefetch_if_space_async(0).get());
    ASSERT_TRUE(cache.erase(2));
    ASSERT_TRUE(cache.prefetch_if_space_async(0).get());
    cache.read(0, output.data());
    EXPECT_EQ(output, data);
    EXPECT_EQ(cache.stats().cpu_hits, 3u);
    EXPECT_EQ(cache.stats().ssd_hits, 1u);
    EXPECT_EQ(cache.metadata(0).location, infer::StorageLocation::CPU);
    EXPECT_EQ(cache.metadata(3).location, infer::StorageLocation::CPU);
    EXPECT_EQ(cache.stats().cpu_reserved_bytes, config.cpu_slots * config.payload_bytes);
}

TEST(TieredReadBatch, FixedAndVariableLengthsShareAdmissionAndReleaseProtection) {
    infer::StorageConfig config;
    config.block_bytes = 4096;
    config.budget.cpu_bytes = 8192;
    infer::TieredStorage storage(config);
    const auto first = storage.create(64);
    const auto second = storage.create(128);
    for (auto id : {first, second}) {
        auto lease = storage.acquire(id, infer::AccessMode::WRITE);
        std::memset(lease.writable_data(), 37, lease.size());
        lease.finish();
    }
    std::array<unsigned char, 128> a{}, b{};
    const auto accesses = storage.metadata(first).access_count;
    EXPECT_THROW(storage.read_to_host_many({first, second}, {a.data(), b.data()}, 64),
                 std::invalid_argument);
    EXPECT_EQ(storage.metadata(first).access_count, accesses);
    EXPECT_EQ(storage.metadata(first).lease_count, 0u);
    EXPECT_THROW(storage.read_to_host_many({first, second}, {a.data(), b.data()}, std::vector<std::size_t>{64}),
                 std::invalid_argument);
    storage.read_to_host_many({first, second}, {a.data(), b.data()}, {64, 128});
    EXPECT_TRUE(std::all_of(a.begin(), a.begin() + 64, [](auto byte) { return byte == 37; }));
    EXPECT_TRUE(std::all_of(b.begin(), b.end(), [](auto byte) { return byte == 37; }));
    storage.read_to_host_many({first}, {a.data()}, 64);
    EXPECT_EQ(storage.metadata(first).lease_count, 0u);
    EXPECT_EQ(storage.metadata(second).lease_count, 0u);
    EXPECT_TRUE(storage.erase(first));
    EXPECT_TRUE(storage.erase(second));
    EXPECT_EQ(storage.stats(infer::StorageLocation::CPU).reserved_bytes, 0u);
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
