#include "infer/cpu/cpu_memory_pool.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>

TEST(CPUAllocatorContract, FineClassesMatchPhysicalReservation) {
    infer::CachingCPUDeviceAllocator pool;
    const std::array<std::pair<std::size_t, std::size_t>, 9> samples{{
        {1, 64}, {65, 128}, {193, 256}, {1025, 1088}, {4097, 4160},
        {5003, 5056}, {80048, 80896}, {7 << 20, 7 << 20}, {(7 << 20) + 1, (7 << 20) + (64 << 10)}}};
    for (const auto& sample : samples) {
        auto* pointer = pool.allocate(sample.first);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(pointer) % 64, 0);
        EXPECT_EQ(pool.stats().active_bytes, sample.second);
        std::memset(pointer, 17, sample.first);
        pool.release(pointer);
    }
    pool.trim();
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
}

TEST(CPUAllocatorContract, RelativeWasteBoundRejectsOversizedCachedBlocks) {
    infer::CachingCPUAllocatorConfig config;
    config.max_reuse_waste = 64;
    infer::CachingCPUDeviceAllocator pool(config);
    auto* medium = pool.allocate(256);
    auto* large = pool.allocate(1024);
    pool.release(medium);
    pool.release(large);
    auto* first = pool.allocate(129);
    auto* second = pool.allocate(960);
    EXPECT_NE(first, medium);
    EXPECT_NE(second, large);
    EXPECT_EQ(pool.stats().active_bytes, 192 + 960);
    EXPECT_EQ(pool.stats().upstream_allocations, 4);
    EXPECT_EQ(pool.stats().reuse_count, 0);
    pool.release(first);
    pool.release(second);
    pool.trim();
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
}

TEST(CPUAllocatorContract, EqualRoundedClassReusesWithZeroWasteAndIoKeepsAlignment) {
    infer::CachingCPUAllocatorConfig config;
    config.max_reuse_waste = 0;
    infer::CachingCPUDeviceAllocator pool(config);
    auto* first = pool.allocate(65);
    pool.release(first);
    auto* second = pool.allocate(127);
    EXPECT_EQ(first, second);
    EXPECT_EQ(pool.stats().reuse_count, 1);
    auto* io = pool.allocate_io(5003);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(io) % 4096, 0);
    EXPECT_EQ(pool.stats().active_bytes, 128 + 8192);
    pool.release(second);
    pool.release(io);
    pool.trim();
    EXPECT_EQ(pool.stats().active_blocks, 0);
    EXPECT_EQ(pool.stats().reserved_bytes, 0);
}
