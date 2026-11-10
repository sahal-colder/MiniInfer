#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/pinned_memory_pool.h"
#include "infer/io/direct_file.h"
#include "infer/storage/paged_kv.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <system_error>

#ifdef TEST_C_DEVICE
#include "infer/io.h"
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace {
using namespace infer;

bool ring_unavailable(int error) {
    return error == ENOTSUP || error == ENOSYS || error == EPERM || error == EACCES;
}

TEST(StorageIO, AlignedCPUAllocationsCanBeReusedAndReleased) {
    CPUDeviceAllocator cpu;
    void* regular = cpu.allocate(17);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(regular) % 64, 0u);
    cpu.release(regular);
    EXPECT_EQ(io_allocation_size(4097), 8192u);
    void* raw = cpu.allocate_io(4097);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(raw) % io_alignment, 0u);
    std::memset(raw, 1, io_allocation_size(4097));
    cpu.release(raw);

    CachingCPUDeviceAllocator pool;
    auto* cached = pool.allocate(5000);
    pool.release(cached);
    auto* aligned = pool.allocate_io(4095);
    EXPECT_NE(aligned, cached);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned) % io_alignment, 0u);
    pool.release(aligned);
    auto* reused = pool.allocate_io(4095);
    EXPECT_EQ(reused, aligned);
    pool.release(reused);
    EXPECT_THROW(pool.allocate_io(0), std::invalid_argument);
    EXPECT_THROW(pool.allocate_io(SIZE_MAX), std::bad_alloc);
}

#ifdef TEST_C_DEVICE
TEST(StorageIO, PositionalBackendsRejectAppendAndPreserveCallerOwnership) {
    char name[] = "/tmp/infer-c-append-XXXXXX";
    const int fd = mkstemp(name);
    ASSERT_GE(fd, 0);
    const auto close_fd = [](const int* p) { close(*p); };
    const std::unique_ptr<const int, decltype(close_fd)> file(&fd, close_fd);
    ASSERT_EQ(unlink(name), 0);
    const int flags = fcntl(fd, F_GETFL);
    ASSERT_GE(flags, 0);
    ASSERT_EQ(fcntl(fd, F_SETFL, flags | O_APPEND), 0);
    for (auto backend : {IF_IO_POSIX, IF_IO_DLSYM, IF_IO_URING}) {
        if_dev* opened = nullptr;
        errno = EDOM;
        const int result = if_dev_open_fd_backend(backend, fd, 4096, 1, &opened);
        const int caller_errno = errno;
        const std::unique_ptr<if_dev, decltype(&if_dev_close)> dev(opened, if_dev_close);
        ASSERT_TRUE(result == -EINVAL || (backend != IF_IO_POSIX && result == -ENOTSUP));
        EXPECT_EQ(opened, nullptr);
        EXPECT_EQ(caller_errno, EDOM);
        EXPECT_EQ(fcntl(fd, F_GETFL), flags | O_APPEND);
    }
    if_dev* opened = nullptr;
    const int result = if_dev_open_uring_fd(fd, 4096, 0, 4, nullptr, 0, &opened);
    const std::unique_ptr<if_dev, decltype(&if_dev_close)> dev(opened, if_dev_close);
    EXPECT_TRUE(result == -EINVAL || result == -ENOTSUP);
    EXPECT_EQ(opened, nullptr);
}

TEST(StorageIO, DlsymAdapterTransfersBothSegmentsAndPreservesErrno) {
    if_dev* opened = nullptr;
    errno = EDOM;
    const int status = if_dev_open_tmp_backend(IF_IO_DLSYM, "/tmp", 4096, &opened);
    if (status == -ENOTSUP) GTEST_SKIP() << "dlsym adapter disabled";
    ASSERT_EQ(status, 0);
    EXPECT_EQ(errno, EDOM);
    const std::unique_ptr<if_dev, decltype(&if_dev_close)> dev(opened, if_dev_close);

    std::array<char, 3> header{'h', 'd', 'r'}, read_header{};
    std::array<char, 4> payload{'d', 'a', 't', 'a'}, read_payload{};
    const if_io write{header.data(), header.size(), payload.data(), payload.size(), 113};
    const if_io read{read_header.data(), read_header.size(), read_payload.data(), read_payload.size(), 113};
    ASSERT_EQ(if_dev_write(dev.get(), &write, 1), 0);
    ASSERT_EQ(if_dev_flush(dev.get()), 0);
    ASSERT_EQ(if_dev_read(dev.get(), &read, 1), 0);
    EXPECT_EQ(read_header, header);
    EXPECT_EQ(read_payload, payload);
    EXPECT_EQ(errno, EDOM);
}

TEST(StorageIO, FixedBuffersMixedIovAndShortRead) {
    char name[] = "/tmp/infer-c-uring-XXXXXX";
    const int fd = mkstemp(name);
    ASSERT_GE(fd, 0);
    const auto close_fd = [](const int* p) { close(*p); };
    const std::unique_ptr<const int, decltype(close_fd)> file(&fd, close_fd);
    ASSERT_EQ(unlink(name), 0);

    alignas(4096) std::array<char, 8192> input{}, output{};
    input.fill('a');
    std::fill(input.begin() + 4096, input.end(), 'b');
    const iovec buffers[] = {{input.data(), input.size()}, {output.data(), output.size()}};
    if_dev* opened = nullptr;
    const int rc = if_dev_open_uring_fd(fd, 16384, 0, 4, buffers, 2, &opened);
    if (ring_unavailable(-rc))
        GTEST_SKIP() << std::strerror(-rc);
    ASSERT_EQ(rc, 0) << std::strerror(-rc);
    const std::unique_ptr<if_dev, decltype(&if_dev_close)> dev(opened, if_dev_close);

    // Mixed fixed and vectored requests, submitted out of disk-offset order.
    const if_io writes[] = {{nullptr, 0, input.data() + 4096, 4096, 4096},
                            {input.data(), 1024, input.data() + 1024, 3072, 0}};
    ASSERT_EQ(if_dev_write(dev.get(), writes, 2), 0);
    ASSERT_EQ(if_dev_flush(dev.get()), 0);
    const if_io reads[] = {{output.data(), 2048, output.data() + 2048, 2048, 0},
                           {nullptr, 0, output.data() + 4096, 4096, 4096}};
    ASSERT_EQ(if_dev_read(dev.get(), reads, 2), 0);
    EXPECT_EQ(input, output);

    // First CQE transfers 4 KiB of the 8 KiB request, then the retry hits EOF.
    // The peer's completion must still be drained and visible on return.
    alignas(4096) std::array<char, 8192> short_read{};
    output.fill(0);
    const if_io eof[] = {{nullptr, 0, short_read.data(), short_read.size(), 4096},
                         {nullptr, 0, output.data(), 4096, 0}};
    EXPECT_EQ(if_dev_read(dev.get(), eof, 2), -EIO);
    EXPECT_EQ(std::memcmp(short_read.data(), input.data() + 4096, 4096), 0);
    EXPECT_EQ(std::memcmp(output.data(), input.data(), 4096), 0);
    ASSERT_EQ(if_dev_read(dev.get(), reads, 2), 0);
    EXPECT_EQ(input, output);
}

TEST(StorageIO, FixedBuffersDirectIOAcrossSubmissionWaves) {
    char name[] = "/tmp/infer-c-direct-XXXXXX";
    const int temporary = mkstemp(name);
    ASSERT_GE(temporary, 0);
    const int fd = open(name, O_RDWR | O_DIRECT | O_CLOEXEC);
    const int error = errno;
    close(temporary);
    unlink(name);
    if (fd < 0 && ring_unavailable(error))
        GTEST_SKIP() << std::strerror(error);
    ASSERT_GE(fd, 0) << std::strerror(error);
    const auto close_fd = [](const int* p) { close(*p); };
    const std::unique_ptr<const int, decltype(close_fd)> file(&fd, close_fd);
    constexpr std::size_t pages = 128;
    constexpr std::size_t bytes = pages * io_alignment;
    ASSERT_EQ(ftruncate(fd, bytes), 0);
    CPUDeviceAllocator cpu;
    const auto release = [&](void* p) { cpu.release(p); };
    const std::unique_ptr<void, decltype(release)> source(cpu.allocate_io(bytes), release);
    const std::unique_ptr<void, decltype(release)> destination(cpu.allocate_io(bytes), release);
    const iovec buffers[] = {{source.get(), bytes}, {destination.get(), bytes}};
    if_dev* opened = nullptr;
    const auto rc = if_dev_open_uring_fd(fd, bytes, 0, 8, buffers, 2, &opened);
    if (ring_unavailable(-rc))
        GTEST_SKIP() << std::strerror(-rc);
    ASSERT_EQ(rc, 0) << std::strerror(-rc);
    const std::unique_ptr<if_dev, decltype(&if_dev_close)> device(opened, if_dev_close);
    std::vector<if_io> writes, reads;
    for (std::size_t i = 0; i < pages; ++i) {
        const auto offset = (pages - i - 1) * io_alignment;
        auto* host = static_cast<char*>(source.get()) + offset;
        std::memset(host, i, io_alignment);
        writes.push_back({nullptr, 0, host, io_alignment, offset});
        reads.push_back({nullptr, 0, static_cast<char*>(destination.get()) + offset, io_alignment, offset});
    }
    ASSERT_EQ(if_dev_write(device.get(), writes.data(), writes.size()), 0);
    ASSERT_EQ(if_dev_read(device.get(), reads.data(), reads.size()), 0);
    EXPECT_EQ(std::memcmp(source.get(), destination.get(), bytes), 0);
}
#endif

void test_chunk_migration(std::unique_ptr<SSDDevice> device) {
    constexpr std::size_t page_bytes = 64 * 1024;
    constexpr std::size_t pages = 40;
    const auto record = SSDStorageBackend::allocation_size(page_bytes, device->alignment());
    auto* physical = device.get();
    StorageConfig config;
    config.block_bytes = page_bytes;
    config.budget = {pages * page_bytes, 0, pages * record};
    config.retain_ssd_copy = true;
    TieredStorage storage(config, {}, {}, std::make_unique<SSDStorageBackend>(
        page_bytes, config.budget.ssd_bytes, std::move(device), SSDVerification::ON_READ));
    std::vector<BlockId> ids;
    for (std::size_t i = 0; i < pages; ++i) {
        ids.push_back(storage.create(page_bytes));
        auto lease = storage.acquire(ids.back(), AccessMode::WRITE);
        std::memset(lease.writable_data(), i + 1, page_bytes);
        lease.finish();
    }
    const auto written = storage.migrate_batch_async(ids, StorageLocation::SSD).get();
    for (const auto& result : written)
        ASSERT_TRUE(result.committed);
    const auto stats = physical->stats();
    EXPECT_LT(stats.completed_ios, pages / 2);
    EXPECT_EQ(stats.written_bytes, pages * record);
    const auto file_stats = physical->file_io_stats();
    if (file_stats.uring_available)
        EXPECT_GE(file_stats.peak_inflight, 2u);
    EXPECT_EQ(storage.stats(StorageLocation::SSD).resident_bytes, pages * record);

    // Independent lookup of a non-leading record in the first chunk.
    std::vector<unsigned char> page(page_bytes);
    storage.read_to_host(ids[4], page.data(), page.size());
    EXPECT_TRUE(std::all_of(page.begin(), page.end(), [](auto byte) { return byte == 5; }));

    std::reverse(ids.begin(), ids.end());
    const auto read = storage.migrate_batch_async(ids, StorageLocation::CPU).get();
    for (std::size_t i = 0; i < pages; ++i) {
        ASSERT_TRUE(read[i].committed);
        auto lease = storage.acquire(ids[i]);
        const auto* bytes = static_cast<const unsigned char*>(lease.data());
        EXPECT_TRUE(std::all_of(bytes, bytes + lease.size(),
                                [&](auto byte) { return byte == pages - i; }));
        lease.finish();
    }
    EXPECT_EQ(storage.stats(StorageLocation::SSD).replica_bytes, pages * record);

    // Rewrite one stale retained replica in place. Its neighbors still map to
    // the same physical records and a later CRC error must affect only one page.
    {
        auto lease = storage.acquire(ids[3], AccessMode::WRITE);
        std::memset(lease.writable_data(), 77, lease.size());
        lease.finish();
    }
    for (const auto& result : storage.migrate_batch_async(ids, StorageLocation::SSD).get())
        ASSERT_TRUE(result.committed);
    storage.read_to_host(ids[3], page.data(), page.size());
    EXPECT_TRUE(std::all_of(page.begin(), page.end(), [](auto byte) { return byte == 77; }));
    const unsigned char damaged = 0;
    physical->write(&damaged, 1, 4 * record + 4096);
    const auto recovered = storage.migrate_batch_async(ids, StorageLocation::CPU).get();
    for (std::size_t i = 0; i < pages; ++i) {
        EXPECT_EQ(recovered[i].committed, i != pages - 5);
        if (i == pages - 5)
            EXPECT_TRUE(recovered[i].error);
    }
    EXPECT_EQ(storage.stats(StorageLocation::CPU).migration_bytes, 0u);
    EXPECT_EQ(storage.stats(StorageLocation::SSD).resident_bytes, record);
}

TEST(StorageIO, DirectFileChunkMigrationAndCRC) {
#ifndef TEST_DIRECT_FILE
    GTEST_SKIP() << "DirectFile requires liburing";
#else
    std::unique_ptr<SSDDevice> device;
    try {
        device = std::make_unique<io::DirectFile>("/tmp", 8 * 1024 * 1024);
    } catch (const std::system_error& error) {
        if (ring_unavailable(error.code().value()))
            GTEST_SKIP() << error.what();
        throw;
    }
    test_chunk_migration(std::move(device));
#endif
}

SpdkBdevConfig malloc_bdev() {
    SpdkBdevConfig config;
    config.json_config = INFER_SPDK_MALLOC_JSON;
    config.bdev_name = "Malloc0";
    config.capacity_bytes = 32 * 1024 * 1024;
    config.no_huge = config.no_pci = true;
    config.io_buffer_bytes = 512 * 1024;
    config.dma_budget_bytes = 16 * 1024 * 1024;
    return config;
}

TEST(StorageIO, SpdkMallocChunkMigrationAndCRC) {
    if (!spdk_available())
        GTEST_SKIP() << "SPDK was omitted from this build";
    test_chunk_migration(open_spdk_device(malloc_bdev()));
}

#if INFER_ENABLE_CUDA
void test_paged_gpu(StorageConfig config) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    constexpr std::size_t page_bytes = 64 * 1024;
    constexpr std::size_t pages = 8;
    CachingPinnedHostAllocator pinned;
    auto* host = pinned.allocate_io(4097);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(host) % io_alignment, 0u);
    cudaPointerAttributes attributes{};
    EXPECT_EQ(cudaPointerGetAttributes(&attributes, host), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    pinned.release(host);

    config.block_bytes = page_bytes;
    config.budget = {pages * page_bytes, pages * page_bytes, 2 * 1024 * 1024};
    config.ssd_verification = SSDVerification::ON_READ;
    std::unique_ptr<PagedKV> kv;
    try {
        kv = std::make_unique<PagedKV>(config, 1, 1, "storage-io");
    } catch (const std::system_error& error) {
        if (ring_unavailable(error.code().value()))
            GTEST_SKIP() << error.what();
        throw;
    }
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    const auto destroy_stream = [](cudaStream_t stream) { cudaStreamDestroy(stream); };
    const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(destroy_stream)>
        stream_owner(stream, destroy_stream);
    PagedKV::Sequence sequence;
    for (std::size_t i = 0; i < pages; ++i) {
        kv->append_page(sequence);
        auto lease = kv->acquire(sequence, i, 0, AccessMode::WRITE, stream);
        ASSERT_EQ(cudaMemsetAsync(lease.writable_data(), i + 1, page_bytes, stream), cudaSuccess);
        lease.finish();
    }
    kv->prefetch_layer(sequence, 0, StorageLocation::SSD).wait();
    EXPECT_LT(kv->storage().io_stats(StorageLocation::SSD).device.completed_ios, pages);
    kv->prefetch_layer(sequence, 0, StorageLocation::GPU).wait();
    std::vector<unsigned char> data(page_bytes);
    for (std::size_t i = 0; i < pages; ++i) {
        auto lease = kv->acquire(sequence, i, 0, AccessMode::READ, stream);
        ASSERT_EQ(cudaMemcpyAsync(data.data(), lease.data(), data.size(), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        lease.finish();
        EXPECT_TRUE(std::all_of(data.begin(), data.end(), [&](auto byte) { return byte == i + 1; }));
    }
}

TEST(StorageIO, PagedKVGpuDirectFileRoundTrip) {
#ifndef TEST_DIRECT_FILE
    GTEST_SKIP() << "DirectFile requires liburing";
#else
    StorageConfig config;
    config.ssd_direct = true;
    test_paged_gpu(config);
#endif
}

TEST(StorageIO, PagedKVGpuSpdkMallocRoundTrip) {
    if (!spdk_available())
        GTEST_SKIP() << "SPDK was omitted from this build";
    StorageConfig config;
    config.ssd_spdk = malloc_bdev();
    // Reserve the entire DMA budget for bdev bounce slots to exercise the fallback.
    config.ssd_spdk.dma_budget_bytes = config.ssd_spdk.queue_depth * config.ssd_spdk.io_buffer_bytes;
    test_paged_gpu(config);
}

TEST(StorageIO, PagedKVGpuSpdkSharedRoundTrip) {
    if (!spdk_available())
        GTEST_SKIP() << "SPDK was omitted from this build";
    StorageConfig config;
    config.ssd_spdk = malloc_bdev();
    test_paged_gpu(config);
}

TEST(StorageIO, SpdkCudaSharedBuffer) {
    if (!spdk_available())
        GTEST_SKIP() << "SPDK was omitted from this build";
    auto device = open_spdk_device(malloc_bdev());
    auto buffer = device->allocate_transfer_buffer(io_alignment);
    if (!buffer)
        GTEST_SKIP() << "CUDA cannot register this SPDK DMA arena";
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, buffer->data()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    std::memset(buffer->data(), 42, buffer->size());
    device->write(buffer->data(), buffer->size(), 0);
    std::memset(buffer->data(), 0, buffer->size());
    device->read(buffer->data(), buffer->size(), 0);
    const auto* bytes = static_cast<const unsigned char*>(buffer->data());
    EXPECT_TRUE(std::all_of(bytes, bytes + buffer->size(), [](auto byte) { return byte == 42; }));
    EXPECT_EQ(device->stats().direct_written_bytes, buffer->size());
    EXPECT_EQ(device->stats().direct_read_bytes, buffer->size());

    constexpr std::size_t pages = 16;
    auto source = device->allocate_transfer_buffer(pages * io_alignment);
    auto destination = device->allocate_transfer_buffer(pages * io_alignment);
    ASSERT_TRUE(source);
    ASSERT_TRUE(destination);
    auto* physical = device.get();
    SSDStorageBackend storage(io_alignment, pages * 2 * io_alignment,
                              std::move(device), SSDVerification::ON_READ);
    std::vector<StorageWrite> writes;
    std::vector<StorageRead> reads;
    for (std::size_t i = 0; i < pages; ++i) {
        const auto handle = storage.reserve();
        BlockMeta meta;
        meta.id = i + 1;
        meta.version = 1;
        meta.size = io_alignment;
        auto* host = static_cast<char*>(source->data()) + i * io_alignment;
        std::memset(host, i + 1, io_alignment);
        writes.push_back({handle, meta, host});
        reads.push_back({handle, meta, static_cast<char*>(destination->data()) + i * io_alignment});
    }
    const auto before = physical->stats();
    storage.write_published_many(writes);
    storage.read_validated_many(reads);
    const auto after = physical->stats();
    EXPECT_EQ(std::memcmp(source->data(), destination->data(), source->size()), 0);
    EXPECT_EQ(after.completed_ios - before.completed_ios, 2u);
    EXPECT_EQ(after.direct_written_bytes - before.direct_written_bytes, source->size());
    EXPECT_EQ(after.direct_read_bytes - before.direct_read_bytes, destination->size());
    EXPECT_EQ(after.copied_write_bytes - before.copied_write_bytes, pages * io_alignment);
    EXPECT_EQ(after.copied_read_bytes - before.copied_read_bytes,
              pages * SSDStorageBackend::record_header_bytes);
}
#endif
} // namespace
