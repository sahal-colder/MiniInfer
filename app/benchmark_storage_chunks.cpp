#include "infer/io/direct_file.h"
#include "infer/storage/storage_backend.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string_view>

namespace {
using namespace infer;
constexpr std::size_t page_bytes = 4096;
constexpr std::size_t pages_per_batch = 64;
constexpr std::size_t page_count = 2048;
constexpr std::size_t capacity = 32 * 1024 * 1024;
constexpr int repetitions = 6;

struct Measurement {
    double seconds = 0;
    std::uint64_t completed = 0;
};

void print(const char* backend, const char* mode, const char* operation, Measurement sample) {
    const auto pages = double(page_count) * repetitions;
    std::printf("%s,%s,%s,%.3f,%.0f,%.0f,%llu\n", backend, mode, operation,
                pages * page_bytes / sample.seconds / 1e9, pages / sample.seconds,
                sample.completed / sample.seconds, static_cast<unsigned long long>(sample.completed));
}

void run(const char* name, std::unique_ptr<SSDDevice> device) {
    const auto record = SSDStorageBackend::allocation_size(page_bytes, device->alignment());
    SSDStorageBackend storage(page_bytes, page_count * record, std::move(device),
                              SSDVerification::ON_READ);
    CPUDeviceAllocator allocator;
    const auto release = [&](void* p) { allocator.release(p); };
    std::unique_ptr<void, decltype(release)> source(allocator.allocate_io(page_count * page_bytes), release);
    std::unique_ptr<void, decltype(release)> destination(allocator.allocate_io(page_count * page_bytes), release);

    std::vector<StorageWrite> writes;
    std::vector<StorageRead> reads;
    for (std::size_t i = 0; i < page_count; ++i) {
        const auto handle = storage.reserve();
        BlockMeta meta;
        meta.id = i + 1;
        meta.version = 1;
        meta.size = page_bytes;
        auto* data = static_cast<char*>(source.get()) + i * page_bytes;
        std::memset(data, i % 251, page_bytes);
        writes.push_back({handle, meta, data});
        reads.push_back({handle, meta, static_cast<char*>(destination.get()) + i * page_bytes});
    }

    std::mt19937 random(123);
    std::vector<std::size_t> order(page_count / pages_per_batch);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), random);
    std::vector<std::vector<StorageWrite>> write_batches;
    std::vector<std::vector<StorageRead>> read_batches;
    for (auto group : order) {
        std::vector<std::size_t> pages(pages_per_batch);
        std::iota(pages.begin(), pages.end(), group * pages_per_batch);
        std::shuffle(pages.begin(), pages.end(), random);
        write_batches.emplace_back();
        read_batches.emplace_back();
        for (auto i : pages) {
            write_batches.back().push_back(writes[i]);
            read_batches.back().push_back(reads[i]);
        }
    }
    storage.write_published(writes[0].handle, writes[0].meta, writes[0].host);
    storage.read_validated(reads[0].handle, reads[0].expected, reads[0].host);

    Measurement measurements[2][2];
    const auto measure = [&](bool batched, bool write) {
        const auto before = storage.io_stats().device.completed_ios;
        const auto start = std::chrono::steady_clock::now();
        if (write) {
            for (const auto& batch : write_batches)
                if (batched)
                    storage.write_published_many(batch);
                else
                    for (const auto& r : batch)
                        storage.write_published(r.handle, r.meta, r.host);
        } else {
            for (const auto& batch : read_batches)
                if (batched)
                    storage.read_validated_many(batch);
                else
                    for (const auto& r : batch)
                        storage.read_validated(r.handle, r.expected, r.host);
        }
        auto& result = measurements[batched][write];
        result.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result.completed += storage.io_stats().device.completed_ios - before;
    };
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        for (bool batched : {bool(repetition % 2), !bool(repetition % 2)}) {
            measure(batched, true);
            measure(batched, false);
        }
    }
    if (std::memcmp(source.get(), destination.get(), page_count * page_bytes))
        throw std::runtime_error("Benchmark payload mismatch");

    std::puts("backend,mode,operation,GB_per_s,page_IOPS,device_IOPS,device_completions");
    for (bool batched : {false, true})
        for (bool write : {true, false})
            print(name, batched ? "chunk" : "single", write ? "write" : "read",
                  measurements[batched][write]);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && argc <= 3 && std::string_view(argv[1]) == "uring") {
            run("io_uring", std::make_unique<infer::io::DirectFile>(argc == 3 ? argv[2] : "/tmp", capacity));
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "spdk-malloc") {
            infer::SpdkBdevConfig config;
            config.bdev_name = "Malloc0";
            config.json_config = INFER_SPDK_MALLOC_JSON;
            config.capacity_bytes = capacity;
            config.no_huge = config.no_pci = true;
            config.io_buffer_bytes = 512 * 1024;
            config.dma_budget_bytes = 16 * 1024 * 1024;
            run("spdk-malloc", infer::open_spdk_device(config));
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "spdk") {
            infer::SpdkBdevConfig config;
            config.json_config = argv[2];
            config.bdev_name = argv[3];
            const std::string_view offset = argv[4];
            const auto parsed = std::from_chars(offset.begin(), offset.end(), config.offset_bytes);
            if (parsed.ec != std::errc{} || parsed.ptr != offset.end())
                throw std::invalid_argument("Expected an explicit byte offset in the bdev");
            config.reactor_mask = argv[5];
            config.capacity_bytes = capacity;
            config.io_buffer_bytes = 512 * 1024;
            config.dma_budget_bytes = 16 * 1024 * 1024;
            run("spdk", infer::open_spdk_device(config));
            return 0;
        }
        std::fprintf(stderr, "Usage: %s uring [directory] | spdk-malloc | "
                             "spdk <json> <bdev> <offset-bytes> <reactor-mask>\n"
                             "The SPDK benchmark writes to its configured 32 MiB region.\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
