#include "bench_common.h"
#include "infer/io/direct_file.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <system_error>
#include <unistd.h>

namespace {
void require(bool success, const char* operation) {
    if (!success) throw std::system_error(errno, std::generic_category(), operation);
}
struct File {
    int descriptor = -1;
    ~File() { if (descriptor >= 0) close(descriptor); }
};
}

int main(int argc, char** argv) {
    try {
        const auto args = bench::arguments(argc, argv);
        const auto backend = args.at("backend");
#ifndef BENCH_HAS_URING
        if (backend == "uring") { std::cerr << "liburing not compiled\n"; return 77; }
#endif
        const auto block = bench::number(args, "block", 4096);
        const auto bytes = bench::number(args, "bytes", 64 * 1024 * 1024);
        const auto depth = bench::number(args, "depth", 8);
        const auto rounds = bench::number(args, "rounds", 2);
        const auto seed = bench::number(args, "seed", 42);
        if (!block || block % 4096 || !bytes || bytes % block || !depth || depth > 128 || !rounds)
            throw std::invalid_argument("Use 4KiB-aligned blocks, a multiple dataset, depth 1..128, rounds > 0");
        const bool posix = backend == "posix-buffered" || backend == "posix-direct";
        const bool spdk = backend == "spdk";
        if (!posix && !spdk && backend != "uring") throw std::invalid_argument("Unknown storage backend");
        std::unique_ptr<infer::SSDDevice> device;
        if (spdk) {
            if (args.at("allow-spdk-write") != "YES")
                throw std::invalid_argument("SPDK requires --allow-spdk-write YES and an exclusively owned region");
            if (!infer::spdk_available()) { std::cerr << "SPDK not compiled\n"; return 77; }
            infer::SpdkBdevConfig config;
            config.json_config = args.at("spdk-config");
            config.bdev_name = args.at("spdk-bdev");
            // No implicit offset, PCI binding, device discovery or formatting.
            config.offset_bytes = bench::number(args, "spdk-offset", UINT64_MAX);
            if (config.offset_bytes == UINT64_MAX || config.offset_bytes % 4096)
                throw std::invalid_argument("Explicit aligned SPDK offset required");
            config.capacity_bytes = bytes;
            config.queue_depth = depth;
            config.io_buffer_bytes = block;
            config.dma_budget_bytes = bytes + depth * block + depth * ((block + 4095) / 4096 * 4096);
            config.environment_mib = bench::number(args, "spdk-memory-mib", 512);
            config.reactor_mask = args.count("spdk-mask") ? args.at("spdk-mask") : "0x1";
            config.no_pci = bench::number(args, "spdk-no-pci", 0);
            config.no_huge = bench::number(args, "spdk-no-huge", 0);
            device = infer::open_spdk_device(config);
        }
        // Fault and initialize buffers outside the timed region. SPDK uses its DMA arena.
        auto source_dma = spdk ? device->allocate_dma_buffer(bytes) : nullptr;
        auto target_dma = spdk ? device->allocate_dma_buffer(depth * block) : nullptr;
        const auto aligned = [](std::size_t size) {
            void* pointer = nullptr;
            const auto error = posix_memalign(&pointer, 4096, size);
            if (error) throw std::system_error(error, std::generic_category(), "posix_memalign");
            return pointer;
        };
        std::unique_ptr<void, decltype(&std::free)> source_heap(spdk ? nullptr : aligned(bytes), &std::free);
        std::unique_ptr<void, decltype(&std::free)> target_heap(spdk ? nullptr : aligned(depth * block), &std::free);
        auto* source = static_cast<char*>(spdk ? source_dma->data() : source_heap.get());
        auto* target = static_cast<char*>(spdk ? target_dma->data() : target_heap.get());
        std::uint64_t random = seed;
        for (std::size_t offset = 0; offset < bytes; offset += sizeof(std::uint64_t)) {
            const auto value = bench::random(random);
            std::memcpy(source + offset, &value, sizeof(value));
        }
        std::memset(target, 0, depth * block);
        File file;
        if (posix) {
            auto name = args.at("directory") + "/miniinfer-benchmark-XXXXXX";
            file.descriptor = mkstemp(name.data());
            require(file.descriptor >= 0, "mkstemp");
            const auto unlinked = unlink(name.c_str());
            require(unlinked == 0, "unlink private benchmark file");
            if (backend == "posix-direct")
                require(fcntl(file.descriptor, F_SETFL, fcntl(file.descriptor, F_GETFL) | O_DIRECT) == 0, "O_DIRECT");
            const auto error = posix_fallocate(file.descriptor, 0, bytes);
            if (error) throw std::system_error(error, std::generic_category(), "posix_fallocate");
        }
        if (backend == "uring") {
            infer::io::DirectFileConfig config;
            config.queue_depth = depth;
            config.sqpoll = config.iopoll = false;
            config.buffers = {{source, bytes}, {target, depth * block}};
            device = std::make_unique<infer::io::DirectFile>(args.at("directory"), bytes, config);
        }
        const auto transfer = [&](bool write, const std::vector<infer::FileIORequest>& requests) {
            if (device) {
                if (write) { device->write_many(requests); return; }
                device->read_many(requests);
                return;
            }
            for (const auto& request : requests) {
                std::size_t completed = 0;
                while (completed < request.bytes) {
                    const auto result = write
                        ? pwrite(file.descriptor, static_cast<char*>(request.data) + completed,
                                 request.bytes - completed, request.offset + completed)
                        : pread(file.descriptor, static_cast<char*>(request.data) + completed,
                                request.bytes - completed, request.offset + completed);
                    if (result < 0 && errno == EINTR) continue;
                    require(result >= 0, write ? "pwrite" : "pread");
                    if (!result) throw std::runtime_error("Zero-length I/O completion");
                    completed += result;
                }
            }
        };
        std::vector<std::size_t> order(bytes / block);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 engine(seed);
        if (args.at("pattern") != "random" && args.at("pattern") != "sequential")
            throw std::invalid_argument("Unknown pattern");
        if (args.at("pattern") == "random") std::shuffle(order.begin(), order.end(), engine);
        // Populate the entire extent before reads; never benchmark holes or unwritten sectors.
        for (std::size_t i = 0; i < order.size(); i += depth) {
            std::vector<infer::FileIORequest> requests;
            for (std::size_t j = i; j < std::min(i + depth, order.size()); ++j)
                requests.push_back({source + order[j] * block, block, order[j] * block});
            transfer(true, requests);
        }
        if (posix) require(fsync(file.descriptor) == 0, "initialization fsync");
        std::ofstream out(args.at("output") + ".csv");
        out << "backend,operation,round,block_bytes,dataset_bytes,requested_depth,effective_depth,pattern,seconds,GB_s,IOPS,operations,device_completions,setup_flags\n";
        out << std::setprecision(12);
        for (std::size_t round = 0; round < rounds; ++round) {
            // Alternate order to reduce a fixed write-before-read bias.
            for (bool write : {bool(round % 2), !bool(round % 2)}) {
                const auto before = device ? device->stats().completed_ios : 0;
                double seconds = 0;
                for (std::size_t i = 0; i < order.size(); i += depth) {
                    std::vector<infer::FileIORequest> requests;
                    for (std::size_t j = i; j < std::min(i + depth, order.size()); ++j)
                        requests.push_back({write ? source + order[j] * block : target + (j - i) * block,
                                            block, order[j] * block});
                    const auto start = bench::Clock::now();
                    transfer(write, requests);
                    seconds += bench::seconds(start);
                    if (!write)
                        for (std::size_t j = 0; j < requests.size(); ++j)
                            if (std::memcmp(requests[j].data, source + requests[j].offset, block))
                                throw std::runtime_error("Storage read verification failed");
                }
                const auto io = device ? device->file_io_stats() : infer::FileIOStats{};
                out << backend << ',' << (write ? "write" : "read") << ',' << round << ','
                    << block << ',' << bytes << ',' << depth << ',' << (posix ? 1 : std::min(depth, order.size())) << ','
                    << args.at("pattern") << ',' << seconds << ',' << bytes / seconds / 1e9 << ','
                    << order.size() / seconds << ',' << order.size() << ','
                    << (device ? std::to_string(device->stats().completed_ios - before) : "") << ','
                    << io.setup_flags << '\n';
            }
        }
        if (!out) throw std::runtime_error("Cannot write storage results");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
