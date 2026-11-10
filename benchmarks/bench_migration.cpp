#include "bench_common.h"
#include "infer/storage/tiered_storage.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const auto args = bench::arguments(argc, argv);
        const auto destination = args.at("target");
        if (destination != "ssd" && destination != "gpu")
            throw std::invalid_argument("target must be ssd or gpu");
        const auto target = destination == "ssd" ? infer::StorageLocation::SSD
                                                 : infer::StorageLocation::GPU;
        const auto batch = bench::number(args, "batch", 16);
        const auto rounds = bench::number(args, "rounds", 100);
        const auto bytes = bench::number(args, "bytes", 65536);
        if (!batch || batch > 64 || !rounds || (bytes != 65536 && bytes != 262144))
            throw std::invalid_argument("Use batch=1..64, rounds>0 and bytes=65536/262144");
        infer::StorageConfig config;
        config.block_bytes = bytes;
        config.budget.cpu_bytes = batch * bytes;
        if (target == infer::StorageLocation::GPU) {
            config.cpu_memory = infer::HostMemoryKind::PINNED;
            config.budget.gpu_bytes = batch * bytes;
        } else {
            config.budget.ssd_bytes = batch * (bytes + 4096);
            config.ssd_directory = args.at("directory");
            config.ssd_io.mode = infer::FileIOMode::POSIX;
        }
        infer::TieredStorage storage(config);
        std::vector<infer::BlockId> ids;
        for (std::size_t i = 0; i < batch; ++i) {
            const auto id = storage.create(bytes);
            ids.push_back(id);
            auto lease = storage.acquire(id, infer::AccessMode::WRITE);
            std::memset(lease.writable_data(), i + 1, bytes);
            lease.finish();
        }
        std::ofstream out(args.at("output") + ".csv");
        out << "direction,round,blocks,block_bytes,latency_us,payload_GB_s\n" << std::setprecision(12);
        for (std::size_t round = 0; round <= rounds; ++round) {
            for (const auto tier : {target, infer::StorageLocation::CPU}) {
                const auto start = bench::Clock::now();
                const auto results = storage.migrate_batch_async(ids, tier).get();
                const auto seconds = bench::seconds(start);
                for (const auto& result : results) {
                    if (result.error) std::rethrow_exception(result.error);
                    if (!result.committed || result.cleanup_pending)
                        throw std::runtime_error("Incomplete migration");
                }
                if (round)
                    out << (tier == target ? "cpu_to_target" : "target_to_cpu") << ',' << round
                        << ',' << batch << ',' << bytes << ',' << seconds * 1e6 << ','
                        << (batch * bytes) / seconds / 1e9 << '\n';
            }
            for (std::size_t i = 0; i < batch; ++i) {
                auto lease = storage.acquire(ids[i]);
                const auto* data = static_cast<const unsigned char*>(lease.data());
                if (!std::all_of(data, data + bytes, [i](unsigned char value) { return value == i + 1; }))
                    throw std::runtime_error("Migration payload mismatch");
            }
        }
        if (!out) throw std::runtime_error("Cannot write migration results");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
