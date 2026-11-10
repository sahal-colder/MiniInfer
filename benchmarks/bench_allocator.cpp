#include "bench_common.h"
#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/pinned_memory_pool.h"
#include "infer/cuda/cuda_memory_pool.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {
struct Sample {
    std::size_t step, bytes;
    bool allocate;
    double microseconds;
};
struct Slot { void* pointer = nullptr; std::size_t bytes = 0; };
struct Counters { std::size_t upstream, reused, central; };
#if INFER_ENABLE_CUDA
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
class RawCUDA final : public infer::DeviceAllocator {
  public:
    RawCUDA() : DeviceAllocator(infer::DeviceType::CUDA) {}
    void* allocate(std::size_t bytes, cudaStream_t = nullptr) const override {
        void* pointer = nullptr;
        cuda_check(cudaMalloc(&pointer, bytes));
        return pointer;
    }
    void release(void* pointer, cudaStream_t = nullptr) const override { cuda_check(cudaFree(pointer)); }
};
#endif
}

int main(int argc, char** argv) {
    try {
        const auto args = bench::arguments(argc, argv);
        const auto backend = args.at("backend");
        const auto threads = bench::number(args, "threads", 4);
        const auto steps = bench::number(args, "steps", 10000);
        const auto warmup = bench::number(args, "warmup", 1000);
        const auto slots = bench::number(args, "slots", 16);
        const auto seed = bench::number(args, "seed", 42);
        const auto tls = bench::number(args, "tls", 1);
        if (!threads || !steps || !slots) throw std::invalid_argument("threads/steps/slots must be positive");
        if (tls > 1) throw std::invalid_argument("--tls must be 0 or 1");
        std::unique_ptr<infer::DeviceAllocator> allocator;
        if (backend == "cpu-pool") {
            infer::CachingCPUAllocatorConfig config;
            config.enable_thread_cache = tls != 0;
            allocator = std::make_unique<infer::CachingCPUDeviceAllocator>(config);
        }
        if (backend == "cpu-raw") allocator = std::make_unique<infer::CPUDeviceAllocator>();
#if INFER_ENABLE_CUDA
        if (backend == "cuda-raw") allocator = std::make_unique<RawCUDA>();
        if (backend == "cuda-pool") allocator = std::make_unique<infer::CachingCUDADeviceAllocator>();
        if (backend == "pinned-pool") allocator = std::make_unique<infer::CachingPinnedHostAllocator>();
        if (backend == "pinned-raw") allocator = std::make_unique<infer::PinnedHostAllocator>();
        if (allocator && backend.rfind("cpu-", 0)) {
            cuda_check(cudaSetDevice(0));
            cuda_check(cudaFree(nullptr));
        }
#endif
        if (!allocator) { std::cerr << "Backend unavailable: " << backend << '\n'; return 77; }
        std::atomic<std::size_t> ready{0}, finished{0};
        std::atomic<bool> start{false}, cleanup{false};
        std::vector<std::vector<Sample>> samples(threads);
        std::vector<std::size_t> requested(threads);
        std::exception_ptr failure;
        std::mutex failure_mutex;
        std::vector<std::thread> workers;
        for (std::size_t thread = 0; thread < threads; ++thread) {
            workers.emplace_back([&, thread] {
                bool admitted = false;
                try {
#if INFER_ENABLE_CUDA
                    if (backend.rfind("cpu-", 0)) cuda_check(cudaSetDevice(0));
#endif
                    std::uint64_t state = seed + thread;
                    std::vector<Slot> live(slots);
                    samples[thread].reserve(steps);
                    const auto operation = [&](std::size_t step, bool measured) {
                        const auto value = bench::random(state);
                        auto& slot = live[(value >> 32) % slots];
                        if (slot.pointer) {
                            const auto begin = bench::Clock::now();
                            allocator->release(slot.pointer);
                            const auto us = bench::seconds(begin) * 1e6;
                            if (measured) samples[thread].push_back({step, slot.bytes, false, us});
                            slot = {};
                            return;
                        }
                        const auto bytes = bench::allocation_size(value);
                        const auto begin = bench::Clock::now();
                        slot.pointer = allocator->allocate(bytes);
                        const auto us = bench::seconds(begin) * 1e6;
                        slot.bytes = bytes;
                        if (measured) samples[thread].push_back({step, bytes, true, us});
                    };
                    for (std::size_t i = 0; i < warmup; ++i) operation(i, false);
                    ++ready;
                    admitted = true;
                    while (!start.load()) std::this_thread::yield();
                    for (std::size_t i = 0; i < steps; ++i) operation(i, true);
                    for (const auto& slot : live) requested[thread] += slot.bytes;
                    ++finished;
                    while (!cleanup.load()) std::this_thread::yield();
                    for (const auto& slot : live) if (slot.pointer) allocator->release(slot.pointer);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    failure = std::current_exception();
                    if (!admitted) ++ready;
                    ++finished;
                }
            });
        }
        while (ready.load() != threads) std::this_thread::yield();
        // Snapshot counter baseline after warmup, while every worker is paused.
        const auto counters = [&]() -> Counters {
            if (backend == "cpu-pool") {
                const auto s = static_cast<infer::CachingCPUDeviceAllocator&>(*allocator).stats();
                return {s.upstream_allocations, s.reuse_count, s.central_allocations + s.central_releases};
            }
#if INFER_ENABLE_CUDA
            if (backend == "pinned-pool") {
                const auto s = static_cast<infer::CachingPinnedHostAllocator&>(*allocator).stats();
                return {s.upstream_allocations, s.reuse_count, s.central_allocations + s.central_releases};
            }
            if (backend == "cuda-pool") {
                const auto s = static_cast<infer::CachingCUDADeviceAllocator&>(*allocator).stats();
                return {s.upstream_allocations, s.reuse_count, 0};
            }
#endif
            return {};
        };
        const auto baseline = counters();
        const auto begin = bench::Clock::now();
        start = true;
        while (finished.load() != threads) std::this_thread::yield();
        const auto wall = bench::seconds(begin);
        const auto final = counters();
        std::size_t active = 0, reserved = 0, logical = 0;
        for (auto bytes : requested) logical += bytes;
        if (backend == "cpu-pool") {
            const auto s = static_cast<infer::CachingCPUDeviceAllocator&>(*allocator).stats();
            active = s.active_bytes; reserved = s.reserved_bytes;
        }
#if INFER_ENABLE_CUDA
        if (backend == "pinned-pool") {
            const auto s = static_cast<infer::CachingPinnedHostAllocator&>(*allocator).stats();
            active = s.active_bytes; reserved = s.reserved_bytes;
        }
        if (backend == "cuda-pool") {
            const auto s = static_cast<infer::CachingCUDADeviceAllocator&>(*allocator).stats();
            active = s.active_capacity_bytes; reserved = s.reserved_bytes;
        }
#endif
        cleanup = true;
        for (auto& worker : workers) worker.join();
        if (failure) std::rethrow_exception(failure);
        std::ofstream out(args.at("output") + ".csv");
        out << "thread,step,operation,bytes,latency_us\n" << std::setprecision(12);
        for (std::size_t thread = 0; thread < threads; ++thread)
            for (const auto& s : samples[thread])
                out << thread << ',' << s.step << ',' << (s.allocate ? "allocate" : "release")
                    << ',' << s.bytes << ',' << s.microseconds << '\n';
        std::ofstream meta(args.at("output") + ".json");
        const bool pool = backend.find("pool") != std::string::npos;
        meta << "{\"backend\":" << bench::quote(backend) << ",\"wall_seconds\":" << wall
             << ",\"active_capacity_bytes\":" << (pool ? std::to_string(active) : "null")
             << ",\"reserved_bytes\":" << (pool ? std::to_string(reserved) : "null")
             << ",\"active_requested_bytes\":" << logical
             << ",\"upstream_allocations\":" << (pool ? std::to_string(final.upstream - baseline.upstream) : "null")
             << ",\"reuse_count\":" << (pool ? std::to_string(final.reused - baseline.reused) : "null")
             << ",\"cpu_tls\":" << (backend == "cpu-pool" ? std::to_string(tls) : "null")
             << ",\"central_operations\":"
             << (backend == "cpu-pool" || backend == "pinned-pool" ?
                 std::to_string(final.central - baseline.central) : "null") << "}\n";
        if (!out || !meta) throw std::runtime_error("Cannot write allocator results");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
