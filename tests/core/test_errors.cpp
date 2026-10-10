#include "mini_infer/core/mini_infer_tensor.h"
#include "mini_infer/core/mini_infer_memory.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace mini_infer;
template<class F> void rejects(F action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid operation was accepted");
}
struct FailingRelease final : DeviceAllocator {
    FailingRelease() : DeviceAllocator(DeviceType::CPU) {}
    void* allocate(std::size_t n) const override { return std::malloc(n); }
    void release(void* p) const override { std::free(p); throw std::runtime_error("release failed"); }
};
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("one test case required");
        const std::string test = argv[1];
        auto cpu = std::make_shared<CPUDeviceAllocator>();
        int source = 7, destination = 0;
        if (test == "null_copy") rejects([&] { cpu->copy(nullptr, &destination, 4, MemcpyKind::CPU2CPU); });
        else if (test == "null_buffer") rejects([&] { Buffer b(4, std::shared_ptr<DeviceAllocator>{}); });
        else if (test == "null_borrow") rejects([&] { Buffer b(4, nullptr, DeviceType::CPU); });
        else if (test == "null_zero") rejects([&] { cpu->memset_zero(nullptr, 4); });
        else if (test == "bad_copy_kind") rejects([&] { cpu->copy(&source, &destination, 4, static_cast<MemcpyKind>(99)); });
        else if (test == "bad_pool_config") rejects([&] { CachingCUDAAllocatorConfig c; c.small_block_threshold = 0; CachingCUDADeviceAllocator pool(c); });
        else if (test == "foreign_pool_pointer") rejects([&] { CachingCUDADeviceAllocator pool; pool.release(&source); });
        else if (test == "cuda_error") rejects([] { check_cuda(cudaErrorInvalidValue, "regression"); });
        else if (test == "cpu_sync") {
            cpu->copy(&source, &destination, 4, MemcpyKind::CPU2CPU, nullptr, true);
            if (destination != 7) throw std::runtime_error("CPU copy failed");
        } else if (test == "no_stdout") {
            std::ostringstream captured;
            auto* previous = std::cout.rdbuf(captured.rdbuf());
            { Buffer b(16, cpu); }
            std::cout.rdbuf(previous);
            if (!captured.str().empty()) throw std::runtime_error("allocator polluted standard output");
        } else if (test == "destructor") {
            Buffer b(16, std::make_shared<FailingRelease>());
        } else if (test == "tensor_move") {
            Tensor source_tensor(DataType::FLOAT32, {2}, cpu);
            source_tensor.ptr<float>()[0] = 42;
            Tensor owner(std::move(source_tensor));
            if (!source_tensor.empty() || source_tensor.size() != 0 || !source_tensor.dims().empty())
                throw std::runtime_error("moved-from Tensor has stale metadata");
            rejects([&] { source_tensor.allocate(cpu); });
            Tensor assigned;
            assigned = std::move(owner);
            if (owner.size() != 0 || !owner.dims().empty() || assigned.ptr<float>()[0] != 42)
                throw std::runtime_error("Tensor move assignment lost its invariant");
            rejects([&] { owner.allocate(cpu); });
            assigned = std::move(assigned);
            if (assigned.ptr<float>()[0] != 42) throw std::runtime_error("self move lost data");
            Tensor shared = assigned;
            if (!shared.overlaps(assigned)) throw std::runtime_error("copy no longer shares storage");
        } else throw std::invalid_argument("unknown test case");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
