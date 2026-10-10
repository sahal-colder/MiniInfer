// Inject failures only at the CUDA allocation boundary; exercise the real pool.
#include "mini_infer/core/mini_infer_memory.h"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

static bool fail_next_new = false, fail_metadata = false, fail_free = false;
static int allocations = 0, frees[2] = {0, 0}, current_device = 0;
static char device_buffers[2][16];
void* operator new(std::size_t size) {
    if (fail_next_new) { fail_next_new = false; throw std::bad_alloc(); }
    if (void* p = std::malloc(size)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
extern "C" cudaError_t cudaGetDevice(int* device) { *device = current_device; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int device) { current_device = device; return cudaSuccess; }
extern "C" cudaError_t cudaMalloc(void** p, std::size_t) {
    *p = device_buffers[allocations++];
    if (fail_metadata) fail_next_new = true;
    return cudaSuccess;
}
extern "C" cudaError_t cudaFree(void* p) {
    const int index = p == device_buffers[0] ? 0 : 1;
    if (index == 1 && fail_free) { fail_free = false; return cudaErrorUnknown; }
    ++frees[index];
    return cudaSuccess;
}
namespace mini_infer {
void check_cuda(cudaError_t error, const char*) {
    if (error != cudaSuccess) throw std::runtime_error("injected CUDA error");
}
DeviceAllocator::DeviceAllocator(DeviceType d) : device_type_(d) {}
DeviceType DeviceAllocator::device_type() const { return device_type_; }
void DeviceAllocator::copy(const void*, void*, std::size_t, MemcpyKind, cudaStream_t, bool) const {}
void DeviceAllocator::memset_zero(void*, std::size_t, cudaStream_t) const {}
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        const std::string mode = argv[1];
        if (mode == "metadata") {
            {
                mini_infer::CachingCUDADeviceAllocator pool;
                fail_metadata = true;
                try { pool.allocate(16); } catch (const std::bad_alloc&) {}
                fail_next_new = false;
            }
            if (allocations != frees[0]) throw std::runtime_error("allocation leaked after bookkeeping OOM");
        } else if (mode == "trim") {
            {
                mini_infer::CachingCUDADeviceAllocator pool;
                auto* first = pool.allocate(16);
                auto* second = pool.allocate(16);
                pool.release(first); pool.release(second);
                current_device = 1;
                fail_free = true;
                bool rejected = false;
                try { pool.trim(0); } catch (const std::exception&) { rejected = true; }
                if (!rejected || current_device != 1) throw std::runtime_error("trim did not restore CUDA device after failure");
                const auto remaining = pool.stats(0);
                if (remaining.cached_bytes != 16 || remaining.cached_blocks != 1 || remaining.cuda_free_calls != 1)
                    throw std::runtime_error("failed trim corrupted pool bookkeeping");
                pool.trim(0);
                if (current_device != 1 || pool.stats(0).reserved_bytes != 0)
                    throw std::runtime_error("trim retry failed");
            }
            if (frees[0] != 1 || frees[1] != 1) throw std::runtime_error("trim freed an allocation twice");
        } else return 2;
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
