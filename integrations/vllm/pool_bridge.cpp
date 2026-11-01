// 仅做所有权桥接：内存实际来自项目既有 CPU / CUDA / Pinned 池。
// DLPack views 保活 TensorOwner；最后一次归还前等待 GPU 消费者完成。
#include "infer/cpu/cpu_block.h"
#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/pinned_memory_pool.h"
#include "infer/cuda/cuda_memory_pool.h"
#include <ATen/dlpack.h>
#include <array>
#include <cstdio>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace {
thread_local std::string error;
enum Kind { CPU = 0, GPU = 1, PINNED = 2 };
struct DeviceScope {
    int previous = 0;
    explicit DeviceScope(int device) {
        infer::check_cuda(cudaGetDevice(&previous), "pool bridge get device");
        infer::check_cuda(cudaSetDevice(device), "pool bridge set device");
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};
infer::CachingCPUAllocatorConfig cpu_config(std::size_t limit) {
    infer::CachingCPUAllocatorConfig c;
    c.max_cached_bytes = limit;
    c.enable_thread_cache = false;
    c.max_reuse_waste = 0;
    return c;
}
infer::CachingCUDAAllocatorConfig gpu_config(std::size_t limit) {
    infer::CachingCUDAAllocatorConfig c;
    c.max_reserved_bytes = limit;
    c.max_cached_bytes = limit;
    c.event_batch_size = 16;
    return c;
}
struct Live {
    int kind;
    std::size_t bytes;
};
struct State {
    int device;
    std::array<std::size_t, 3> limits;
    infer::CachingCPUDeviceAllocator cpu;
    infer::CachingCUDADeviceAllocator gpu;
    infer::CachingPinnedHostAllocator pinned;
    cudaStream_t stream = nullptr;
    std::mutex mutex;
    std::unordered_map<void*, Live> live;
    std::size_t delete_errors = 0;
    State(int dev, std::size_t gpu_limit, std::size_t cpu_limit, std::size_t pinned_limit)
        : device(dev), limits{cpu_limit, gpu_limit, pinned_limit}, cpu(cpu_config(cpu_limit)),
          gpu(gpu_config(gpu_limit)), pinned(cpu_config(pinned_limit)) {
        infer::check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                          "pool bridge stream");
        try {
            gpu.register_stream(stream);
        } catch (...) {
            cudaStreamDestroy(stream);
            stream = nullptr;
            throw;
        }
    }
    ~State() {
        try {
            DeviceScope scope(device);
            gpu.unregister_stream(stream);
            gpu.trim();
            cpu.trim();
            pinned.trim();
            cudaStreamDestroy(stream);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Pool bridge cleanup: %s\n", e.what());
        }
    }
    void* allocate(int kind, std::size_t bytes) {
        if (kind < CPU || kind > PINNED || !bytes || bytes > limits[kind])
            throw std::invalid_argument("Pool kind, size or budget invalid");
        if (kind == GPU)
            return gpu.allocate(bytes, stream);
        const auto capacity = infer::cpu_detail::capacities[infer::cpu_detail::size_class(bytes)];
        const auto stats = kind == CPU ? cpu.stats() : pinned.stats();
        if (capacity > limits[kind] || stats.active_bytes > limits[kind] - capacity)
            throw std::bad_alloc();
        // 先释放闲置容量再向上游申请，避免 active+cached 超出实验的物理预算。
        if (stats.reserved_bytes > limits[kind] - capacity) {
            if (kind == CPU)
                cpu.trim();
            else
                pinned.trim();
        }
        return kind == CPU ? cpu.allocate(bytes) : pinned.allocate(bytes);
    }
    void release(int kind, void* ptr) {
        if (kind == GPU) {
            gpu.release(ptr, stream);
            gpu.flush();
            infer::check_cuda(cudaStreamSynchronize(stream), "pool bridge retire");
            gpu.collect();
        } else if (kind == PINNED)
            pinned.release(ptr);
        else
            cpu.release(ptr);
    }
};
using Handle = std::shared_ptr<State>;
struct TensorOwner {
    DLManagedTensor managed{};
    Handle state;
    int kind = CPU;
    std::int64_t shape = 0;
};
void destroy_tensor(DLManagedTensor* managed) noexcept {
    if (!managed)
        return;
    auto* owner = static_cast<TensorOwner*>(managed->manager_ctx);
    try {
        auto state = owner->state;
        DeviceScope scope(state->device);
        // 外部 tensor 可能经任意计算/复制流访问；它不受池内部流跟踪覆盖。
        if (owner->kind != CPU)
            infer::check_cuda(cudaDeviceSynchronize(), "pool bridge last consumer");
        std::lock_guard<std::mutex> lock(state->mutex);
        state->release(owner->kind, managed->dl_tensor.data);
        state->live.erase(managed->dl_tensor.data);
    } catch (const std::exception& e) {
        // 完成状态未知时保留 owner 和空间，不能把仍可能在用的内存还给池。
        std::lock_guard<std::mutex> lock(owner->state->mutex);
        ++owner->state->delete_errors;
        std::fprintf(stderr, "Pool tensor retained after failure: %s\n", e.what());
        return;
    }
    delete owner;
}
template <class F> auto checked(F f, decltype(f()) fallback) noexcept -> decltype(f()) {
    try {
        error.clear();
        return f();
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "Unknown pool bridge failure";
    }
    return fallback;
}
void cpu_json(std::ostringstream& out, const infer::CPUMemoryPoolStats& s) {
    out << "{\"active_bytes\":" << s.active_bytes << ",\"cached_bytes\":" << s.cached_bytes
        << ",\"reserved_bytes\":" << s.reserved_bytes << ",\"active_blocks\":" << s.active_blocks
        << ",\"upstream_allocations\":" << s.upstream_allocations
        << ",\"upstream_releases\":" << s.upstream_releases << ",\"reuse_count\":" << s.reuse_count
        << '}';
}
} // namespace
extern "C" {
const char* mi_pool_error() {
    return error.c_str();
}
void* mi_pool_create(int device, std::size_t gpu, std::size_t cpu, std::size_t pinned) {
    return checked(
        [&]() -> void* {
            if (!gpu || !cpu || !pinned)
                throw std::invalid_argument("Pool budgets must be positive");
            DeviceScope scope(device);
            return new Handle(std::make_shared<State>(device, gpu, cpu, pinned));
        },
        static_cast<void*>(nullptr));
}
void mi_pool_close(void* handle) {
    delete static_cast<Handle*>(handle);
}
DLManagedTensor* mi_pool_tensor(void* handle, int kind, std::size_t bytes) {
    return checked(
        [&]() -> DLManagedTensor* {
            if (!handle || bytes > std::size_t(INT64_MAX))
                throw std::invalid_argument("Invalid pool tensor request");
            auto owner = std::make_unique<TensorOwner>();
            owner->state = *static_cast<Handle*>(handle);
            owner->kind = kind;
            owner->shape = bytes;
            auto& state = *owner->state;
            DeviceScope scope(state.device);
            std::lock_guard<std::mutex> lock(state.mutex);
            void* ptr = state.allocate(kind, bytes);
            try {
                state.live.emplace(ptr, Live{kind, bytes});
            } catch (...) {
                state.release(kind, ptr);
                throw;
            }
            owner->managed.dl_tensor = {
                ptr,
                {kind == GPU ? kDLCUDA : kDLCPU, kind == GPU ? state.device : 0},
                1,
                {static_cast<std::uint8_t>(kDLUInt), 8, 1},
                &owner->shape,
                nullptr,
                0};
            owner->managed.manager_ctx = owner.get();
            owner->managed.deleter = destroy_tensor;
            return &owner.release()->managed;
        },
        static_cast<DLManagedTensor*>(nullptr));
}
void mi_pool_delete_tensor(DLManagedTensor* tensor) {
    destroy_tensor(tensor);
}
int mi_pool_owns(void* handle, int kind, std::uintptr_t address) {
    return checked(
        [&] {
            if (!handle)
                throw std::invalid_argument("Pool is closed");
            auto state = *static_cast<Handle*>(handle);
            std::lock_guard<std::mutex> lock(state->mutex);
            for (const auto& item : state->live) {
                const auto start = reinterpret_cast<std::uintptr_t>(item.first);
                if (item.second.kind == kind && address >= start &&
                    address - start < item.second.bytes)
                    return 1;
            }
            return 0;
        },
        -1);
}
const char* mi_pool_stats(void* handle) {
    thread_local std::string result;
    return checked(
        [&]() -> const char* {
            if (!handle)
                throw std::invalid_argument("Pool is closed");
            auto state = *static_cast<Handle*>(handle);
            DeviceScope scope(state->device);
            std::lock_guard<std::mutex> lock(state->mutex);
            std::ostringstream out;
            out << "{\"cpu\":";
            cpu_json(out, state->cpu.stats());
            out << ",\"pinned\":";
            cpu_json(out, state->pinned.stats());
            const auto s = state->gpu.stats();
            out << ",\"gpu\":{\"active_bytes\":" << s.active_capacity_bytes
                << ",\"requested_bytes\":" << s.active_requested_bytes
                << ",\"reserved_bytes\":" << s.reserved_bytes
                << ",\"peak_reserved_bytes\":" << s.peak_reserved_bytes
                << ",\"active_blocks\":" << s.active_blocks
                << ",\"upstream_allocations\":" << s.upstream_allocations
                << ",\"upstream_releases\":" << s.upstream_releases
                << ",\"reuse_count\":" << s.reuse_count
                << "},\"delete_errors\":" << state->delete_errors
                << ",\"live_allocations\":" << state->live.size() << '}';
            result = out.str();
            return result.c_str();
        },
        static_cast<const char*>(nullptr));
}
}
