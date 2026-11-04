#include "infer/io/weight_io.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#ifdef INFER_HAS_IO_URING
#include <liburing.h>
#endif
#ifdef INFER_HAS_CUFILE
#include <cufile.h>
#endif
namespace infer::io {
namespace {
void check_cuda(cudaError_t rc, const char* operation) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
}
#define INFER_IO_CUDA(call) check_cuda((call), #call)
struct DeviceScope {
    int old = 0;
    explicit DeviceScope(int device) { INFER_IO_CUDA(cudaGetDevice(&old)); INFER_IO_CUDA(cudaSetDevice(device)); }
    ~DeviceScope() { cudaSetDevice(old); }
};
struct DirectFd {
    int value = -1;
    explicit DirectFd(int original) {
        value = open(("/proc/self/fd/" + std::to_string(original)).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (value < 0) throw std::system_error(errno, std::generic_category(), "Open O_DIRECT weight descriptor");
    }
    ~DirectFd() { if (value >= 0) close(value); }
};
// Validate all externally supplied destinations before issuing any I/O.
std::vector<GpuRead> plan(const Safetensors& file, std::vector<GpuRead> input, int device) {
    // Sort once by destination: O(n log n), without an O(n^2) tensor-pair scan.
    std::sort(input.begin(),input.end(),[](const auto& a,const auto& b) {
        return reinterpret_cast<std::uintptr_t>(a.destination)<reinterpret_cast<std::uintptr_t>(b.destination);
    });
    CUdeviceptr base=0; std::size_t extent=0;
    std::uintptr_t previous_end=0;
    for (const auto& r : input) {
        const auto pointer=reinterpret_cast<CUdeviceptr>(r.destination);
        if (!r.bytes || !r.destination || r.file_offset>file.file_bytes() ||
            r.bytes>file.file_bytes()-r.file_offset || r.bytes>UINTPTR_MAX-pointer)
            throw std::invalid_argument("Invalid GPU weight range");
        if (pointer<previous_end) throw std::invalid_argument("Overlapping GPU destinations");
        // Subranges of the same live allocation inherit its device/type. The
        // packed arena needs one attributes/range query, not one per tensor.
        if (!base || pointer<base || pointer-base>=extent) {
            cudaPointerAttributes attributes{};
            INFER_IO_CUDA(cudaPointerGetAttributes(&attributes,r.destination));
            if (attributes.type!=cudaMemoryTypeDevice || attributes.device!=device)
                throw std::invalid_argument("Weights require device memory on the selected GPU");
            if (cuMemGetAddressRange(&base,&extent,pointer)!=CUDA_SUCCESS)
                throw std::out_of_range("GPU allocation does not expose a readable extent");
        }
        if (pointer<base || pointer-base>extent || r.bytes>extent-(pointer-base))
            throw std::out_of_range("GPU weight destination exceeds its allocation");
        previous_end=pointer+r.bytes;
    }
    std::sort(input.begin(), input.end(), [](const auto& a, const auto& b) { return a.file_offset < b.file_offset; });
    std::vector<GpuRead> output;
    for (const auto& r : input) {
        // COALESCED_DMA: adjacent file extents AND adjacent destinations become
        // one transfer. Gaps in the GPU arena never get overwritten.
        if (!output.empty() && output.back().file_offset + output.back().bytes == r.file_offset &&
            static_cast<char*>(output.back().destination) + output.back().bytes == r.destination) {
            CUdeviceptr base=0; std::size_t extent=0;
            const auto address=reinterpret_cast<CUdeviceptr>(output.back().destination);
            // Adjacent virtual addresses may belong to different cudaMalloc
            // allocations. A merged DMA must stay inside ONE allocation.
            if (cuMemGetAddressRange(&base,&extent,address)==CUDA_SUCCESS &&
                output.back().bytes+r.bytes<=extent-(address-base)) {
                output.back().bytes+=r.bytes;
                continue;
            }
            output.push_back(r);
        }
        else output.push_back(r);
    }
    return output;
}
#ifdef INFER_HAS_IO_URING
class Pipeline {
    enum class State { FREE, READING, READY, DMA };
    // CACHELINE_SLOTS: completion state for different halves never shares a line.
    struct alignas(64) Slot {
        char* data = nullptr;
        cudaEvent_t event = nullptr;
        State state = State::FREE;
        std::uint64_t offset = 0;
        std::size_t bytes = 0, required = 0, done = 0;
        int error = 0;
        bool unfenced = false;
    };
    DirectFd fd_;
    void* allocation_ = nullptr;
    std::array<Slot, 2> slots_;
    io_uring ring_{};
    bool ring_live_ = false;
    std::size_t pending_ = 0, chunk_;
    cudaStream_t stream_;
    LoadStats stats_;
    void submit(Slot& slot, std::size_t index) {
        auto* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) throw std::runtime_error("Direct reader SQ exhausted");
        // SINGLE_OWNER_RING: only this load thread touches SQ/CQ; no queue mutex,
        // no Python/window dispatch, no producer/consumer cache-line bouncing.
        io_uring_prep_read_fixed(sqe, stats_.registered_file ? 0 : fd_.value, slot.data + slot.done,
            slot.bytes - slot.done, slot.offset + slot.done, index);
        if (stats_.registered_file) sqe->flags|=IOSQE_FIXED_FILE;
        sqe->user_data = index;
        int rc;
        do { rc = io_uring_submit(&ring_); } while (rc == -EINTR);
        if (rc != 1) throw std::system_error(rc < 0 ? -rc : EIO, std::generic_category(), "Submit direct read");
        ++pending_; ++stats_.read_operations;
        slot.state = State::READING;
    }
    void complete() noexcept {
        io_uring_cqe* cqe = nullptr;
        int rc;
        do { rc = io_uring_wait_cqe(&ring_, &cqe); }
        while (rc == -EINTR || rc == -EAGAIN || rc == -EBUSY || rc == -ENOMEM);
        // Same fail-stop ownership policy as FileIO: unknown CQ ownership may
        // never return a live DMA destination to an allocator.
        if (rc < 0 || cqe->user_data >= slots_.size()) std::terminate();
        auto& slot = slots_[cqe->user_data];
        const int result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        --pending_;
        slot.state = State::READY;
        if (result < 0) slot.error = -result;
        else if (std::size_t(result) > slot.bytes - slot.done) slot.error = EIO;
        else {
            slot.done += result;
            stats_.disk_bytes += result;
            if (!result && slot.done < slot.required) slot.error = EIO; // zero progress is EOF/failure
        }
    }
    void read(std::size_t index, std::uint64_t offset, std::size_t bytes, std::uint64_t file_size) {
        auto& slot = slots_[index];
        if (slot.state == State::DMA) INFER_IO_CUDA(cudaEventSynchronize(slot.event));
        slot.offset = offset; slot.bytes = bytes;
        slot.required = std::min<std::uint64_t>(bytes, file_size - offset);
        slot.done = 0; slot.error = 0;
        submit(slot, index);
    }
  public:
    Pipeline(const Safetensors& file, std::size_t chunk, cudaStream_t stream)
        : fd_(file.fd()), chunk_(chunk), stream_(stream) {
        stats_.backend = "io_uring+O_DIRECT+pinned";
        const auto alignment=direct_alignment(fd_.value);
        stats_.alignment=alignment.bytes; stats_.alignment_queried=alignment.queried;
        if (chunk%stats_.alignment) throw std::invalid_argument("Chunk does not satisfy reported direct-I/O alignment");
        // cudaHostAlloc does not promise a 4 KiB base in its API: overallocate
        // and explicitly align the registered read buffers inside its owner.
        try {
            INFER_IO_CUDA(cudaHostAlloc(&allocation_, chunk * 2 + stats_.alignment, cudaHostAllocPortable));
            const auto base = align_up(reinterpret_cast<std::uintptr_t>(allocation_),stats_.alignment);
            for (std::size_t i = 0; i < slots_.size(); ++i) {
                slots_[i].data = reinterpret_cast<char*>(base + i * chunk);
                INFER_IO_CUDA(cudaEventCreateWithFlags(&slots_[i].event, cudaEventDisableTiming));
            }
            const int rc = io_uring_queue_init(4, &ring_, 0);
            if (rc < 0) throw std::system_error(-rc, std::generic_category(), "Create direct read ring");
            ring_live_ = true;
            stats_.registered_file=io_uring_register_files(&ring_,&fd_.value,1)==0;
            iovec buffers[2]{{slots_[0].data,chunk},{slots_[1].data,chunk}};
            const int registered = io_uring_register_buffers(&ring_, buffers, 2);
            if (registered < 0) throw std::system_error(-registered, std::generic_category(), "Register pinned read ring");
            stats_.registered_buffers = true;
            stats_.staging_bytes = chunk * 2 + stats_.alignment;
        } catch (...) { cleanup(); throw; }
    }
    Pipeline(const Pipeline&) = delete;
    ~Pipeline() { cleanup(); }
    void cleanup() noexcept {
        while (pending_) complete(); // every accepted read releases its destination
        if (ring_live_) { io_uring_queue_exit(&ring_); ring_live_ = false; }
        bool safe = true;
        for (auto& slot : slots_) if (slot.event) {
            if (slot.unfenced ? cudaStreamSynchronize(stream_) != cudaSuccess :
                (slot.state == State::DMA && cudaEventSynchronize(slot.event) != cudaSuccess)) safe = false;
            cudaEventDestroy(slot.event); slot.event = nullptr;
        }
        if (allocation_ && safe) cudaFreeHost(allocation_);
        if (allocation_ && !safe) {
            std::fputs("Fatal: CUDA completion unknown; caller GPU destinations cannot be safely reused\n",stderr);
            std::terminate();
        }
        allocation_ = nullptr;
    }
    LoadStats load(const Safetensors& file, const std::vector<GpuRead>& ranges) {
        struct Window { std::uint64_t offset; std::size_t bytes; };
        std::vector<Window> windows;
        for (const auto& r : ranges) {
            auto extent = aligned_range(r.file_offset, r.bytes,stats_.alignment);
            std::uint64_t begin = extent.offset, end = extent.offset + extent.bytes;
            if (!windows.empty() && begin < windows.back().offset + windows.back().bytes)
                begin = windows.back().offset + windows.back().bytes;
            if (begin >= end) continue;
            // Merge neighboring tiny tensors into the same physical read window.
            if (!windows.empty() && begin == windows.back().offset + windows.back().bytes &&
                windows.back().bytes < chunk_) {
                const auto extra = std::min<std::uint64_t>(end - begin, chunk_ - windows.back().bytes);
                windows.back().bytes += extra; begin += extra;
            }
            while (begin < end) {
                const auto n = std::min<std::uint64_t>(chunk_, end - begin);
                windows.push_back({begin, std::size_t(n)}); begin += n;
            }
        }
        for (std::size_t i = 0; i < std::min<std::size_t>(2, windows.size()); ++i)
            read(i, windows[i].offset, windows[i].bytes, file.file_bytes());
        std::size_t first_range = 0;
        for (std::size_t i = 0; i < windows.size(); ++i) {
            auto& slot = slots_[i % 2];
            for (;;) {
                while (slot.state == State::READING) complete();
                if (slot.error) throw std::system_error(slot.error, std::generic_category(), "Direct read completion");
                if (slot.done>=slot.required) break;
                if (slot.done%stats_.alignment)
                    throw std::runtime_error("Unaligned short O_DIRECT read before required tensor end");
                submit(slot,i%2); // positive aligned short completion: retry the remainder
            }
            const auto end = slot.offset + slot.bytes;
            while (first_range < ranges.size() && ranges[first_range].file_offset + ranges[first_range].bytes <= slot.offset) ++first_range;
            for (auto j = first_range; j < ranges.size() && ranges[j].file_offset < end; ++j) {
                const auto& r = ranges[j];
                const auto begin = std::max(slot.offset, r.file_offset);
                const auto last = std::min(end, r.file_offset + r.bytes);
                if (begin >= last) continue;
                const auto copy_rc=cudaMemcpyAsync(static_cast<char*>(r.destination)+(begin-r.file_offset),
                    slot.data+(begin-slot.offset),last-begin,cudaMemcpyHostToDevice,stream_);
                if (copy_rc!=cudaSuccess)
                    throw std::runtime_error("Weight H2D failed at file offset "+std::to_string(begin)+
                        " bytes "+std::to_string(last-begin)+": "+cudaGetErrorString(copy_rc));
                slot.unfenced = true;
                slot.state = State::DMA; // set immediately: even a later event-record failure must drain
                stats_.tensor_bytes += last - begin; ++stats_.dma_calls;
            }
            if (slot.state == State::DMA) {
                const auto rc = cudaEventRecord(slot.event, stream_);
                if (rc != cudaSuccess) { cudaStreamSynchronize(stream_); INFER_IO_CUDA(rc); }
                slot.unfenced = false;
            }
            // DOUBLE_BUFFER_FENCE: only this half waits for its last H2D before
            // the next disk read overwrites it. The other read/DMA keeps running.
            if (i + 2 < windows.size()) read(i % 2, windows[i+2].offset, windows[i+2].bytes, file.file_bytes());
        }
        INFER_IO_CUDA(cudaStreamSynchronize(stream_));
        return stats_;
    }
};
#endif
#ifdef INFER_HAS_CUFILE
struct GdsDriver {
    CUfileDrvProps_t props{};
    GdsDriver() {
        auto rc = cuFileDriverOpen();
        if (rc.err != CU_FILE_SUCCESS) throw std::runtime_error("cuFile driver unavailable: " + std::to_string(rc.err));
        rc = cuFileDriverGetProperties(&props);
        if (rc.err != CU_FILE_SUCCESS) { cuFileDriverClose(); throw std::runtime_error("cuFile properties unavailable"); }
    }
    ~GdsDriver() { cuFileDriverClose(); }
};
std::shared_ptr<GdsDriver> gds_driver() {
    // Only driver lifecycle is locked; read traffic never takes this mutex.
    static std::mutex mutex;
    static std::weak_ptr<GdsDriver> cached;
    std::lock_guard<std::mutex> lock(mutex);
    auto owner = cached.lock();
    if (!owner) { owner = std::make_shared<GdsDriver>(); cached = owner; }
    return owner;
}
bool cufile_read(const Safetensors& file, const std::vector<GpuRead>& ranges,
                 cudaStream_t stream, LoadStats& stats) {
    std::shared_ptr<GdsDriver> driver;
    try { driver = gds_driver(); }
    catch (const std::runtime_error& error) { stats.fallback_reason = error.what(); return false; }
    if (driver->props.nvfs.dcontrolflags & (1U << CU_FILE_ALLOW_COMPAT_MODE)) {
        stats.fallback_reason = "cuFile compatibility fallback enabled; prefer explicit io_uring path";
        return false;
    }
    for (const auto& r : ranges) if ((r.file_offset | r.bytes | reinterpret_cast<std::uintptr_t>(r.destination)) % page_bytes) {
        stats.fallback_reason = "cuFile strict path needs aligned file, GPU address and size";
        return false;
    }
    DirectFd fd(file.fd());
    CUfileDescr_t descriptor{}; descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; descriptor.handle.fd = fd.value;
    CUfileHandle_t handle{};
    auto rc = cuFileHandleRegister(&handle, &descriptor);
    if (rc.err != CU_FILE_SUCCESS) {
        stats.fallback_reason = "File does not support cuFile: " + std::to_string(rc.err); return false;
    }
    struct Handle { CUfileHandle_t value; ~Handle() { cuFileHandleDeregister(value); } } owner{handle};
    INFER_IO_CUDA(cudaStreamSynchronize(stream)); // synchronous cuFileRead has no CUDA stream ordering
    for (const auto& r : ranges) {
        CUdeviceptr base = 0; std::size_t extent = 0;
        if (cuMemGetAddressRange(&base, &extent, reinterpret_cast<CUdeviceptr>(r.destination)) != CUDA_SUCCESS)
            throw std::runtime_error("cuFile destination allocation disappeared");
        rc = cuFileBufRegister(reinterpret_cast<void*>(base), extent, 0);
        if (rc.err != CU_FILE_SUCCESS) { stats.fallback_reason = "cuFile buffer registration unavailable"; return false; }
        const auto n = cuFileRead(handle, reinterpret_cast<void*>(base), r.bytes, r.file_offset,
                                  reinterpret_cast<CUdeviceptr>(r.destination) - base);
        cuFileBufDeregister(reinterpret_cast<void*>(base));
        if (n < 0 && (n == -ENOTSUP || n == -EOPNOTSUPP || n == -CU_FILE_IO_NOT_SUPPORTED ||
                     n == -CU_FILE_DEVICE_NOT_SUPPORTED || n == -CU_FILE_PLATFORM_NOT_SUPPORTED)) {
            stats.fallback_reason = "cuFile operation unsupported"; return false;
        }
        if (n < 0 || std::size_t(n) != r.bytes) throw std::runtime_error("cuFile read failed/short: " + std::to_string(n));
        stats.disk_bytes += n; stats.tensor_bytes += n; ++stats.read_operations;
    }
    stats.backend = "cuFile-strict"; // backend selection alone is not a P2P attestation
    return true;
}
#endif
class AutoBackend final : public StorageBackend {
    WeightIOConfig config_;
  public:
    explicit AutoBackend(WeightIOConfig config) : config_(config) {
        if (config.chunk_bytes < page_bytes || config.chunk_bytes > 64 * 1024 * 1024 || config.chunk_bytes % page_bytes)
            throw std::invalid_argument("Weight chunk must be 4 KiB aligned, in [4 KiB,64 MiB]");
    }
    LoadStats read(const Safetensors& file, std::vector<GpuRead> requests, cudaStream_t stream) override {
        if (!stream) throw std::invalid_argument("Weight loading requires an explicit CUDA stream");
        DeviceScope scope(config_.device);
        auto ranges = plan(file, std::move(requests), config_.device);
        if (ranges.empty()) return {"empty", {}};
        LoadStats attempt;
        auto fallback = ranges;
#ifdef INFER_HAS_CUFILE
        if (config_.try_cufile) {
            std::vector<GpuRead> aligned;
            fallback.clear();
            for (const auto& r:ranges) {
                if (!((r.file_offset | r.bytes | reinterpret_cast<std::uintptr_t>(r.destination)) % page_bytes))
                    aligned.push_back(r);
                else fallback.push_back(r);
            }
            if (aligned.empty()) attempt.fallback_reason="Unaligned tensors require pinned staging";
            else if (!cufile_read(file,aligned,stream,attempt)) fallback=ranges;
            if (fallback.empty()) return attempt;
        }
#else
        if (config_.try_cufile) attempt.fallback_reason = "Built without cuFile";
#endif
#ifdef INFER_HAS_IO_URING
        Pipeline pipeline(file, config_.chunk_bytes, stream);
        auto result = pipeline.load(file, fallback);
        result.disk_bytes += attempt.disk_bytes;
        result.read_operations += attempt.read_operations;
        if (attempt.backend=="cuFile-strict") {
            result.tensor_bytes += attempt.tensor_bytes;
            result.backend="cuFile-strict+io_uring+O_DIRECT+pinned";
        }
        result.fallback_reason = std::move(attempt.fallback_reason);
        return result;
#else
        throw std::runtime_error("Direct weight fallback requires INFER_ENABLE_IO_URING and liburing");
#endif
    }
};
} // namespace
std::unique_ptr<StorageBackend> make_weight_backend(WeightIOConfig config) {
    return std::make_unique<AutoBackend>(config);
}
struct DeviceWeights::Impl {
    struct View { std::string name; std::size_t offset; };
    void* allocation = nullptr;
    std::size_t bytes = 0;
    int device = 0;
    LoadStats stats;
    std::vector<View> views;
    ~Impl() {
        if (allocation) {
            int old = 0; cudaGetDevice(&old); cudaSetDevice(device);
            cudaFree(allocation); cudaSetDevice(old);
        }
    }
};
DeviceWeights::DeviceWeights(const Safetensors& file, StorageBackend& backend, cudaStream_t stream, int device)
    : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    for (const auto& tensor : file.tensors()) {
        impl_->bytes = align_up(impl_->bytes, 256);
        impl_->views.push_back({std::string(tensor.name), impl_->bytes});
        if (tensor.bytes > SIZE_MAX - impl_->bytes) throw std::overflow_error("GPU arena size overflow");
        impl_->bytes += tensor.bytes;
    }
    if (!impl_->bytes) { impl_->stats.backend = "empty"; return; }
    DeviceScope scope(device);
    INFER_IO_CUDA(cudaMalloc(&impl_->allocation, impl_->bytes));
    std::vector<GpuRead> requests;
    for (std::size_t i = 0; i < file.tensors().size(); ++i) {
        const auto& tensor = file.tensors()[i];
        if (tensor.bytes) requests.push_back({tensor.file_offset,
            static_cast<char*>(impl_->allocation) + impl_->views[i].offset, tensor.bytes});
    }
    impl_->stats = backend.read(file, std::move(requests), stream);
}
DeviceWeights::~DeviceWeights() = default;
const void* DeviceWeights::data(std::string_view name) const {
    for (const auto& view : impl_->views) if (view.name == name)
        return impl_->allocation ? static_cast<char*>(impl_->allocation) + view.offset : nullptr;
    throw std::out_of_range("Unknown device tensor");
}
std::size_t DeviceWeights::allocated_bytes() const noexcept { return impl_->bytes; }
const LoadStats& DeviceWeights::stats() const noexcept { return impl_->stats; }
} // namespace infer::io
