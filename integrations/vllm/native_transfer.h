#pragma once

#include "host_io_queue.h"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if CUDA_VERSION >= 12080
#define MINIINFER_NATIVE_TRANSFERS 1

// One caller per direction owns these descriptors, events and Pinned halves.
// The existing Python coordinator publishes its independent completion event.
class NativeTransfer {
  public:
    using Work = HostIOQueue::Work;
    struct Result {
        std::uint64_t blocks = 0, windows = 0, prefetched_blocks = 0, prefetched_windows = 0;
    };

  private:
    static void check(CUresult result, const char* operation) {
        if (result == CUDA_SUCCESS)
            return;
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(std::string(operation) + ": " +
                                 (message ? message : "CUDA driver error"));
    }
    struct Context {
        CUcontext previous = nullptr;
        bool changed = false;
        explicit Context(CUcontext context) {
            check(cuCtxGetCurrent(&previous), "Read CUDA context");
            changed = previous != context;
            if (changed)
                check(cuCtxSetCurrent(context), "Set transfer CUDA context");
        }
        ~Context() {
            if (changed)
                cuCtxSetCurrent(previous);
        }
    };
    struct Event {
        CUevent value = nullptr;
        ~Event() {
            if (value)
                cuEventDestroy(value);
        }
    };
    struct Direction {
        std::mutex mutex;
        std::array<Event, 2> events;
        std::array<bool, 2> used{{false, false}};
        bool completion_unknown = false;
        std::vector<CUdeviceptr> src, dst;
        std::vector<std::size_t> lengths, indices;
        std::vector<void*> buffers;
    };
    Work work_;
    HostIOQueue& queue_;
    std::size_t bytes_, half_, gpu_capacity_, disk_capacity_, descriptor_cap_;
    std::vector<CUdeviceptr> bases_;
    std::vector<std::size_t> layer_bytes_;
    std::array<CUdeviceptr, 2> hosts_;
    std::array<CUstream, 2> streams_;
    bool read_ahead_;
    CUcontext context_ = nullptr;
    decltype(&cuMemcpyBatchAsync) batch_copy_ = nullptr;
    std::array<Direction, 2> directions_;

    void targets(Direction& state, const std::size_t* disk, std::size_t count,
                 std::size_t group, bool write) {
        state.indices.assign(disk, disk + count);
        state.buffers.resize(count);
        for (std::size_t i = 0; i < count; ++i)
            state.buffers[i] = reinterpret_cast<void*>(hosts_[write] +
                                                       (group * half_ + i) * bytes_);
    }
    void copy(Direction& state, const std::size_t* gpu, std::size_t count,
              std::size_t group, bool write) {
        const auto host = hosts_[write] + group * half_ * bytes_;
        if (bases_.size() == 1 &&
            std::equal(gpu + 1, gpu + count, gpu,
                       [](auto next, auto previous) { return next == previous + 1; })) {
            const auto device = bases_[0] + gpu[0] * bytes_;
            check(cuMemcpyAsync(write ? host : device, write ? device : host,
                                count * bytes_, streams_[write]), "Contiguous transfer DMA");
            return;
        }
        std::size_t index = 0;
        for (std::size_t block = 0; block < count; ++block) {
            std::size_t offset = 0;
            for (std::size_t layer = 0; layer < bases_.size(); ++layer, ++index) {
                const auto device = bases_[layer] + gpu[block] * layer_bytes_[layer];
                const auto pointer = host + block * bytes_ + offset;
                state.src[index] = write ? device : pointer;
                state.dst[index] = write ? pointer : device;
                state.lengths[index] = layer_bytes_[layer];
                offset += layer_bytes_[layer];
            }
        }
        CUmemcpyAttributes attributes{};
        attributes.srcAccessOrder = write ? CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
                                          : CU_MEMCPY_SRC_ACCESS_ORDER_ANY;
        std::size_t attribute_index = 0;
#if CUDA_VERSION < 13000
        std::size_t failed = 0;
#endif
        const auto step = descriptor_cap_ ? descriptor_cap_ : index;
        for (std::size_t begin = 0; begin < index; begin += step)
            check(batch_copy_(state.dst.data() + begin, state.src.data() + begin,
                              state.lengths.data() + begin, std::min(step, index - begin),
                              &attributes, &attribute_index, 1,
#if CUDA_VERSION < 13000
                              &failed,
#endif
                              streams_[write]),
                  "Batch transfer DMA");
    }

  public:
    NativeTransfer(Work work, HostIOQueue& queue, std::size_t bytes, std::size_t slots,
                   std::size_t gpu_capacity, std::size_t disk_capacity,
                   const std::vector<CUdeviceptr>& bases,
                   const std::vector<std::size_t>& layer_bytes, void* store_host,
                   void* load_host, CUstream store_stream, CUstream load_stream,
                   std::size_t descriptor_cap, bool read_ahead)
        : work_(std::move(work)), queue_(queue), bytes_(bytes), half_(std::min(slots / 2, std::size_t(64))),
          gpu_capacity_(gpu_capacity), disk_capacity_(disk_capacity),
          descriptor_cap_(descriptor_cap), bases_(bases), layer_bytes_(layer_bytes),
          hosts_{{reinterpret_cast<CUdeviceptr>(load_host), reinterpret_cast<CUdeviceptr>(store_host)}},
          streams_{{load_stream, store_stream}}, read_ahead_(read_ahead) {
        if (!bytes || slots < 2 || slots % 2 || !gpu_capacity || !disk_capacity ||
            bases.empty() || bases.size() != layer_bytes.size() || !store_host || !load_host ||
            !store_stream || !load_stream || slots > SIZE_MAX / bytes ||
            half_ > SIZE_MAX / bases.size())
            throw std::invalid_argument("Invalid native transfer configuration");
        std::size_t total = 0;
        for (std::size_t i = 0; i < bases.size(); ++i) {
            if (!layer_bytes[i] || layer_bytes[i] > bytes - total || !bases[i] ||
                gpu_capacity > (UINT64_MAX - bases[i]) / layer_bytes[i])
                throw std::invalid_argument("Invalid native transfer layer");
            total += layer_bytes[i];
        }
        if (total != bytes || slots * bytes > UINT64_MAX - hosts_[0] ||
            slots * bytes > UINT64_MAX - hosts_[1] ||
            (hosts_[0] < hosts_[1] + slots * bytes && hosts_[1] < hosts_[0] + slots * bytes))
            throw std::invalid_argument("Invalid native transfer extent");
        check(cuCtxGetCurrent(&context_), "Read transfer CUDA context");
        if (!context_)
            throw std::runtime_error("Native transfer requires the owning CUDA context");
        void* function = nullptr;
        CUdriverProcAddressQueryResult status{};
        check(cuGetProcAddress("cuMemcpyBatchAsync", &function,
                              CUDA_VERSION >= 13000 ? 13000 : 12080,
                              CU_GET_PROC_ADDRESS_DEFAULT, &status), "Resolve batch DMA");
        if (!function)
            throw std::runtime_error("Native batch DMA is unavailable");
        batch_copy_ = reinterpret_cast<decltype(batch_copy_)>(function);
        for (auto& state : directions_) {
            state.src.resize(half_ * bases.size());
            state.dst.resize(state.src.size());
            state.lengths.resize(state.src.size());
            state.indices.reserve(half_);
            state.buffers.reserve(half_);
            for (auto& event : state.events)
                check(cuEventCreate(&event.value, CU_EVENT_DISABLE_TIMING), "Create transfer fence");
        }
    }
    ~NativeTransfer() {
        CUcontext previous = nullptr;
        if (cuCtxGetCurrent(&previous) != CUDA_SUCCESS ||
            cuCtxSetCurrent(context_) != CUDA_SUCCESS)
            return;
        for (auto stream : streams_)
            cuStreamSynchronize(stream);
        for (auto& state : directions_)
            for (auto& event : state.events) {
                cuEventDestroy(event.value);
                event.value = nullptr;
            }
        cuCtxSetCurrent(previous);
    }
    void execute(bool write, const std::size_t* gpu, const std::size_t* disk,
                 std::size_t count, Result& result) {
        if (count && (!gpu || !disk))
            throw std::invalid_argument("Missing native transfer plan");
        for (std::size_t i = 0; i < count; ++i) {
            if (gpu[i] >= gpu_capacity_ || disk[i] >= disk_capacity_)
                throw std::invalid_argument("Native transfer block is out of bounds");
            const auto first = i / half_ * half_;
            for (auto j = first; j < i; ++j)
                if (disk[i] == disk[j])
                    throw std::invalid_argument("Duplicate native transfer disk slot");
        }
        auto& state = directions_[write];
        std::lock_guard<std::mutex> lock(state.mutex);
        Context context(context_);
        if (state.completion_unknown) {
            check(cuStreamSynchronize(streams_[write]), "Recover transfer completion");
            state.completion_unknown = false;
        }
        std::uint64_t pending = 0;
        const auto io = [&](std::size_t begin, std::size_t group) {
            const auto n = std::min(half_, count - begin);
            targets(state, disk + begin, n, group, write);
            work_(write, state.indices, state.buffers);
            result.blocks += n;
            ++result.windows;
        };
        try {
            if (write) {
                std::array<std::size_t, 2> offsets{};
                std::array<bool, 2> waiting{{false, false}};
                for (std::size_t begin = 0; begin < count; begin += half_) {
                    const auto group = (begin / half_) % 2, other = 1 - group;
                    copy(state, gpu + begin, std::min(half_, count - begin), group, true);
                    check(cuEventRecord(state.events[group].value, streams_[true]), "Record D2H fence");
                    offsets[group] = begin;
                    waiting[group] = true;
                    if (waiting[other]) {
                        check(cuEventSynchronize(state.events[other].value), "Wait D2H fence");
                        io(offsets[other], other);
                        waiting[other] = false;
                    }
                }
                for (std::size_t group = 0; group < waiting.size(); ++group)
                    if (waiting[group]) {
                        check(cuEventSynchronize(state.events[group].value), "Wait final D2H fence");
                        io(offsets[group], group);
                    }
                return;
            }
            for (std::size_t begin = 0; begin < count; begin += half_) {
                const auto group = (begin / half_) % 2, next = begin + half_;
                if (pending) {
                    const auto token = pending;
                    pending = 0; // wait consumes a failed token too.
                    queue_.wait(token);
                    const auto n = std::min(half_, count - begin);
                    result.blocks += n;
                    result.prefetched_blocks += n;
                    ++result.windows;
                    ++result.prefetched_windows;
                } else {
                    if (state.used[group])
                        check(cuEventSynchronize(state.events[group].value), "Wait reusable H2D half");
                    io(begin, group);
                }
                if (read_ahead_ && next < count) {
                    const auto other = 1 - group;
                    if (state.used[other])
                        check(cuEventSynchronize(state.events[other].value), "Wait prefetch H2D half");
                    targets(state, disk + next, std::min(half_, count - next), other, false);
                    pending = queue_.submit(false, state.indices, state.buffers);
                }
                copy(state, gpu + begin, std::min(half_, count - begin), group, false);
                check(cuEventRecord(state.events[group].value, streams_[false]), "Record H2D fence");
                state.used[group] = true;
            }
        } catch (...) {
            // Return neither half while accepted disk work or CUDA still owns it.
            if (pending) {
                try { queue_.wait(pending); } catch (...) {}
            }
            state.completion_unknown = cuStreamSynchronize(streams_[write]) != CUDA_SUCCESS;
            throw;
        }
    }
};
#else
#define MINIINFER_NATIVE_TRANSFERS 0
#endif
