#include "infer/storage/async_kv_pipeline.h"
#include "infer/core/allocator.h"
#include "infer/cuda/cuda_block.h"
#include "infer/cuda/cuda_memory_pool.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <semaphore.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace infer {
namespace {
template <class T>
class Ring {
  public:
    static constexpr unsigned capacity = 8;
    bool push(T value) noexcept {
        const auto tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_.load(std::memory_order_acquire) == capacity) return false;
        entries_[tail % capacity] = std::move(value);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& value) noexcept {
        const auto head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) return false;
        value = std::move(entries_[head % capacity]);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

  private:
    static_assert(std::atomic<unsigned>::is_always_lock_free);
    std::array<T, capacity> entries_{};
    std::atomic<unsigned> head_{0}, tail_{0};
};

struct Job {
    LayerRead read;
    const std::byte* key = nullptr;
    const std::byte* value = nullptr;
    std::size_t position = 0, count = 1, row_bytes = 0;
};

struct Done {
    ReadDone read;
    bool write = false;
};

class LayerIO {
  public:
    LayerIO(const AsyncKVPipeline::Read& read, const AsyncKVPipeline::Write& write)
        : read_(read), write_(write) {
        if (sem_init(&wake_, 0, 0) != 0)
            throw std::system_error(errno, std::generic_category(), "initialize KV I/O semaphore");
        try {
            worker_ = std::thread([this] { run(); });
        } catch (...) {
            sem_destroy(&wake_);
            throw;
        }
    }

    ~LayerIO() {
        stopping_.store(true, std::memory_order_release);
        sem_post(&wake_);
        worker_.join();
        sem_destroy(&wake_);
    }

    bool try_submit(Job job) noexcept {
        // One reserved completion per accepted job, including active jobs.
        if (outstanding_ == Ring<Done>::capacity || !requests_.push(job)) return false;
        ++outstanding_;
        sem_post(&wake_);
        return true;
    }

    bool try_pop(Done& done) noexcept {
        if (!completions_.pop(done)) return false;
        --outstanding_;
        return true;
    }

    bool idle() const noexcept { return outstanding_ == 0; }

  private:
    void run() noexcept {
        for (;;) {
            int status;
            do { status = sem_wait(&wake_); } while (status < 0 && errno == EINTR);
            if (status != 0) std::terminate();
            Job job{};
            if (!requests_.pop(job)) {
                if (stopping_.load(std::memory_order_acquire)) return;
                continue;
            }
            Done done{{job.read.generation, job.read.layer, {}}, job.key != nullptr};
            try {
                if (done.write) {
                    for (std::size_t row=0;row<job.count;++row)
                        write_(job.read.layer, job.position+row, job.key+row*job.row_bytes, job.value+row*job.row_bytes);
                }
                else read_(job.read);
            } catch (...) {
                done.read.error = std::current_exception();
            }
            if (!completions_.push(std::move(done))) std::terminate(); // Reserved at admission.
        }
    }

    // Impl owns callbacks until this worker has drained and joined.
    const AsyncKVPipeline::Read& read_;
    const AsyncKVPipeline::Write& write_;
    Ring<Job> requests_;
    Ring<Done> completions_;
    unsigned outstanding_ = 0;
    std::atomic<bool> stopping_{false};
    sem_t wake_{};
    std::thread worker_;
};

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > SIZE_MAX / b) throw std::length_error("KV pipeline capacity overflow");
    return a * b;
}

bool completed(cudaEvent_t event) {
    const auto status = cudaEventQuery(event);
    if (status == cudaErrorNotReady) return false;
    check_cuda(status, "query KV pipeline event");
    return true;
}

void create_event(cudaEvent_t& event) {
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create KV event");
}
} // namespace

struct AsyncKVPipeline::Impl {
    enum class Phase { Idle, Running, Failed, Retiring, Fenced };
    enum class Input { Free, Reading, Ready, Uploading };
    enum class Output { Free, Downloading, Writing };
    struct Slot {
        std::byte* device = nullptr;
        std::byte* input = nullptr;
        std::byte* output = nullptr;
        const float** table = nullptr;
        cudaEvent_t ready{}, compute_done{}, reusable{};
        Input in_state = Input::Free;
        Output out_state = Output::Free;
        std::size_t in_layer = 0, out_layer = 0;
    };

    Config config;
    std::unique_ptr<CachingCUDADeviceAllocator> pool;
    PinnedHostAllocator pinned;
    std::unique_ptr<LayerIO> io;
    cudaStream_t compute_stream_{}, prefetch_h2d_stream_{}, offload_d2h_stream_{};
    std::vector<Slot> slots;
    Read read;
    Write write;
    bool resident_initialized = false;
    std::size_t dirty_begin = 0, count = 1;
    std::array<cudaEvent_t, 3> retirement{};
    cudaEvent_t output_produced{}, output_ready{};
    bool registered = false, downloading = false;
    Phase phase = Phase::Idle;
    std::size_t device_bytes = 0, host_bytes = 0;
    std::size_t position = 0, pages = 0, next_read = 0, next_upload = 0, next_compute = 0;
    std::size_t committed = 0;
    std::uint64_t generation = 0;

    explicit Impl(Config value) : config(value), slots(value.gpu_resident ? value.layers : 2) {}

    std::array<cudaStream_t, 3> streams() const {
        return {compute_stream_, prefetch_h2d_stream_, offload_d2h_stream_};
    }

    void initialize(Read reader, Write writer) {
        read = std::move(reader); write = std::move(writer);
        const auto plane_bytes = multiply(config.tokens_per_page, config.row_bytes);
        if (!config.layers || !config.max_pages || !plane_bytes ||
            config.page_bytes % alignof(void*) || config.row_bytes % sizeof(float) ||
            plane_bytes > config.page_bytes / 2 || !read || !write)
            throw std::invalid_argument("invalid KV pipeline configuration");
        const auto page_capacity = multiply(config.max_pages, config.page_bytes);
        const auto table_bytes = multiply(config.max_pages, sizeof(void*));
        const auto output_bytes = multiply(2, multiply(config.max_pages, plane_bytes));
        if (table_bytes > SIZE_MAX - page_capacity)
            throw std::length_error("KV pipeline slot overflow");
        const auto slot_bytes = page_capacity + table_bytes;
        if (output_bytes > SIZE_MAX - slot_bytes)
            throw std::length_error("KV pipeline staging overflow");
        host_bytes = multiply(slots.size(), slot_bytes + output_bytes);
        CachingCUDAAllocatorConfig pool_config;
        const auto slot_capacity = cuda_pool_detail::allocation_bucket(slot_bytes, pool_config).second;
        const auto capacity = multiply(slots.size(), slot_capacity);
        if (config.max_device_bytes && capacity > config.max_device_bytes)
            throw std::length_error("async KV GPU budget must hold two max-context slots and page tables");
        // Reserve enough cache for both rounded allocations to retire by event.
        pool_config.max_reserved_bytes = capacity;
        pool_config.max_cached_bytes = capacity;
        pool = std::make_unique<CachingCUDADeviceAllocator>(pool_config);
        check_cuda(cudaSetDevice(config.device), "select KV pipeline device");
        for (auto* stream : {&compute_stream_, &prefetch_h2d_stream_, &offload_d2h_stream_})
            check_cuda(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking), "create KV stream");
        for (auto& event : retirement) create_event(event);
        create_event(output_produced);
        create_event(output_ready);
        pool->register_stream(compute_stream_);
        registered = true;
        for (auto& slot : slots) {
            create_event(slot.ready);
            create_event(slot.compute_done);
            create_event(slot.reusable);
            slot.device = static_cast<std::byte*>(pool->allocate(slot_bytes, compute_stream_));
            slot.input = static_cast<std::byte*>(pinned.allocate(slot_bytes));
            slot.output = static_cast<std::byte*>(pinned.allocate(output_bytes));
            slot.table = reinterpret_cast<const float**>(slot.device + page_capacity);
            auto** host = reinterpret_cast<const float**>(slot.input + page_capacity);
            for (std::size_t page = 0; page < config.max_pages; ++page)
                host[page] = reinterpret_cast<const float*>(slot.device + page * config.page_bytes);
            check_cuda(cudaMemcpyAsync(slot.table, host, table_bytes, cudaMemcpyHostToDevice,
                                       prefetch_h2d_stream_), "initialize KV page table");
            check_cuda(cudaEventRecord(slot.ready, prefetch_h2d_stream_), "initialize table fence");
        }
        device_bytes = pool->stats().reserved_bytes;
        if (!config.gpu_resident) io = std::make_unique<LayerIO>(read, write);
    }

    void poll_completions() {
        Done done;
        while (io->try_pop(done)) {
            if (done.read.error) std::rethrow_exception(done.read.error);
            auto& slot = slots[done.read.layer % slots.size()];
            if (done.read.generation != generation ||
                (done.write ? slot.out_state != Output::Writing || slot.out_layer != done.read.layer
                            : slot.in_state != Input::Reading || slot.in_layer != done.read.layer))
                throw std::logic_error("unexpected KV I/O completion");
            if (done.write) {
                slot.out_state = Output::Free;
                ++committed;
            } else {
                slot.in_state = Input::Ready;
            }
        }
        for (auto& slot : slots) {
            if (slot.in_state == Input::Uploading && completed(slot.ready))
                slot.in_state = Input::Free;
            if (slot.out_state != Output::Downloading || !completed(slot.reusable)) continue;
            if (io->try_submit({{generation, slot.out_layer, pages, nullptr},
                                slot.output, slot.output + count * config.row_bytes, position, count, config.row_bytes}))
                slot.out_state = Output::Writing;
        }
    }

    void submit_reads() {
        while (next_read < config.layers) {
            auto& slot = slots[next_read % slots.size()];
            if (slot.in_state != Input::Free ||
                !io->try_submit({{generation, next_read, pages, slot.input}})) return;
            slot.in_layer = next_read++;
            slot.in_state = Input::Reading;
        }
    }

    bool try_upload() {
        const auto layer = next_upload;
        if (layer == config.layers) return false;
        auto& slot = slots[layer % slots.size()];
        if (slot.in_state != Input::Ready || slot.in_layer != layer) return false;
        if (layer >= slots.size()) {
            // The record must already be enqueued. CUDA waits snapshot that record.
            if (next_compute <= layer - slots.size()) return false;
            check_cuda(cudaStreamWaitEvent(prefetch_h2d_stream_, slot.reusable, 0), "reuse KV slot");
        }
        check_cuda(cudaMemcpyAsync(slot.device, slot.input, pages * config.page_bytes,
                                   cudaMemcpyHostToDevice, prefetch_h2d_stream_), "upload KV pages");
        check_cuda(cudaEventRecord(slot.ready, prefetch_h2d_stream_), "record KV ready");
        slot.in_state = Input::Uploading;
        ++next_upload;
        return true;
    }

    LayerView layer_view(const Slot& slot, std::size_t token_position) const {
        auto* key = slot.device + (token_position / config.tokens_per_page) * config.page_bytes +
                    (token_position % config.tokens_per_page) * config.row_bytes;
        return {reinterpret_cast<float*>(key),
                reinterpret_cast<float*>(key + config.tokens_per_page * config.row_bytes),
                slot.table, config.tokens_per_page, config.row_bytes / sizeof(float)};
    }

    bool try_compute(const Compute& compute) {
        const auto layer = next_compute;
        if (layer == next_upload) return false;
        auto& slot = slots[layer % slots.size()];
        if (slot.out_state != Output::Free) return false;

        check_cuda(cudaStreamWaitEvent(compute_stream_, slot.ready, 0), "wait for KV pages");
        compute(layer, layer_view(slot, position), compute_stream_);
        check_cuda(cudaEventRecord(slot.compute_done, compute_stream_), "record KV consumer");
        check_cuda(cudaStreamWaitEvent(offload_d2h_stream_, slot.compute_done, 0), "wait for KV producer");
        download_rows(slot, position, count);
        // One D2H fence protects both device reuse and host writeback submission.
        check_cuda(cudaEventRecord(slot.reusable, offload_d2h_stream_), "record reusable KV slot");
        slot.out_layer = layer;
        slot.out_state = Output::Downloading;
        ++next_compute;
        return true;
    }

    void download_rows(Slot& slot, std::size_t first, std::size_t rows) {
        const auto plane = config.tokens_per_page * config.row_bytes;
        for (std::size_t row=0;row<rows;) {
            const auto pos = first + row;
            const auto n = std::min(rows-row, config.tokens_per_page-pos%config.tokens_per_page);
            auto* source = slot.device + (pos/config.tokens_per_page)*config.page_bytes + (pos%config.tokens_per_page)*config.row_bytes;
            check_cuda(cudaMemcpy2DAsync(slot.output+row*config.row_bytes, rows*config.row_bytes,
                source,plane,n*config.row_bytes,2,cudaMemcpyDeviceToHost,offload_d2h_stream_), "download batched KV rows");
            row += n;
        }
    }

    void resident_compute(const Compute& compute) {
        for (std::size_t layer=0;layer<config.layers;++layer) {
            auto& slot=slots[layer];
            if (!resident_initialized) {
                check_cuda(cudaStreamWaitEvent(compute_stream_,slot.ready,0), "wait for resident table");
                if (position) {
                    read({generation,layer,pages,slot.input});
                    check_cuda(cudaMemcpyAsync(slot.device,slot.input,pages*config.page_bytes,
                        cudaMemcpyHostToDevice,compute_stream_), "restore resident prefix");
                }
            }
            compute(layer, layer_view(slot, position), compute_stream_);
        }
        if (!resident_initialized) dirty_begin=position;
        resident_initialized=true;
        phase = Phase::Idle;
    }

    void flush_resident() {
        if (!config.gpu_resident || !resident_initialized) return;
        if (phase != Phase::Idle) throw std::logic_error("resident writeback requires completed admission");
        const auto rows=position+count-dirty_begin;
        if (!rows) return;
        check_cuda(cudaEventRecord(output_produced,compute_stream_), "resident flush producer");
        check_cuda(cudaStreamWaitEvent(offload_d2h_stream_,output_produced,0), "resident flush dependency");
        for (auto& slot:slots) download_rows(slot,dirty_begin,rows);
        check_cuda(cudaEventRecord(output_ready,offload_d2h_stream_), "resident flush completion");
        while (!completed(output_ready)) std::this_thread::yield();
        for (std::size_t layer=0;layer<config.layers;++layer)
            for (std::size_t row=0;row<rows;++row)
                write(layer,dirty_begin+row,slots[layer].output+row*config.row_bytes,
                    slots[layer].output+(rows+row)*config.row_bytes);
        dirty_begin=position+count;
    }

    void begin_retire() {
        if (phase == Phase::Fenced) return;
        phase = Phase::Retiring;
        const auto handles = streams();
        for (std::size_t i = 0; i < handles.size(); ++i)
            if (handles[i] && retirement[i])
                check_cuda(cudaEventRecord(retirement[i], handles[i]), "fence KV pipeline");
        phase = Phase::Fenced;
    }

    bool poll_retired() {
        if (phase != Phase::Fenced) throw std::logic_error("KV pipeline retirement must be fenced");
        if (io) {
            Done done;
            while (io->try_pop(done)) {}
            if (!io->idle()) return false;
        }
        for (auto event : retirement)
            if (event && !completed(event)) return false;
        phase = Phase::Idle;
        downloading = false;
        resident_initialized = false;
        for (auto& slot : slots) {
            slot.in_state = Input::Free;
            slot.out_state = Output::Free;
        }
        return true;
    }

    void release() {
        io.reset();
        for (auto& slot : slots) {
            if (slot.device) {
                pool->release(slot.device, compute_stream_);
                slot.device = nullptr;
            }
            if (slot.input) pinned.release(slot.input);
            if (slot.output) pinned.release(slot.output);
            for (auto event : {slot.ready, slot.compute_done, slot.reusable})
                if (event) cudaEventDestroy(event);
        }
        if (registered) {
            pool->flush();
            pool->collect();
            pool->unregister_stream(compute_stream_);
        }
        for (auto event : retirement) if (event) cudaEventDestroy(event);
        if (output_produced) cudaEventDestroy(output_produced);
        if (output_ready) cudaEventDestroy(output_ready);
        for (auto stream : streams()) if (stream) cudaStreamDestroy(stream);
    }

    // The owning API is synchronous at destruction. Unknown DMA completion
    // keeps the affected memory and allocator out of circulation.
    static void dispose(std::unique_ptr<Impl> state) noexcept {
        try {
            if (state->compute_stream_) check_cuda(cudaSetDevice(state->config.device), "select KV device");
            state->begin_retire();
            while (!state->poll_retired()) std::this_thread::sleep_for(std::chrono::microseconds(20));
            state->release();
        } catch (const std::exception& error) {
            state->io.reset(); // Join before callbacks' storage can be destroyed.
            std::fprintf(stderr, "KV pipeline resources quarantined: %s\n", error.what());
            (void)state.release();
        }
    }
};

AsyncKVPipeline::AsyncKVPipeline(Config config, Read read, Write write)
    : impl_(std::make_unique<Impl>(config)) {
    try {
        impl_->initialize(std::move(read), std::move(write));
    } catch (...) {
        Impl::dispose(std::move(impl_));
        throw;
    }
}

AsyncKVPipeline::~AsyncKVPipeline() { Impl::dispose(std::move(impl_)); }

void AsyncKVPipeline::begin_token(std::size_t position, std::size_t count) {
    auto& s = *impl_;
    if (s.phase != Impl::Phase::Idle || s.downloading)
        throw std::logic_error("KV pipeline needs to finish or retire first");
    if (!count || position >= s.config.max_pages*s.config.tokens_per_page ||
        count > s.config.max_pages*s.config.tokens_per_page-position)
        throw std::out_of_range("KV pipeline context is full");
    s.position = position;
    s.count = count;
    s.pages = (position+count-1) / s.config.tokens_per_page + 1;
    s.next_read = s.next_upload = s.next_compute = s.committed = 0;
    ++s.generation;
    s.phase = Impl::Phase::Running;
    if (!s.config.gpu_resident) s.submit_reads();
}

bool AsyncKVPipeline::tick(const Compute& compute) {
    auto& s = *impl_;
    if (s.phase != Impl::Phase::Running) throw std::logic_error("no runnable KV token");
    try {
        if (s.config.gpu_resident) { s.resident_compute(compute); return true; }
        do {
            // Reclaim input staging between layer submissions so the next
            // NVMe read can start as soon as its previous H2D has finished.
            s.poll_completions();
            s.submit_reads();
            while (s.try_upload()) {}
        } while (s.try_compute(compute));
    } catch (...) {
        s.phase = Impl::Phase::Failed;
        throw;
    }
    if (s.committed != s.config.layers) return false;
    s.phase = Impl::Phase::Idle;
    return true;
}

bool AsyncKVPipeline::gpu_resident() const noexcept { return impl_->config.gpu_resident; }
std::size_t AsyncKVPipeline::layer_count() const noexcept { return impl_->config.layers; }
void AsyncKVPipeline::flush_resident() { impl_->flush_resident(); }

cudaStream_t AsyncKVPipeline::compute_stream() const noexcept { return impl_->compute_stream_; }

AsyncKVPipeline::LayerView AsyncKVPipeline::graph_slot(std::size_t index) const {
    const auto& s = *impl_;
    return s.layer_view(s.slots.at(index), 0);
}

void AsyncKVPipeline::download(void* pinned, const void* device, std::size_t bytes) {
    auto& s = *impl_;
    if (s.phase != Impl::Phase::Idle || s.downloading)
        throw std::logic_error("KV pipeline needs to finish or retire first");
    try {
        check_cuda(cudaEventRecord(s.output_produced, s.compute_stream_), "record output producer");
        check_cuda(cudaStreamWaitEvent(s.offload_d2h_stream_, s.output_produced, 0), "wait for output");
        check_cuda(cudaMemcpyAsync(pinned, device, bytes, cudaMemcpyDeviceToHost, s.offload_d2h_stream_),
                   "download output");
        check_cuda(cudaEventRecord(s.output_ready, s.offload_d2h_stream_), "record output ready");
        s.downloading = true;
    } catch (...) {
        s.phase = Impl::Phase::Failed;
        throw;
    }
}

bool AsyncKVPipeline::poll_download() {
    auto& s = *impl_;
    if (s.downloading && !completed(s.output_ready)) return false;
    s.downloading = false;
    return true;
}

void AsyncKVPipeline::begin_retire() { impl_->begin_retire(); }
bool AsyncKVPipeline::poll_retired() { return impl_->poll_retired(); }
std::size_t AsyncKVPipeline::reserved_device_bytes() const noexcept { return impl_->device_bytes; }
std::size_t AsyncKVPipeline::staging_bytes() const noexcept { return impl_->host_bytes; }
} // namespace infer
