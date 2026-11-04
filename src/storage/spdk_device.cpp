#include "infer/storage/ssd_device.h"
#include "infer/config.h"

#include <stdexcept>

#ifdef INFER_HAS_SPDK
#if INFER_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <spdk/bdev.h>
#include <spdk/env.h>
#include <spdk/env_dpdk.h>
#include <spdk/event.h>
#include <spdk/thread.h>
#include <system_error>
#include <sys/uio.h>
#include <thread>

namespace infer {
namespace {
std::atomic<bool> app_owned{false};
std::atomic<bool> app_retired{false};
std::atomic<std::size_t> arena_bytes{0};
constexpr std::size_t sector_alignment = 4096;
#if INFER_ENABLE_CUDA
bool register_cuda(void* address, std::size_t bytes) noexcept {
    if (cudaHostRegister(address, bytes, cudaHostRegisterPortable) == cudaSuccess)
        return true;
    (void)cudaGetLastError();
    return false;
}
#endif
bool snapshot_arena() noexcept {
    if (spdk_env_dpdk_external_init())
        return false;
    std::size_t total = 0;
    std::int32_t node;
    SPDK_ENV_FOREACH_NUMA_ID(node) {
        spdk_env_dpdk_mem_stats stats{};
        if (spdk_env_dpdk_get_mem_stats(&stats, static_cast<std::uint32_t>(node)) ||
            stats.heap_totalsz_bytes > SIZE_MAX - total)
            return false;
        total += stats.heap_totalsz_bytes;
    }
    arena_bytes.store(total, std::memory_order_relaxed);
    return true;
}

class SpdkDevice final : public SSDDevice, public std::enable_shared_from_this<SpdkDevice> {
    struct Buffer final : public SSDDmaBuffer {
        std::shared_ptr<SpdkDevice> owner;
        void* address;
        std::size_t bytes;
        bool cuda_pinned = false;
        Buffer(std::shared_ptr<SpdkDevice> value, void* data, std::size_t size)
            : owner(std::move(value)), address(data), bytes(size) {}
        ~Buffer() override {
            if (address)
                owner->release_buffer(address, bytes, cuda_pinned);
        }
        void* data() const noexcept override { return address; }
        std::size_t size() const noexcept override { return bytes; }
    };
    struct Batch {
        SpdkDevice& owner;
        const std::vector<FileIORequest>& ranges;
        std::vector<std::shared_ptr<Buffer>> leases;
        bool write;
        std::size_t index = 0, position = 0, inflight = 0;
        int error = 0;
        std::promise<int> done;
    };
    enum class Phase { READ, WRITE, READ_MODIFY };
    struct Slot {
        SpdkDevice* owner = nullptr;
        Batch* batch = nullptr;
        const FileIORequest* range = nullptr;
        std::array<iovec, 128> vectors{};
        unsigned vector_count = 0;
        void* dma = nullptr;
        void* io_data = nullptr;
        bool direct = false;
        std::uint64_t offset = 0;
        std::size_t length = 0, prefix = 0, bytes = 0, position = 0;
        Phase phase = Phase::READ;
        spdk_bdev_io_wait_entry wait{};
    };
    SpdkBdevConfig config_;
    std::vector<Slot> slots_;
    std::thread reactor_;
    std::mutex mutex_;
    mutable std::mutex allocation_mutex_;
    std::map<std::uintptr_t, std::pair<std::size_t, std::weak_ptr<Buffer>>> buffers_;
    std::size_t leased_bytes_ = 0;
    std::promise<void> started_;
    bool ready_ = false, removed_ = false;
    spdk_bdev_desc* desc_ = nullptr;
    spdk_bdev* bdev_ = nullptr;
    spdk_io_channel* channel_ = nullptr;
    spdk_thread* thread_ = nullptr;
    std::atomic<std::uint64_t> read_bytes_{0}, written_bytes_{0}, completed_{0};
    std::atomic<std::uint64_t> copied_read_{0}, copied_write_{0};
    std::atomic<std::uint64_t> direct_read_{0}, direct_written_{0};
    std::size_t environment_bytes_ = 0;
    bool cuda_dma_supported_ = false;

    void release_buffer(void* data, std::size_t bytes, bool cuda_pinned) noexcept {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        buffers_.erase(reinterpret_cast<std::uintptr_t>(data));
#if INFER_ENABLE_CUDA
        if (cuda_pinned && cudaHostUnregister(data) != cudaSuccess) {
            std::fputs("SPDK DMA buffer retained after CUDA unregister failure\n", stderr);
            return;
        }
#else
        (void)cuda_pinned;
#endif
        spdk_dma_free(data);
        leased_bytes_ -= bytes;
    }
    std::weak_ptr<Buffer> find_buffer(const void* data, std::size_t bytes) const {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        const auto address = reinterpret_cast<std::uintptr_t>(data);
        auto found = buffers_.upper_bound(address);
        if (found == buffers_.begin())
            return {};
        --found;
        const auto offset = address - found->first;
        return offset <= found->second.first && bytes <= found->second.first - offset
                   ? found->second.second : std::weak_ptr<Buffer>{};
    }
    std::vector<std::shared_ptr<Buffer>> retain_buffers(const std::vector<FileIORequest>& ranges) {
        std::vector<std::shared_ptr<Buffer>> result;
        result.reserve(ranges.size());
        for (const auto& range : ranges)
            result.push_back(find_buffer(range.data, range.bytes).lock());
        return result;
    }

    bool validate(const std::vector<FileIORequest>& ranges, bool write) const {
        if (ranges.size() > 128)
            throw std::invalid_argument("SPDK batch exceeds 128 ranges");
        bool shared_sector = false;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const auto& r = ranges[i];
            const auto address = reinterpret_cast<std::uintptr_t>(r.data);
            const auto prefix = reinterpret_cast<std::uintptr_t>(r.prefix);
            if (!r.data || !r.bytes || r.offset > config_.capacity_bytes ||
                r.prefix_bytes > config_.capacity_bytes - r.offset ||
                r.bytes > config_.capacity_bytes - r.offset - r.prefix_bytes ||
                r.bytes > UINTPTR_MAX - address ||
                (r.prefix_bytes && (!write || !r.prefix || r.prefix_bytes > UINTPTR_MAX - prefix)))
                throw std::invalid_argument("Invalid SPDK byte range");
            for (std::size_t j = 0; j < i; ++j) {
                const auto& other = ranges[j];
                if (write) {
                    const auto last = (r.offset + r.prefix_bytes + r.bytes - 1) / sector_alignment;
                    const auto other_last =
                        (other.offset + other.prefix_bytes + other.bytes - 1) / sector_alignment;
                    shared_sector |= r.offset / sector_alignment <= other_last &&
                                     other.offset / sector_alignment <= last;
                } else {
                    const auto other_address = reinterpret_cast<std::uintptr_t>(other.data);
                    if (address < other_address + other.bytes && other_address < address + r.bytes)
                        throw std::invalid_argument("Overlapping SPDK destinations");
                }
            }
        }
        return shared_sector;
    }
    void execute(const std::vector<FileIORequest>& ranges, bool write) {
        if (ranges.empty())
            return;
        const auto shared_sector = validate(ranges, write);
        // Synchronous callers share the bounded DMA slots. SPDK owns dispatch
        // and completion; no second submission queue or polling scheduler.
        std::lock_guard<std::mutex> lock(mutex_);
        const auto submit = [&](const std::vector<FileIORequest>& requests) {
            Batch batch{*this, requests, retain_buffers(requests), write};
            auto done = batch.done.get_future();
            const auto rc = spdk_thread_send_msg(thread_, begin, &batch);
            if (rc)
                throw std::system_error(-rc, std::generic_category(), "SPDK submit batch");
            if (const auto error = done.get())
                throw std::system_error(error, std::generic_category(), "SPDK bdev I/O");
        };
        // Distinct records have disjoint sectors. Generic byte writes sharing a
        // sector retain sequential RMW semantics instead of racing DMA buffers.
        if (shared_sector) {
            for (const auto& range : ranges)
                submit({range});
            return;
        }
        submit(ranges);
    }
    static void begin(void* argument) noexcept {
        auto& batch = *static_cast<Batch*>(argument);
        batch.inflight = batch.owner.slots_.size();
        for (auto& slot : batch.owner.slots_) {
            slot.owner = &batch.owner;
            slot.batch = &batch;
            batch.owner.advance(slot, 0);
        }
    }
    static void retry(void* argument) noexcept {
        auto& slot = *static_cast<Slot*>(argument);
        slot.owner->issue(slot);
    }
    void advance(Slot& slot, int error) noexcept {
        auto& batch = *slot.batch;
        if (error && !batch.error)
            batch.error = error;
        if (!batch.error && batch.index < batch.ranges.size()) {
            launch(slot, batch);
            return;
        }
        slot.batch = nullptr;
        if (--batch.inflight == 0)
            batch.done.set_value(batch.error);
    }
    static void completed(spdk_bdev_io* io, bool success, void* argument) noexcept {
        auto& slot = *static_cast<Slot*>(argument);
        auto& self = *slot.owner;
        spdk_bdev_free_io(io);
        self.completed_.fetch_add(1, std::memory_order_relaxed);
        if (!success) {
            self.advance(slot, EIO);
            return;
        }
        if (slot.vector_count) {
            const bool write = slot.phase == Phase::WRITE;
            (write ? self.written_bytes_ : self.read_bytes_).fetch_add(slot.length, std::memory_order_relaxed);
            for (unsigned i = 0; i < slot.vector_count; i += 2) {
                if (!write)
                    std::memcpy(slot.range[i].data, slot.vectors[i].iov_base, slot.range[i].bytes);
                (write ? self.copied_write_ : self.copied_read_).fetch_add(
                    slot.range[i].prefix_bytes + slot.range[i].bytes, std::memory_order_relaxed);
                (write ? self.direct_written_ : self.direct_read_).fetch_add(
                    slot.vectors[i + 1].iov_len, std::memory_order_relaxed);
            }
            self.advance(slot, 0);
            return;
        }
        if (slot.phase == Phase::WRITE) {
            self.written_bytes_.fetch_add(slot.length, std::memory_order_relaxed);
            if (slot.direct)
                self.direct_written_.fetch_add(slot.bytes, std::memory_order_relaxed);
            self.advance(slot, 0);
        } else {
            self.read_bytes_.fetch_add(slot.length, std::memory_order_relaxed);
            if (slot.phase == Phase::READ_MODIFY) {
                self.stage(slot);
                slot.phase = Phase::WRITE;
                self.issue(slot);
            } else {
                if (slot.direct)
                    self.direct_read_.fetch_add(slot.bytes, std::memory_order_relaxed);
                else {
                    std::memcpy(static_cast<char*>(slot.range->data) + slot.position,
                                static_cast<char*>(slot.dma) + slot.prefix, slot.bytes);
                    self.copied_read_.fetch_add(slot.bytes, std::memory_order_relaxed);
                }
                self.advance(slot, 0);
            }
        }
    }
    void issue(Slot& slot) noexcept {
        if (removed_) {
            advance(slot, ENODEV);
            return;
        }
        // ENOMEM means no I/O was accepted. Retain the slot/batch while waiting
        // on the same SPDK channel; completion or retry owns every DMA buffer.
        const int rc = slot.vector_count
                           ? (slot.phase == Phase::WRITE
                                  ? spdk_bdev_writev(desc_, channel_, slot.vectors.data(), slot.vector_count,
                                                     slot.offset, slot.length, completed, &slot)
                                  : spdk_bdev_readv(desc_, channel_, slot.vectors.data(), slot.vector_count,
                                                    slot.offset, slot.length, completed, &slot))
                           : slot.phase == Phase::WRITE
                           ? spdk_bdev_write(desc_, channel_, slot.io_data, slot.offset, slot.length,
                                             completed, &slot)
                           : spdk_bdev_read(desc_, channel_, slot.io_data, slot.offset, slot.length,
                                            completed, &slot);
        if (rc == -ENOMEM) {
            slot.wait.bdev = bdev_;
            slot.wait.cb_fn = retry;
            slot.wait.cb_arg = &slot;
            const int queued = spdk_bdev_queue_io_wait(bdev_, channel_, &slot.wait);
            if (queued)
                advance(slot, -queued);
        } else if (rc)
            advance(slot, -rc);
    }
    void stage(const Slot& slot) noexcept {
        copied_write_.fetch_add(slot.bytes, std::memory_order_relaxed);
        const auto& range = *slot.range;
        const auto head = slot.position < range.prefix_bytes
                              ? std::min(slot.bytes, range.prefix_bytes - slot.position)
                              : 0;
        auto* destination = static_cast<char*>(slot.dma) + slot.prefix;
        if (head)
            std::memcpy(destination, static_cast<const char*>(range.prefix) + slot.position, head);
        if (slot.bytes > head)
            std::memcpy(destination + head,
                        static_cast<const char*>(range.data) + (slot.position + head - range.prefix_bytes),
                        slot.bytes - head);
    }
    bool launch_scatter(Slot& slot, Batch& batch) noexcept {
        constexpr std::size_t max_chunk = 2 * 1024 * 1024;
        if (batch.position)
            return false;
        slot.range = &batch.ranges[batch.index];
        slot.offset = config_.offset_bytes + slot.range->offset;
        slot.length = 0;
        while (batch.index + 1 < batch.ranges.size() &&
               slot.vector_count + 2 <= slot.vectors.size() &&
               slot.vector_count / 2 < config_.io_buffer_bytes / sector_alignment) {
            const auto& head = batch.ranges[batch.index];
            const auto& payload = batch.ranges[batch.index + 1];
            if (batch.leases[batch.index] || !batch.leases[batch.index + 1] ||
                head.offset % sector_alignment || config_.offset_bytes + head.offset != slot.offset + slot.length ||
                payload.offset - head.offset != sector_alignment || payload.prefix_bytes ||
                slot.length >= max_chunk || sector_alignment + payload.bytes > max_chunk - slot.length ||
                payload.bytes % sector_alignment ||
                reinterpret_cast<std::uintptr_t>(payload.data) % sector_alignment ||
                (batch.write ? head.prefix_bytes + head.bytes != sector_alignment
                             : head.bytes > sector_alignment))
                break;
            auto* metadata = static_cast<char*>(slot.dma) + slot.vector_count / 2 * sector_alignment;
            if (batch.write) {
                if (head.prefix_bytes)
                    std::memcpy(metadata, head.prefix, head.prefix_bytes);
                std::memcpy(metadata + head.prefix_bytes, head.data, head.bytes);
            }
            slot.vectors[slot.vector_count++] = {metadata, sector_alignment};
            slot.vectors[slot.vector_count++] = {payload.data, payload.bytes};
            slot.length += sector_alignment + payload.bytes;
            batch.index += 2;
        }
        if (!slot.vector_count)
            return false;
        slot.phase = batch.write ? Phase::WRITE : Phase::READ;
        issue(slot);
        return true;
    }
    void launch(Slot& slot, Batch& batch) noexcept {
        slot.vector_count = 0;
        if (launch_scatter(slot, batch))
            return;
        const auto& range = batch.ranges[batch.index];
        slot.range = &range;
        slot.position = batch.position;
        const auto offset = config_.offset_bytes + range.offset + batch.position;
        slot.prefix = offset % sector_alignment;
        slot.offset = offset - slot.prefix;
        const auto remaining = range.prefix_bytes + range.bytes - batch.position;
        const auto address = reinterpret_cast<std::uintptr_t>(range.data) +
                             (batch.position >= range.prefix_bytes
                                  ? batch.position - range.prefix_bytes : 0);
        slot.direct = batch.leases[batch.index] && batch.position >= range.prefix_bytes &&
                      !slot.prefix && address % sector_alignment == 0 &&
                      remaining >= sector_alignment;
        slot.io_data = slot.direct ? reinterpret_cast<void*>(address) : slot.dma;
        // Split an aligned metadata prefix at the payload boundary so the
        // subsequent device-owned payload never passes through a bounce slot.
        const auto aligned_payload = batch.leases[batch.index] &&
                                     (range.offset + range.prefix_bytes) % sector_alignment == 0 &&
                                     reinterpret_cast<std::uintptr_t>(range.data) % sector_alignment == 0 &&
                                     range.bytes >= sector_alignment;
        const auto head = aligned_payload && batch.position < range.prefix_bytes
                              ? range.prefix_bytes - batch.position
                              : remaining;
        slot.bytes = slot.direct ? remaining / sector_alignment * sector_alignment
                                 : std::min(head, config_.io_buffer_bytes - slot.prefix);
        slot.length =
            (slot.prefix + slot.bytes + sector_alignment - 1) / sector_alignment * sector_alignment;
        slot.phase = !batch.write                                 ? Phase::READ
                     : (slot.prefix || slot.length != slot.bytes) ? Phase::READ_MODIFY
                                                                  : Phase::WRITE;
        batch.position += slot.bytes;
        if (batch.position == range.prefix_bytes + range.bytes) {
            ++batch.index;
            batch.position = 0;
        }
        if (slot.phase == Phase::WRITE && !slot.direct)
            stage(slot);
        issue(slot);
    }
    static void event(spdk_bdev_event_type type, spdk_bdev*, void* argument) noexcept {
        if (type == SPDK_BDEV_EVENT_REMOVE)
            static_cast<SpdkDevice*>(argument)->removed_ = true;
    }
    void cleanup() noexcept {
        if (channel_) {
            spdk_put_io_channel(channel_);
            channel_ = nullptr;
        }
        if (desc_) {
            spdk_bdev_close(desc_);
            desc_ = nullptr;
        }
        for (auto& slot : slots_)
            if (slot.dma) {
#if INFER_ENABLE_CUDA
                if (&slot == &slots_.front() && cuda_dma_supported_ &&
                    cudaHostUnregister(slot.dma) != cudaSuccess) {
                    std::fputs("SPDK CUDA probe buffer retained after unregister failure\n", stderr);
                    continue;
                }
#endif
                spdk_dma_free(slot.dma);
                slot.dma = nullptr;
            }
    }
    static void start(void* argument) noexcept {
        auto& self = *static_cast<SpdkDevice*>(argument);
        try {
            const auto rc =
                spdk_bdev_open_ext(self.config_.bdev_name.c_str(), true, event, &self, &self.desc_);
            if (rc)
                throw std::system_error(-rc, std::generic_category(), "SPDK open bdev");
            self.bdev_ = spdk_bdev_desc_get_bdev(self.desc_);
            const auto block = spdk_bdev_get_block_size(self.bdev_);
            const auto count = spdk_bdev_get_num_blocks(self.bdev_);
            if (!block || sector_alignment % block || count > UINT64_MAX / block ||
                spdk_bdev_get_md_size(self.bdev_) != 0 ||
                spdk_bdev_get_buf_align(self.bdev_) > sector_alignment)
                throw std::invalid_argument(
                    "SPDK requires ordinary sectors dividing 4 KiB and <=4 KiB DMA alignment");
            const auto capacity = count * block;
            if (self.config_.offset_bytes > capacity ||
                self.config_.capacity_bytes > capacity - self.config_.offset_bytes)
                throw std::invalid_argument("SPDK region exceeds bdev capacity");
            self.channel_ = spdk_bdev_get_io_channel(self.desc_);
            if (!self.channel_)
                throw std::runtime_error("SPDK I/O channel allocation failed");
            for (auto& slot : self.slots_) {
                slot.dma =
                    spdk_dma_zmalloc(self.config_.io_buffer_bytes, sector_alignment, nullptr);
                if (!slot.dma)
                    throw std::bad_alloc();
            }
            if (!snapshot_arena())
                throw std::runtime_error("SPDK environment accounting is unavailable");
            self.environment_bytes_ = arena_bytes.load(std::memory_order_relaxed);
            if (!self.environment_bytes_ ||
                self.environment_bytes_ > (std::uint64_t(self.config_.environment_mib) << 20))
                throw std::runtime_error("SPDK environment exceeds its configured budget");
#if INFER_ENABLE_CUDA
            // Probe real EAL-backed memory, which CUDA may refuse even when a GPU is present.
            self.cuda_dma_supported_ = register_cuda(self.slots_.front().dma,
                                                     self.config_.io_buffer_bytes);
#endif
            self.thread_ = spdk_get_thread();
            self.started_.set_value();
        } catch (...) {
            self.started_.set_exception(std::current_exception());
            self.cleanup();
            spdk_app_stop(-1);
        }
        self.ready_ = true;
    }
    void run() noexcept {
        spdk_app_opts opts;
        spdk_app_opts_init(&opts, sizeof(opts));
        opts.name = "miniinfer_storage";
        opts.json_config_file = config_.json_config.c_str();
        opts.reactor_mask = config_.reactor_mask.c_str();
        opts.rpc_addr = nullptr;
        opts.disable_signal_handlers = true;
        opts.no_huge = config_.no_huge;
        opts.no_pci = config_.no_pci;
        opts.iova_mode = config_.iova_mode == "auto" ? nullptr : config_.iova_mode.c_str();
        if (config_.iova_mode == "va")
            opts.base_virtaddr = UINT64_C(1) << 36; // Below limited IOMMU masks, above CUDA's low VA ranges.
        opts.mem_size = config_.environment_mib;
        opts.num_entries = 0;
        // A fixed EAL arena bounds DMA/mempools within the host reservation.
        // no_huge already selects legacy memory in env_dpdk.
        if (!config_.no_huge)
            opts.env_context = const_cast<char*>("--legacy-mem");
        const int rc = spdk_app_start(&opts, start, this);
        snapshot_arena();
        spdk_app_fini();
        // EAL keeps its original main-thread TLS. A subsequent std::thread
        // cannot safely reinitialize that app; leases defer this retirement.
        app_retired.store(true, std::memory_order_release);
        if (!ready_)
            started_.set_exception(std::make_exception_ptr(
                std::runtime_error("SPDK app initialization failed: " + std::to_string(rc))));
    }

  public:
    explicit SpdkDevice(SpdkBdevConfig config) : config_(std::move(config)) {
        if ((config_.iova_mode != "auto" && config_.iova_mode != "va" && config_.iova_mode != "pa") ||
            config_.bdev_name.empty() || config_.json_config.empty() ||
            config_.reactor_mask.empty() || config_.json_config.find('\0') != std::string::npos ||
            config_.bdev_name.find('\0') != std::string::npos || !config_.capacity_bytes ||
            config_.offset_bytes % sector_alignment || config_.capacity_bytes % sector_alignment ||
            config_.capacity_bytes > UINT64_MAX - config_.offset_bytes || !config_.queue_depth ||
            config_.queue_depth > 64 || config_.io_buffer_bytes < sector_alignment ||
            config_.io_buffer_bytes > 8 * 1024 * 1024 ||
            config_.io_buffer_bytes % sector_alignment ||
            config_.queue_depth > config_.dma_budget_bytes / config_.io_buffer_bytes ||
            config_.environment_mib < 64 || config_.environment_mib > 4096 ||
            config_.dma_budget_bytes > (std::uint64_t(config_.environment_mib) << 20))
            throw std::invalid_argument("Invalid or unbounded SPDK configuration");
        slots_.resize(config_.queue_depth);
        bool expected = false;
        if (!app_owned.compare_exchange_strong(expected, true))
            throw std::runtime_error("An SPDK app already owns this process");
        if (app_retired.load(std::memory_order_acquire)) {
            app_owned = false;
            throw std::runtime_error("SPDK app has shut down; create the next store in a new process");
        }
        try {
            auto started = started_.get_future();
            reactor_ = std::thread([this] { run(); });
            started.get();
        } catch (...) {
            if (reactor_.joinable())
                reactor_.join();
            app_owned = false;
            throw;
        }
    }
    ~SpdkDevice() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (spdk_thread_send_msg(thread_, [](void* argument) {
                static_cast<SpdkDevice*>(argument)->cleanup();
                spdk_app_stop(0);
            }, this))
            std::terminate();
        reactor_.join();
        app_owned = false;
    }
    std::size_t alignment() const noexcept override { return infer::sector_alignment; }
    std::uint64_t capacity_bytes() const noexcept override { return config_.capacity_bytes; }
    void read(void* host, std::size_t bytes, std::uint64_t offset) override {
        if (bytes)
            execute({{host, bytes, offset}}, false);
    }
    void write(const void* host, std::size_t bytes, std::uint64_t offset) override {
        if (bytes)
            execute({{const_cast<void*>(host), bytes, offset}}, true);
    }
    void read_many(const std::vector<FileIORequest>& requests) override {
        execute(requests, false);
    }
    void write_many(const std::vector<FileIORequest>& requests) override {
        execute(requests, true);
    }
    SSDDeviceStats stats() const override {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        return {config_.queue_depth * config_.io_buffer_bytes + leased_bytes_, environment_bytes_,
                read_bytes_.load(), written_bytes_.load(), completed_.load(),
                copied_read_.load(), copied_write_.load(), direct_read_.load(), direct_written_.load()};
    }
    std::shared_ptr<SSDDmaBuffer> allocate_dma_buffer(std::size_t bytes) override {
        return allocate_buffer(bytes);
    }
    bool owns_dma_range(const void* data, std::size_t bytes) const override {
        return !find_buffer(data, bytes).expired();
    }
    std::shared_ptr<SSDDmaBuffer> allocate_transfer_buffer(std::size_t bytes) override {
#if INFER_ENABLE_CUDA
        if (cuda_dma_supported_) {
            auto buffer = allocate_buffer(bytes);
            if (register_cuda(buffer->data(), buffer->size())) {
                buffer->cuda_pinned = true;
                return buffer;
            }
        }
#else
        (void)bytes;
#endif
        return {};
    }

  private:
    std::shared_ptr<Buffer> allocate_buffer(std::size_t bytes) {
        if (!bytes || bytes > SIZE_MAX - sector_alignment + 1)
            throw std::invalid_argument("Invalid SPDK DMA allocation size");
        const auto rounded = (bytes + sector_alignment - 1) / sector_alignment * sector_alignment;
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        const auto bounce = config_.queue_depth * config_.io_buffer_bytes;
        if (rounded > config_.dma_budget_bytes - bounce - leased_bytes_)
            throw std::bad_alloc();
        void* data = spdk_dma_zmalloc(rounded, sector_alignment, nullptr);
        if (!data)
            throw std::bad_alloc();
        // Construct the shared owner before publishing its weak registration.
        // Allocation bookkeeping failure must return the physical DMA memory.
        std::shared_ptr<Buffer> result;
        try {
            result = std::make_shared<Buffer>(shared_from_this(), data, rounded);
            buffers_.emplace(reinterpret_cast<std::uintptr_t>(data),
                             std::make_pair(rounded, std::weak_ptr<Buffer>(result)));
        } catch (...) {
            if (result) {
                result->address = nullptr;
                result->bytes = 0;
            }
            spdk_dma_free(data);
            throw;
        }
        leased_bytes_ += rounded;
        return result;
    }
};
class SpdkDeviceHandle final : public SSDDevice {
    std::shared_ptr<SpdkDevice> device_;
  public:
    explicit SpdkDeviceHandle(const SpdkBdevConfig& config)
        : device_(std::make_shared<SpdkDevice>(config)) {}
    std::size_t alignment() const noexcept override { return device_->alignment(); }
    std::uint64_t capacity_bytes() const noexcept override { return device_->capacity_bytes(); }
    void read(void* data, std::size_t bytes, std::uint64_t offset) override {
        device_->read(data, bytes, offset);
    }
    void write(const void* data, std::size_t bytes, std::uint64_t offset) override {
        device_->write(data, bytes, offset);
    }
    void read_many(const std::vector<FileIORequest>& ranges) override { device_->read_many(ranges); }
    void write_many(const std::vector<FileIORequest>& ranges) override { device_->write_many(ranges); }
    SSDDeviceStats stats() const override { return device_->stats(); }
    std::shared_ptr<SSDDmaBuffer> allocate_dma_buffer(std::size_t bytes) override {
        return device_->allocate_dma_buffer(bytes);
    }
    bool owns_dma_range(const void* data, std::size_t bytes) const override {
        return device_->owns_dma_range(data, bytes);
    }
    std::shared_ptr<SSDDmaBuffer> allocate_transfer_buffer(std::size_t bytes) override {
        return device_->allocate_transfer_buffer(bytes);
    }
};
} // namespace
bool spdk_available() noexcept {
    return true;
}
std::size_t spdk_environment_bytes() noexcept {
    return arena_bytes.load(std::memory_order_relaxed);
}
std::unique_ptr<SSDDevice> open_spdk_device(const SpdkBdevConfig& config) {
    return std::make_unique<SpdkDeviceHandle>(config);
}
} // namespace infer
#else
namespace infer {
bool spdk_available() noexcept {
    return false;
}
std::size_t spdk_environment_bytes() noexcept {
    return 0;
}
std::unique_ptr<SSDDevice> open_spdk_device(const SpdkBdevConfig&) {
    throw std::runtime_error(
        "SPDK support is disabled; configure with INFER_ENABLE_SPDK=ON and an installed SPDK SDK");
}
} // namespace infer
#endif
