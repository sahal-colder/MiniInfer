#include "infer/storage/ssd_device.h"

#include <stdexcept>

#ifdef INFER_HAS_SPDK
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <spdk/bdev.h>
#include <spdk/env.h>
#include <spdk/env_dpdk.h>
#include <spdk/event.h>
#include <spdk/thread.h>
#include <system_error>
#include <thread>

namespace infer {
namespace {
std::atomic<bool> app_owned{false};
std::atomic<std::size_t> arena_bytes{0};
constexpr std::size_t sector_alignment = 4096;
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

class SpdkDevice final : public SSDDevice {
    struct Batch {
        SpdkDevice& owner;
        const std::vector<FileIORequest>& ranges;
        bool write;
        std::size_t index = 0, position = 0, inflight = 0;
        int error = 0;
        std::promise<int> done;
    };
    enum class Phase { READ, WRITE, READ_MODIFY };
    struct Slot {
        SpdkDevice* owner = nullptr;
        Batch* batch = nullptr;
        void* dma = nullptr;
        char* host = nullptr;
        std::uint64_t offset = 0;
        std::size_t length = 0, prefix = 0, bytes = 0;
        Phase phase = Phase::READ;
        spdk_bdev_io_wait_entry wait{};
    };
    SpdkBdevConfig config_;
    std::vector<Slot> slots_;
    std::thread reactor_;
    std::mutex mutex_;
    std::promise<void> started_;
    bool ready_ = false, removed_ = false;
    spdk_bdev_desc* desc_ = nullptr;
    spdk_bdev* bdev_ = nullptr;
    spdk_io_channel* channel_ = nullptr;
    spdk_thread* thread_ = nullptr;
    std::atomic<std::uint64_t> read_bytes_{0}, written_bytes_{0}, completed_{0};
    std::size_t environment_bytes_ = 0;

    void validate(const std::vector<FileIORequest>& ranges, bool write) const {
        if (ranges.size() > 128)
            throw std::invalid_argument("SPDK batch exceeds 128 ranges");
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const auto& r = ranges[i];
            const auto address = reinterpret_cast<std::uintptr_t>(r.data);
            if (!r.data || !r.bytes || r.offset > config_.capacity_bytes ||
                r.bytes > config_.capacity_bytes - r.offset || r.bytes > UINTPTR_MAX - address)
                throw std::invalid_argument("Invalid SPDK byte range");
            if (!write)
                for (std::size_t j = 0; j < i; ++j) {
                    const auto other = reinterpret_cast<std::uintptr_t>(ranges[j].data);
                    if (address < other + ranges[j].bytes && other < address + r.bytes)
                        throw std::invalid_argument("Overlapping SPDK destinations");
                }
        }
    }
    void execute(const std::vector<FileIORequest>& ranges, bool write) {
        if (ranges.empty())
            return;
        validate(ranges, write);
        // Synchronous callers share the bounded DMA slots. SPDK owns dispatch
        // and completion; no second submission queue or polling scheduler.
        std::lock_guard<std::mutex> lock(mutex_);
        Batch batch{*this, ranges, write};
        auto done = batch.done.get_future();
        const auto rc = spdk_thread_send_msg(thread_, begin, &batch);
        if (rc)
            throw std::system_error(-rc, std::generic_category(), "SPDK submit batch");
        if (const auto error = done.get())
            throw std::system_error(error, std::generic_category(), "SPDK bdev I/O");
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
        if (slot.phase == Phase::WRITE) {
            self.written_bytes_.fetch_add(slot.length, std::memory_order_relaxed);
            self.advance(slot, 0);
        } else {
            self.read_bytes_.fetch_add(slot.length, std::memory_order_relaxed);
            if (slot.phase == Phase::READ_MODIFY) {
                std::memcpy(static_cast<char*>(slot.dma) + slot.prefix, slot.host, slot.bytes);
                slot.phase = Phase::WRITE;
                self.issue(slot);
            } else {
                std::memcpy(slot.host, static_cast<char*>(slot.dma) + slot.prefix, slot.bytes);
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
        const int rc = slot.phase == Phase::WRITE
                           ? spdk_bdev_write(desc_, channel_, slot.dma, slot.offset, slot.length,
                                             completed, &slot)
                           : spdk_bdev_read(desc_, channel_, slot.dma, slot.offset, slot.length,
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
    void launch(Slot& slot, Batch& batch) noexcept {
        const auto& range = batch.ranges[batch.index];
        const auto offset = config_.offset_bytes + range.offset + batch.position;
        slot.prefix = offset % sector_alignment;
        slot.offset = offset - slot.prefix;
        slot.bytes = std::min(range.bytes - batch.position, config_.io_buffer_bytes - slot.prefix);
        slot.length =
            (slot.prefix + slot.bytes + sector_alignment - 1) / sector_alignment * sector_alignment;
        slot.host = static_cast<char*>(range.data) + batch.position;
        slot.phase = !batch.write                                 ? Phase::READ
                     : (slot.prefix || slot.length != slot.bytes) ? Phase::READ_MODIFY
                                                                  : Phase::WRITE;
        batch.position += slot.bytes;
        if (batch.position == range.bytes) {
            ++batch.index;
            batch.position = 0;
        }
        if (slot.phase == Phase::WRITE)
            std::memcpy(slot.dma, slot.host, slot.bytes);
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
        opts.mem_size = config_.environment_mib;
        opts.num_entries = 0;
        // A fixed EAL arena bounds DMA/mempools within the host reservation.
        // no_huge already selects legacy memory in env_dpdk.
        if (!config_.no_huge)
            opts.env_context = const_cast<char*>("--legacy-mem");
        const int rc = spdk_app_start(&opts, start, this);
        snapshot_arena();
        spdk_app_fini();
        if (!ready_)
            started_.set_exception(std::make_exception_ptr(
                std::runtime_error("SPDK app initialization failed: " + std::to_string(rc))));
    }

  public:
    explicit SpdkDevice(SpdkBdevConfig config) : config_(std::move(config)) {
        if (config_.bdev_name.empty() || config_.json_config.empty() ||
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
    SSDDeviceStats stats() const override {
        return {config_.queue_depth * config_.io_buffer_bytes, environment_bytes_,
                read_bytes_.load(), written_bytes_.load(), completed_.load()};
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
    return std::make_unique<SpdkDevice>(config);
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
