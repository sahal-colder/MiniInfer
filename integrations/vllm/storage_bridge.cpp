// 实验用 C ABI：vLLM 负责 token 哈希、淘汰和 DMA，项目 SSD 后端负责文件与校验。
#include "host_io_queue.h"
#include "native_transfer.h"
#include "native_staging.h"
#include "infer/storage/kv_offload_cache.h"
#include "infer/storage/storage_backend.h"
#include "infer/storage/tiered_storage.h"
#include "infer/io/direct_file.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

namespace {
thread_local std::string last_error;
std::uint64_t env_number(const char* key, std::uint64_t fallback) {
    const char* value = std::getenv(key);
    if (!value)
        return fallback;
    std::uint64_t result = 0;
    const char* end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end || end == value)
        throw std::invalid_argument(std::string("Invalid numeric environment: ") + key);
    return result;
}
infer::SSDVerification ssd_verification(infer::SSDVerification fallback) {
    const char* value = std::getenv("MINIINFER_SSD_VERIFY");
    if (!value)
        return fallback;
    if (std::strcmp(value, "none") == 0)
        return infer::SSDVerification::NONE;
    if (std::strcmp(value, "read_back") == 0)
        return infer::SSDVerification::READ_BACK;
    if (std::strcmp(value, "on_read") == 0)
        return infer::SSDVerification::ON_READ;
    throw std::invalid_argument("MINIINFER_SSD_VERIFY must be none, read_back or on_read");
}
infer::FileIOConfig file_io_config() {
    infer::FileIOConfig result;
    result.mode = infer::FileIOMode::POSIX;
    if (const char* value = std::getenv("MINIINFER_FILE_IO")) {
        if (std::strcmp(value, "posix") == 0)
            result.mode = infer::FileIOMode::POSIX;
        else if (std::strcmp(value, "uring") == 0 || std::strcmp(value, "direct") == 0)
            result.mode = infer::FileIOMode::IO_URING;
        else if (std::strcmp(value, "auto") == 0)
            result.mode = infer::FileIOMode::AUTO;
        else
            throw std::invalid_argument("MINIINFER_FILE_IO must be auto, posix, uring or direct");
    }
    result.file_shards = env_number("MINIINFER_FILE_SHARDS", 4);
    result.queue_depth = env_number("MINIINFER_IO_DEPTH", 64);
    const auto force = env_number("MINIINFER_IO_ASYNC", 0);
    if (force > 1)
        throw std::invalid_argument("MINIINFER_IO_ASYNC must be 0 or 1");
    result.force_async = force != 0;
    return result;
}
void append_file_io_stats(std::ostream& out, const infer::FileIOStats& s) {
    out << ",\"uring_registered_file\":" << (s.registered_file ? "true" : "false")
        << ",\"uring_read_operations\":" << s.read_operations
        << ",\"uring_write_operations\":" << s.write_operations
        << ",\"uring_read_bytes\":" << s.read_bytes << ",\"uring_written_bytes\":" << s.write_bytes
        << ",\"uring_submissions\":" << s.submissions << ",\"uring_completions\":" << s.completions
        << ",\"uring_peak_inflight\":" << s.peak_inflight
        << ",\"uring_completion_waits\":" << s.completion_waits << ",\"uring_rings\":" << s.rings
        << ",\"uring_setup_flags\":" << s.setup_flags
        << ",\"uring_registered_buffer_bytes\":" << s.registered_buffer_bytes
        << ",\"uring_fixed_buffer_operations\":" << s.fixed_buffer_operations
        << ",\"uring_ordinary_buffer_operations\":" << s.ordinary_buffer_operations
        << ",\"uring_buffer_registration_failures\":" << s.buffer_registration_failures
        << ",\"uring_buffer_window_registrations\":" << s.buffer_window_registrations
        << ",\"uring_hugetlb_pools\":" << s.hugetlb_pools;
}
void append_device_stats(std::ostream& out, const infer::SSDDeviceStats& s) {
    out << ",\"ssd_environment_bytes\":" << s.environment_bytes
        << ",\"ssd_dma_reserved_bytes\":" << s.dma_reserved_bytes
        << ",\"ssd_device_read_bytes\":" << s.read_bytes
        << ",\"ssd_device_written_bytes\":" << s.written_bytes
        << ",\"ssd_device_completed_ios\":" << s.completed_ios
        << ",\"ssd_copied_read_bytes\":" << s.copied_read_bytes
        << ",\"ssd_copied_write_bytes\":" << s.copied_write_bytes
        << ",\"ssd_direct_read_bytes\":" << s.direct_read_bytes
        << ",\"ssd_direct_written_bytes\":" << s.direct_written_bytes;
}
void configure_spdk(infer::KVOffloadConfig& config) {
    const char* name = std::getenv("MINIINFER_SPDK_BDEV");
    if (!name || !*name)
        return;
    auto& spdk = config.ssd_spdk;
    if (const char* mode = std::getenv("MINIINFER_SPDK_IOVA_MODE"))
        spdk.iova_mode = mode;
    spdk.bdev_name = name;
    const char* json = std::getenv("MINIINFER_SPDK_CONFIG");
    if (!json || !*json)
        throw std::invalid_argument("MINIINFER_SPDK_CONFIG is required");
    spdk.json_config = json;
    if (const char* mask = std::getenv("MINIINFER_SPDK_REACTOR_MASK"))
        spdk.reactor_mask = mask;
    spdk.capacity_bytes = env_number("MINIINFER_SPDK_CAPACITY_BYTES", 0);
    spdk.offset_bytes = env_number("MINIINFER_SPDK_OFFSET_BYTES", 0);
    const auto memory = env_number("MINIINFER_SPDK_MEM_MIB", 256);
    const auto no_huge = env_number("MINIINFER_SPDK_NO_HUGE", 0),
               no_pci = env_number("MINIINFER_SPDK_NO_PCI", 0);
    if (memory < 64 || memory > 4096 || no_huge > 1 || no_pci > 1)
        throw std::invalid_argument("Invalid SPDK environment options");
    spdk.environment_mib = static_cast<int>(memory);
    spdk.no_huge = no_huge;
    spdk.no_pci = no_pci;
    spdk.queue_depth = config.ssd_io.queue_depth;
    const auto kib = env_number("MINIINFER_SPDK_CHUNK_KIB", 256);
    if (!kib || kib > 8192)
        throw std::invalid_argument("Invalid SPDK I/O chunk size");
    spdk.io_buffer_bytes = kib * 1024;
    // The entire EAL arena is already charged to the host budget. Bounce slots
    // and CUDA-visible staging leases share it instead of a second allocation.
    spdk.dma_budget_bytes = memory * 1024 * 1024;
}
struct TieredStore {
    infer::KVOffloadCache cache;
    std::mutex operations_mutex;
    std::map<std::uint64_t, std::shared_future<bool>> operations;
    std::uint64_t next_operation = 1;
    std::uint64_t rejected = 0, wait_ns = 0;
    std::size_t bytes, capacity;
    HostIOQueue queue;
    explicit TieredStore(infer::KVOffloadConfig config)
        : cache(prepare(config)), bytes(config.payload_bytes), capacity(config.logical_slots),
          queue(
              [this](bool write, const auto& slots, const auto& buffers) {
                  if (write)
                      cache.write_many(slots, buffers);
                  else
                      cache.read_many(slots, buffers);
              },
              config.payload_bytes) {}
    static infer::KVOffloadConfig prepare(infer::KVOffloadConfig config) {
        config.ssd_verification = ssd_verification(config.ssd_verification);
        if (const auto mode = std::getenv("MINIINFER_FILE_IO"))
            config.ssd_direct = std::strcmp(mode, "direct") == 0;
        const auto force = env_number("MINIINFER_IO_ASYNC", 0);
        if (force > 1)
            throw std::invalid_argument("MINIINFER_IO_ASYNC must be 0 or 1");
        config.ssd_io.force_async = force != 0;
        config.ssd_io.file_shards = env_number("MINIINFER_FILE_SHARDS", 4);
        configure_spdk(config);
        return config;
    }
};
struct Slot {
    std::mutex mutex;
    infer::StorageHandle handle = 0;
    infer::BlockMeta meta;
};
struct Store {
    std::size_t bytes;
    infer::SSDStorageBackend backend;
    std::vector<Slot> slots;
    HostIOQueue queue;
    Store(std::size_t payload, std::size_t slot_bytes, std::size_t count, const char* directory)
        : bytes(payload),
          backend(open_backend(slot_bytes, count, directory)),
          slots(count), queue([this](bool write, const auto& indices,
                                     const auto& buffers) { batch(write, indices, buffers); },
                              payload, payload>=1024*1024 ? 4 : 1) {
        if (!payload || payload > slot_bytes)
            throw std::invalid_argument("Invalid payload size");
    }
    void batch(bool write, const std::vector<std::size_t>& indices,
               const std::vector<void*>& buffers) {
        std::vector<std::size_t> order = indices;
        std::sort(order.begin(), order.end());
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(order.size());
        for (auto index : order)
            locks.emplace_back(slots.at(index).mutex);
        std::vector<infer::StorageWrite> writes;
        std::vector<infer::StorageRead> reads;
        writes.reserve(indices.size());
        reads.reserve(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto& slot = slots.at(indices[i]);
            if (write) {
                if (slot.meta.version == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("Version overflow");
            } else if (!slot.handle) {
                throw std::logic_error("Slot has not been written");
            }
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto& slot = slots[indices[i]];
            if (write) {
                if (!slot.handle)
                    slot.handle = backend.reserve();
                slot.meta.id = indices[i] + 1;
                slot.meta.size = bytes;
                ++slot.meta.version;
                writes.push_back({slot.handle, slot.meta, buffers[i]});
            } else {
                reads.push_back({slot.handle, slot.meta, buffers[i]});
            }
        }
        if (write)
            backend.write_published_many(writes);
        else
            // Large requests already run on independent queue workers. Preserve
            // expected identity/version and CRC checks without nested fan-out.
            backend.read_validated_many(reads,bytes<1024*1024);
    }
    static infer::SSDStorageBackend open_backend(std::size_t size, std::size_t count,
                                                const char* directory) {
        infer::KVOffloadConfig config;
        config.ssd_io = file_io_config();
        configure_spdk(config);
        const auto verification = ssd_verification(infer::default_ssd_verification);
        if (const auto mode = std::getenv("MINIINFER_FILE_IO"); mode && std::strcmp(mode, "direct") == 0) {
            if (!config.ssd_spdk.bdev_name.empty())
                throw std::invalid_argument("Choose direct-file or SPDK, not both");
            const auto limit = budget(size, count, 4096);
            return infer::SSDStorageBackend(size, limit,
                infer::io::open_direct_file_device(directory, limit, config.ssd_io.queue_depth), verification);
        }
        if (!config.ssd_spdk.bdev_name.empty()) {
            auto device = infer::open_spdk_device(config.ssd_spdk);
            const auto limit = budget(size, count, device->alignment());
            return infer::SSDStorageBackend(size, limit, std::move(device), verification);
        }
        return infer::SSDStorageBackend(size, budget(size, count), directory,
                                        config.ssd_io, verification);
    }
    static std::size_t budget(std::size_t size, std::size_t count, std::size_t alignment = 1) {
        const auto record = infer::SSDStorageBackend::allocation_size(size, alignment);
        if (!count || count > std::numeric_limits<std::size_t>::max() / record)
            throw std::invalid_argument("Invalid slot capacity");
        return count * record;
    }
};
void* require_store(void* ptr) {
    if (!ptr)
        throw std::invalid_argument("Null storage");
    return ptr;
}
void validate_batch(std::size_t count, const std::size_t* slots, void* const* buffers,
                    std::size_t capacity, std::size_t bytes, bool write) {
    if (!count || count > 64 || !slots || !buffers)
        throw std::invalid_argument("Invalid storage batch");
    for (std::size_t i = 0; i < count; ++i) {
        const auto address = reinterpret_cast<std::uintptr_t>(buffers[i]);
        if (slots[i] >= capacity || !address || bytes > UINTPTR_MAX - address)
            throw std::invalid_argument("Invalid storage batch slot or buffer");
        for (std::size_t j = 0; j < i; ++j) {
            if (slots[i] == slots[j])
                throw std::invalid_argument("Duplicate storage batch slot");
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[j]);
            if (!write && address < other + bytes && other < address + bytes)
                throw std::invalid_argument("Overlapping storage batch destinations");
        }
    }
}
template <class F> int checked(F action) noexcept {
    try {
        action();
        last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    } catch (...) {
        last_error = "Unknown C++ storage error";
        return -1;
    }
}
} // namespace
extern "C" {
int mi_transfer_available() { return MINIINFER_NATIVE_TRANSFERS; }
#if MINIINFER_NATIVE_TRANSFERS
void* mi_transfer_create(void* pointer, int tiered, std::size_t gpu_blocks,
                         std::size_t buffer_slots, std::size_t layers,
                         const CUdeviceptr* gpu_bases, const std::size_t* layer_bytes,
                         void* store_buffer, void* load_buffer, CUstream store_stream,
                         CUstream load_stream, std::size_t descriptor_cap, int read_ahead) {
    NativeTransfer* result = nullptr;
    checked([&] {
        require_store(pointer);
        if ((tiered != 0 && tiered != 1) || (read_ahead != 0 && read_ahead != 1) ||
            !layers || !gpu_bases || !layer_bytes)
            throw std::invalid_argument("Invalid native transfer configuration");
        const std::vector<CUdeviceptr> bases(gpu_bases, gpu_bases + layers);
        const std::vector<std::size_t> sizes(layer_bytes, layer_bytes + layers);
        if (tiered) {
            auto& store = *static_cast<TieredStore*>(pointer);
            result = new NativeTransfer(
                [&store](bool write, const auto& slots, const auto& buffers) {
                    if (write) store.cache.write_many(slots, buffers);
                    else store.cache.read_many(slots, buffers);
                }, store.queue, store.bytes, buffer_slots, gpu_blocks, store.capacity,
                bases, sizes, store_buffer, load_buffer, store_stream, load_stream,
                descriptor_cap, read_ahead);
            return;
        }
        auto& store = *static_cast<Store*>(pointer);
        result = new NativeTransfer(
            [&store](bool write, const auto& slots, const auto& buffers) {
                store.batch(write, slots, buffers);
            }, store.queue, store.bytes, buffer_slots, gpu_blocks, store.slots.size(),
            bases, sizes, store_buffer, load_buffer, store_stream, load_stream,
            descriptor_cap, read_ahead);
    });
    return result;
}
int mi_transfer_execute(void* pointer, int write, std::size_t count,
                        const std::size_t* gpu, const std::size_t* disk,
                        NativeTransfer::Result* result) {
    return checked([&] {
        if (!pointer || !result || (write != 0 && write != 1))
            throw std::invalid_argument("Invalid native transfer call");
        *result = {};
        static_cast<NativeTransfer*>(pointer)->execute(write, gpu, disk, count, *result);
    });
}
void mi_transfer_destroy(void* pointer) { delete static_cast<NativeTransfer*>(pointer); }
#endif
void* mi_store_staging_tensor(void* pointer, std::size_t bytes, int device) {
    void* result = nullptr;
    checked([&] {
        auto& store = *static_cast<Store*>(require_store(pointer));
        auto buffer = store.backend.allocate_dma_buffer(bytes);
        if (!buffer)
            throw std::runtime_error("Store does not expose device-owned DMA staging");
        result = miniinfer_staging::make_tensor(std::move(buffer), bytes, device);
    });
    return result;
}
void* mi_tiered_staging_tensor(void* pointer, std::size_t bytes, int device) {
    void* result = nullptr;
    checked([&] {
        auto& store = *static_cast<TieredStore*>(require_store(pointer));
        auto buffer = store.cache.allocate_dma_buffer(bytes);
        if (!buffer)
            throw std::runtime_error("Store does not expose device-owned DMA staging");
        result = miniinfer_staging::make_tensor(std::move(buffer), bytes, device);
    });
    return result;
}
void mi_store_delete_staging_tensor(void* pointer) {
    miniinfer_staging::destroy_tensor(static_cast<miniinfer_staging::ManagedTensor*>(pointer));
}
int mi_spdk_available() {
    return infer::spdk_available() ? 1 : 0;
}
std::size_t mi_spdk_environment_bytes() {
    return infer::spdk_environment_bytes();
}
const char* mi_storage_error() {
    return last_error.c_str();
}
void* mi_storage_create(std::size_t payload, std::size_t slots, const char* directory) {
    Store* result = nullptr;
    checked([&] {
        if (!directory)
            throw std::invalid_argument("Missing SSD directory");
        std::size_t size = 4096;
        while (size < payload) {
            if (size > std::numeric_limits<std::size_t>::max() / 2)
                throw std::overflow_error("Slot overflow");
            size *= 2;
        }
        result = new Store(payload, size, slots, directory);
    });
    return result;
}
int mi_storage_write(void* ptr, std::size_t index, const void* data) {
    return checked([&] {
        if (!ptr || !data)
            throw std::invalid_argument("Null storage or buffer");
        auto& store = *static_cast<Store*>(ptr);
        store.batch(true, {index}, {const_cast<void*>(data)});
    });
}
int mi_storage_read(void* ptr, std::size_t index, void* data) {
    return checked([&] {
        if (!ptr || !data)
            throw std::invalid_argument("Null storage or buffer");
        static_cast<Store*>(ptr)->batch(false, {index}, {data});
    });
}
int mi_storage_write_many(void* ptr, std::size_t count, const std::size_t* slots,
                          void* const* buffers) {
    return checked([&] {
        auto& store = *static_cast<Store*>(require_store(ptr));
        validate_batch(count, slots, buffers, store.slots.size(), store.bytes, true);
        store.batch(true, {slots, slots + count}, {buffers, buffers + count});
    });
}
int mi_storage_read_many(void* ptr, std::size_t count, const std::size_t* slots,
                         void* const* buffers) {
    return checked([&] {
        auto& store = *static_cast<Store*>(require_store(ptr));
        validate_batch(count, slots, buffers, store.slots.size(), store.bytes, false);
        store.batch(false, {slots, slots + count}, {buffers, buffers + count});
    });
}
int mi_storage_io_many(void* ptr, int write, std::size_t count, const std::size_t* slots,
                       void* const* buffers) {
    return checked([&] {
        auto& store = *static_cast<Store*>(require_store(ptr));
        if (write != 0 && write != 1)
            throw std::invalid_argument("Invalid storage I/O direction");
        validate_batch(count, slots, buffers, store.slots.size(), store.bytes, write);
        store.batch(write, {slots, slots + count}, {buffers, buffers + count});
    });
}
int mi_storage_submit(void* ptr, int write, std::size_t count, const std::size_t* slots,
                      void* const* buffers, std::uint64_t* token) {
    return checked([&] {
        auto& store = *static_cast<Store*>(require_store(ptr));
        if (!token || (write != 0 && write != 1))
            throw std::invalid_argument("Invalid native I/O submission");
        validate_batch(count, slots, buffers, store.slots.size(), store.bytes, write);
        *token = store.queue.submit(write, {slots, slots + count}, {buffers, buffers + count});
    });
}
int mi_storage_wait(void* ptr, std::uint64_t token) {
    return checked([&] { static_cast<Store*>(require_store(ptr))->queue.wait(token); });
}
int mi_tiered_submit(void* ptr, int write, std::size_t count, const std::size_t* slots,
                     void* const* buffers, std::uint64_t* token) {
    return checked([&] {
        auto& store = *static_cast<TieredStore*>(require_store(ptr));
        if (!token || (write != 0 && write != 1))
            throw std::invalid_argument("Invalid native I/O submission");
        validate_batch(count, slots, buffers, store.capacity, store.bytes, write);
        *token = store.queue.submit(write, {slots, slots + count}, {buffers, buffers + count});
    });
}
int mi_tiered_wait(void* ptr, std::uint64_t token) {
    return checked([&] { static_cast<TieredStore*>(require_store(ptr))->queue.wait(token); });
}
const char* mi_storage_stats(void* ptr) {
    thread_local std::string serialized;
    const auto status = checked([&] {
        if (!ptr)
            throw std::invalid_argument("Null storage");
        const auto s = static_cast<Store*>(ptr)->backend.io_stats();
        std::ostringstream out;
        out << "{\"ssd_payload_written_bytes\":" << s.payload_written_bytes
            << ",\"ssd_payload_read_bytes\":" << s.payload_read_bytes
            << ",\"ssd_verification_read_bytes\":" << s.verification_read_bytes
            << ",\"record_writes\":" << s.record_writes
            << ",\"source_crc_bytes\":" << s.source_crc_bytes
            << ",\"io_mode\":" << int(s.file_io.selected)
            << ",\"io_queue_depth\":" << s.file_io.queue_depth;
        out << ",\"native_queue_peak\":" << static_cast<Store*>(ptr)->queue.peak();
        append_file_io_stats(out, s.file_io);
        append_device_stats(out, s.device);
        out << '}';
        serialized = out.str();
    });
    return status == 0 ? serialized.c_str() : nullptr;
}
void mi_storage_destroy(void* ptr) {
    delete static_cast<Store*>(ptr);
}

void* mi_tiered_create(std::size_t payload, std::size_t slots, std::size_t cpu_slots,
                       const char* directory) {
    TieredStore* result = nullptr;
    checked([&] {
        if (!directory)
            throw std::invalid_argument("Missing SSD directory");
        result = new TieredStore({payload, slots, cpu_slots, directory});
    });
    return result;
}
void* mi_tiered_create_ex(std::size_t payload, std::size_t slots, std::size_t cpu_slots,
                          const char* directory, int io_mode, std::size_t depth,
                          std::size_t writes) {
    TieredStore* result = nullptr;
    checked([&] {
        if (!directory || io_mode < 0 || io_mode > 2)
            throw std::invalid_argument("Invalid offload I/O configuration");
        infer::KVOffloadConfig config{payload, slots, cpu_slots, directory};
        config.ssd_io = {static_cast<infer::FileIOMode>(io_mode), depth};
        config.write_concurrency = writes;
        result = new TieredStore(config);
    });
    return result;
}
void* mi_tiered_create_options(std::size_t payload, std::size_t slots, std::size_t cpu_slots,
                               const char* directory, int io_mode, std::size_t depth,
                               std::size_t writes, unsigned options) {
    TieredStore* result = nullptr;
    checked([&] {
        if (!directory || io_mode < 0 || io_mode > 2 || (options & ~15u))
            throw std::invalid_argument("Invalid tiered options");
        infer::KVOffloadConfig config{payload, slots, cpu_slots, directory};
        config.ssd_io = {static_cast<infer::FileIOMode>(io_mode), depth};
        config.write_concurrency = writes;
        config.compact_slots = (options & 1u) != 0;
        config.batch_promotions = (options & 2u) ? 2 : 0;
        config.overlap_writeback = (options & 4u) != 0;
        config.protect_hot_cpu = (options & 8u) != 0;
        result = new TieredStore(config);
    });
    return result;
}
int mi_tiered_write(void* ptr, std::size_t index, const void* data) {
    return checked([&] {
        if (!ptr)
            throw std::invalid_argument("Null tiered cache");
        static_cast<TieredStore*>(ptr)->cache.write(index, data);
    });
}
int mi_tiered_read(void* ptr, std::size_t index, void* data) {
    return checked([&] {
        if (!ptr)
            throw std::invalid_argument("Null tiered cache");
        static_cast<TieredStore*>(ptr)->cache.read(index, data);
    });
}
int mi_tiered_io_many(void* ptr, int write, std::size_t count, const std::size_t* slots,
                      void* const* buffers) {
    return checked([&] {
        if (!ptr || !slots || !buffers || !count || count > 64)
            throw std::invalid_argument("Invalid tiered I/O batch");
        if (write != 0 && write != 1)
            throw std::invalid_argument("Invalid storage I/O direction");
        auto& cache = static_cast<TieredStore*>(ptr)->cache;
        const std::vector<std::size_t> indices(slots, slots + count);
        const std::vector<void*> data(buffers, buffers + count);
        if (write)
            cache.write_many(indices, data);
        else
            cache.read_many(indices, data);
    });
}
const char* mi_tiered_stats(void* ptr) {
    thread_local std::string serialized;
    const auto status = checked([&] {
        if (!ptr)
            throw std::invalid_argument("Null tiered cache");
        auto& store = *static_cast<TieredStore*>(ptr);
        std::lock_guard<std::mutex> lock(store.operations_mutex);
        const auto s = store.cache.stats();
        std::ostringstream out;
        out << "{\"writes\":" << s.writes << ",\"cpu_hits\":" << s.cpu_hits << ",\"ssd_engine\":\""
            << (s.ssd_device.environment_bytes ? "spdk" : "file") << "\""
            << ",\"spdk_environment_bytes\":" << s.ssd_device.environment_bytes
            << ",\"spdk_dma_bytes\":" << s.ssd_device.dma_reserved_bytes
            << ",\"spdk_read_bytes\":" << s.ssd_device.read_bytes
            << ",\"spdk_written_bytes\":" << s.ssd_device.written_bytes
            << ",\"spdk_completed_ios\":" << s.ssd_device.completed_ios
            << ",\"ssd_record_writes\":" << s.ssd_record_writes
            << ",\"ssd_source_crc_bytes\":" << s.ssd_source_crc_bytes
            << ",\"overlapped_writes\":" << s.overlapped_writes
            << ",\"protected_cpu_slots\":" << s.protected_cpu_slots
            << ",\"hot_promotions\":" << s.hot_promotions
            << ",\"writer_wait_ns\":" << s.writer_wait_ns << ",\"ssd_hits\":" << s.ssd_hits
            << ",\"demotions\":" << s.demotions << ",\"populated_slots\":" << s.populated_slots
            << ",\"slot_bytes\":" << s.slot_bytes
            << ",\"cpu_reserved_bytes\":" << s.cpu_reserved_bytes
            << ",\"cpu_limit_bytes\":" << s.cpu_limit_bytes
            << ",\"cpu_pool_reserved_bytes\":" << s.cpu_pool_reserved_bytes
            << ",\"ssd_reserved_bytes\":" << s.ssd_reserved_bytes
            << ",\"ssd_limit_bytes\":" << s.ssd_limit_bytes << ",\"promotions\":" << s.promotions
            << ",\"read_ns\":" << s.read_ns << ",\"write_ns\":" << s.write_ns
            << ",\"lock_wait_ns\":" << s.lock_wait_ns << ",\"demotion_ns\":" << s.demotion_ns
            << ",\"ssd_reserve_ns\":" << s.ssd_reserve_ns << ",\"ssd_read_ns\":" << s.ssd_read_ns
            << ",\"ssd_write_ns\":" << s.ssd_write_ns << ",\"ssd_publish_ns\":" << s.ssd_publish_ns
            << ",\"uring_available\":" << (s.file_io.uring_available ? "true" : "false")
            << ",\"io_calibrated\":" << (s.file_io.calibrated ? "true" : "false")
            << ",\"io_mode\":" << int(s.file_io.selected)
            << ",\"io_queue_depth\":" << s.file_io.queue_depth
            << ",\"io_calibration_ns\":" << s.file_io.calibration_ns
            << ",\"prefetch_submitted\":" << s.prefetch_submitted
            << ",\"prefetch_deduplicated\":" << s.prefetch_deduplicated
            << ",\"prefetch_completed\":" << s.prefetch_completed
            << ",\"prefetch_skipped\":" << s.prefetch_skipped
            << ",\"prefetch_failed\":" << s.prefetch_failed
            << ",\"prefetch_rejected\":" << s.prefetch_rejected
            << ",\"prefetch_hits\":" << s.prefetch_hits
            << ",\"prefetch_unused\":" << s.prefetch_unused
            << ",\"prefetch_inflight\":" << s.prefetch_inflight
            << ",\"prefetch_peak\":" << s.prefetch_peak
            << ",\"prefetch_handles\":" << store.operations.size()
            << ",\"prefetch_handle_rejections\":" << store.rejected
            << ",\"prefetch_wait_ns\":" << store.wait_ns
            << ",\"prefetch_queue_ns\":" << s.prefetch_queue_ns
            << ",\"prefetch_ns\":" << s.prefetch_ns
            << ",\"ssd_payload_written_bytes\":" << s.ssd_payload_written_bytes
            << ",\"ssd_payload_read_bytes\":" << s.ssd_payload_read_bytes
            << ",\"ssd_verification_read_bytes\":" << s.ssd_verification_read_bytes;
        append_file_io_stats(out, s.file_io);
        append_device_stats(out, s.ssd_device);
        out << '}';
        serialized = out.str();
    });
    return status == 0 ? serialized.c_str() : nullptr;
}
// 0 accepted, 1 capacity fallback, -1 error. A completed handle still counts
// against the bound until polled/drained. Callers serialize destroy with calls.
int mi_tiered_prefetch(void* ptr, std::size_t index, std::uint64_t* operation) {
    int result = 0;
    const int status = checked([&] {
        if (!ptr || !operation)
            throw std::invalid_argument("Null prefetch argument");
        *operation = 0;
        auto& store = *static_cast<TieredStore*>(ptr);
        std::lock_guard<std::mutex> lock(store.operations_mutex);
        if (store.operations.size() >= 8) {
            ++store.rejected;
            result = 1;
            return;
        }
        if (store.next_operation == UINT64_MAX)
            throw std::overflow_error("Operation ID exhausted");
        const auto id = store.next_operation++;
        // Reserve the map node before accepting asynchronous work.
        auto position = store.operations.emplace(id, std::shared_future<bool>{}).first;
        try {
            position->second = store.cache.prefetch_if_space_async(index);
        } catch (const infer::StorageBusy&) {
            store.operations.erase(position);
            ++store.rejected;
            result = 1;
            return;
        } catch (...) {
            store.operations.erase(position);
            throw;
        }
        *operation = id;
    });
    return status ? status : result;
}
// 0 completed and released, 1 pending, -1 error (failed futures are released).
int mi_tiered_prefetch_poll(void* ptr, std::uint64_t operation, int wait, int* resident) {
    int result = 0;
    const int status = checked([&] {
        if (!ptr || !resident)
            throw std::invalid_argument("Null prefetch poll argument");
        auto& store = *static_cast<TieredStore*>(ptr);
        std::lock_guard<std::mutex> lock(store.operations_mutex);
        auto position = store.operations.find(operation);
        if (position == store.operations.end())
            throw std::invalid_argument("Unknown prefetch operation");
        if (!wait &&
            position->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            result = 1;
            return;
        }
        const auto begin = std::chrono::steady_clock::now();
        auto completion = position->second;
        store.operations.erase(position);
        try {
            *resident = completion.get();
        } catch (...) {
            store.wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count();
            throw;
        }
        store.wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    });
    return status ? status : result;
}
int mi_tiered_prefetch_drain(void* ptr) {
    return checked([&] {
        if (!ptr)
            throw std::invalid_argument("Null tiered cache");
        auto& store = *static_cast<TieredStore*>(ptr);
        std::lock_guard<std::mutex> lock(store.operations_mutex);
        const auto begin = std::chrono::steady_clock::now();
        std::exception_ptr failure;
        for (auto& item : store.operations) {
            try {
                item.second.get();
            } catch (...) {
                if (!failure)
                    failure = std::current_exception();
            }
        }
        store.operations.clear();
        store.wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
        if (failure)
            std::rethrow_exception(failure);
    });
}
void mi_tiered_destroy(void* ptr) {
    if (!ptr)
        return;
    mi_tiered_prefetch_drain(ptr);
    delete static_cast<TieredStore*>(ptr);
}
}
