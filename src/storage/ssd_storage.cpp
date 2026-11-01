#include "crc32c/crc32c.h"
#include "infer/storage/storage_backend.h"
#include "slot_table.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

namespace infer {
namespace {
struct IOTimer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~IOTimer() {
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
    }
};
using Header = std::array<unsigned char, SSDStorageBackend::record_header_bytes>;
constexpr unsigned char magic[8] = {'M', 'I', 'N', 'I', 'K', 'V', '0', '4'};
// Google CRC32C 按运行时 CPU 能力选择实现；其接口使用最终取反后的 CRC，
// 这里转换为原有的增量状态，保持文件格式和分段写入结果完全一致。
std::uint32_t extend_crc(std::uint32_t crc, const void* data, std::size_t bytes) {
    return ~::crc32c::Extend(~crc, static_cast<const std::uint8_t*>(data), bytes);
}
void put(Header& header, std::size_t offset, std::uint64_t value, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i)
        header[offset + i] = static_cast<unsigned char>(value >> (8 * i));
}
std::uint64_t get(const Header& header, std::size_t offset, std::size_t bytes) {
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        result |= std::uint64_t(header[offset + i]) << (8 * i);
    return result;
}
Header record_header(const BlockMeta& meta, std::size_t capacity, std::uint32_t crc) {
    Header header{};
    std::memcpy(header.data(), magic, 8);
    put(header, 8, 1, 4);
    put(header, 12, SSDStorageBackend::record_header_bytes, 4);
    put(header, 16, meta.id, 8);
    put(header, 24, meta.version, 8);
    put(header, 32, meta.size, 8);
    put(header, 40, capacity, 8);
    put(header, 48, crc, 4);
    put(header, 52, storage_detail::crc32c(header.data(), header.size()), 4);
    return header;
}
[[noreturn]] void io_error(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}
void read_all(int fd, void* data, std::size_t bytes, off_t offset) {
    auto* p = static_cast<unsigned char*>(data);
    while (bytes) {
        const auto n = ::pread(fd, p, std::min(bytes, std::size_t(SSIZE_MAX)), offset);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            io_error("SSD pread");
        }
        if (!n)
            throw std::runtime_error("SSD unexpected EOF");
        p += n;
        bytes -= std::size_t(n);
        offset += n;
    }
}
void write_all(int fd, const void* data, std::size_t bytes, off_t offset) {
    auto* p = static_cast<const unsigned char*>(data);
    while (bytes) {
        const auto n = ::pwrite(fd, p, std::min(bytes, std::size_t(SSIZE_MAX)), offset);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            io_error("SSD pwrite");
        }
        if (!n)
            throw std::runtime_error("SSD pwrite made no progress");
        p += n;
        bytes -= std::size_t(n);
        offset += n;
    }
}
// 大块页缓存读取与 CRC 都消耗 CPU/内存带宽。固定三条后台线程加 caller
// 分四片处理；没有无界任务队列，也不为每批创建线程或额外 payload 缓冲。
class BatchIO {
  public:
    BatchIO() {
        try {
            for (std::size_t i = 0; i < workers_.size(); ++i)
                workers_[i] = std::thread([this, i] { work(i + 1); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~BatchIO() { stop(); }
    void run(const std::function<void(std::size_t)>& task) {
        // 每个方向最多一个并行批次；另一方向和独立单条读取仍可继续。
        std::lock_guard<std::mutex> batch(batch_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            errors_.fill(nullptr);
            task_ = &task;
            remaining_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        invoke(task, 0);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [this] { return remaining_ == 0; });
            task_ = nullptr;
        }
        // 所有已接受的分片（包括失败分片）均已退出后才能释放 caller 的锁/缓冲。
        for (const auto& error : errors_)
            if (error)
                std::rethrow_exception(error);
    }

  private:
    void invoke(const std::function<void(std::size_t)>& task, std::size_t index) noexcept {
        try {
            task(index);
        } catch (...) {
            errors_[index] = std::current_exception();
        }
    }
    void work(std::size_t index) {
        std::size_t observed = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stopped_ || generation_ != observed; });
            if (stopped_)
                return;
            observed = generation_;
            const auto* task = task_;
            lock.unlock();
            invoke(*task, index);
            lock.lock();
            if (--remaining_ == 0)
                done_.notify_one();
        }
    }
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_)
            if (worker.joinable())
                worker.join();
    }
    std::mutex batch_mutex_, mutex_;
    std::condition_variable ready_, done_;
    std::array<std::thread, 3> workers_;
    std::array<std::exception_ptr, 4> errors_;
    const std::function<void(std::size_t)>* task_ = nullptr;
    std::size_t generation_ = 0, remaining_ = 0;
    bool stopped_ = false;
};
} // namespace
std::uint32_t storage_detail::crc32c(const void* data, std::size_t bytes) {
    return ~extend_crc(~0u, data, bytes);
}

struct SSDStorageBackend::Impl {
    struct Allocation {
        mutable std::mutex mutex;
        std::size_t slot = 0, written = 0;
        std::uint32_t crc = ~0u;
        bool sealed = false;
        bool released = false;
        BlockMeta meta;
    };
    int fd = -1;
    std::array<int, 4> files{{-1, -1, -1, -1}};
    std::size_t file_count = 1;
    std::string pending_file;
    std::size_t payload = 0, record = 0, max_slots = 0;
    std::size_t read_piece_bytes = 0;
    storage_detail::SlotTable<std::shared_ptr<Allocation>> allocations;
    mutable std::mutex mutex;
    mutable std::atomic<std::uint64_t> reserve_ns{0}, read_ns{0}, write_ns{0}, publish_ns{0};
    mutable std::atomic<std::uint64_t> written_bytes{0}, read_bytes{0}, verification_bytes{0};
    std::atomic<std::uint64_t> record_writes{0}, source_crc_bytes{0};
    SSDVerification verification = SSDVerification::READ_BACK;
    std::unique_ptr<FileIO> io;
    std::unique_ptr<SSDDevice> device;
    bool parallel_posix = false;
    struct ParallelIO {
        std::once_flag once;
        std::unique_ptr<BatchIO> batch;
        std::array<std::unique_ptr<FileIO>, 4> lanes;
    };
    // A demand read must not wait for an unrelated write batch or its rings.
    std::array<ParallelIO, 2> parallel;
    FileIOConfig io_config;
    mutable std::mutex parallel_mutex;
    ~Impl() {
        for (auto& direction : parallel) {
            direction.batch.reset();
            for (auto& lane : direction.lanes)
                lane.reset();
        }
        io.reset(); // Drain async calls and release registered files before closing fd.
        if (fd >= 0) {
            for (auto file : files)
                if (file >= 0)
                    ::close(file);
            if (!pending_file.empty() && ::unlink(pending_file.c_str()) < 0)
                std::fprintf(stderr, "SSD temporary file cleanup failed: %s\n",
                             std::strerror(errno));
        }
    }
    void prepare_parallel(bool write) {
        auto& direction = parallel[write];
        std::call_once(direction.once, [&] {
            direction.batch = std::make_unique<BatchIO>();
            if (!parallel_posix) {
                std::array<std::unique_ptr<FileIO>, 4> lanes;
                auto config = io_config;
                config.queue_depth = std::max(std::size_t(1), config.queue_depth / 4);
                for (std::size_t i = 0; i < lanes.size(); ++i)
                    lanes[i] = std::make_unique<FileIO>(config, files[i % file_count]);
                std::lock_guard<std::mutex> lock(parallel_mutex);
                direction.lanes = std::move(lanes);
            }
        });
    }
    int file_at(std::uint64_t offset) const { return files[(offset / record) % file_count]; }
    std::uint64_t file_offset(std::uint64_t offset) const {
        return (offset / record / file_count) * record + offset % record;
    }
    void read_range(void* host, std::size_t bytes, off_t offset) const {
        if (device)
            device->read(host, bytes, offset);
        else if (!parallel_posix)
            io->read_many(file_at(offset), {{host, bytes, file_offset(offset)}});
        else
            read_all(file_at(offset), host, bytes, file_offset(offset));
    }
    void write_range(const void* host, std::size_t bytes, off_t offset) {
        if (device)
            device->write(host, bytes, offset);
        else if (!parallel_posix) {
            prepare_parallel(true);
            const auto lane = (std::size_t(offset) / record) % parallel[true].lanes.size();
            parallel[true].lanes[lane]->write_many(file_at(offset),
                                         {{const_cast<void*>(host), bytes, file_offset(offset)}});
        } else
            write_all(file_at(offset), host, bytes, file_offset(offset));
    }
    std::shared_ptr<Allocation> lookup(StorageHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex);
        return allocations.at(handle);
    }
    static void check_live(const Allocation& allocation) {
        if (allocation.released)
            throw std::out_of_range("Released SSD handle");
    }
    off_t offset(const Allocation& allocation) const {
        // 构造时已限制 file budget <= off_t::max，slot < max_slots。
        return static_cast<off_t>(allocation.slot * record);
    }
    // 在同一遍读取中计算 CRC 并提取请求区间，避免先校验后另读引入内容差异。
    std::uint32_t payload_crc(const Allocation& allocation, std::size_t size, void* host = nullptr,
                              std::size_t begin = 0, std::size_t bytes = 0) const {
        // 完整块直接读入最终 CPU/Pinned 地址；部分读取仍校验整条记录。
        if (host && begin == 0 && bytes == size) {
            std::uint32_t crc = ~0u;
            for (std::size_t done = 0; done < size;) {
                const auto n = std::min(read_piece_bytes ? read_piece_bytes : size, size - done);
                auto* destination = static_cast<unsigned char*>(host) + done;
                read_range(destination, n, offset(allocation) + record_header_bytes + done);
                read_bytes.fetch_add(n, std::memory_order_relaxed);
                crc = extend_crc(crc, destination, n);
                done += n;
            }
            return ~crc;
        }
        std::array<unsigned char, 64 * 1024> buffer{};
        std::uint32_t crc = ~0u;
        for (std::size_t done = 0; done < size;) {
            const auto n = std::min(buffer.size(), size - done);
            read_range(buffer.data(), n, offset(allocation) + record_header_bytes + done);
            (host ? read_bytes : verification_bytes).fetch_add(n, std::memory_order_relaxed);
            crc = extend_crc(crc, buffer.data(), n);
            const auto first = std::max(done, begin), last = std::min(done + n, begin + bytes);
            if (first < last)
                std::memcpy(static_cast<unsigned char*>(host) + first - begin,
                            buffer.data() + first - done, last - first);
            done += n;
        }
        return ~crc;
    }
    void check_header(const Allocation& allocation, const BlockMeta& expected,
                      Header header) const {
        if (!expected.id || !expected.version || !expected.size || expected.size > payload)
            throw std::system_error(EBADMSG, std::generic_category(), "Invalid SSD record metadata");
        const auto header_crc = get(header, 52, 4);
        put(header, 52, 0, 4);
        if (std::memcmp(header.data(), magic, 8) || get(header, 8, 4) != 1 ||
            get(header, 12, 4) != record_header_bytes || get(header, 16, 8) != expected.id ||
            get(header, 24, 8) != expected.version || get(header, 32, 8) != expected.size ||
            get(header, 40, 8) != payload || get(header, 56, 8) != 0 ||
            get(header, 48, 4) != std::uint32_t(~allocation.crc) ||
            header_crc != storage_detail::crc32c(header.data(), header.size())) {
            throw std::system_error(EBADMSG, std::generic_category(), "SSD record header/identity mismatch");
        }
    }
    void checked_read(const Allocation& allocation, const BlockMeta& expected, void* host = nullptr,
                      std::size_t begin = 0, std::size_t bytes = 0) const {
        Header header{};
        read_range(header.data(), header.size(), offset(allocation));
        check_header(allocation, expected, header);
        if (payload_crc(allocation, expected.size, host, begin, bytes) != get(header, 48, 4)) {
            throw std::system_error(EBADMSG, std::generic_category(), "SSD payload CRC32C mismatch");
        }
    }
};

std::size_t SSDStorageBackend::allocation_size(std::size_t bytes) {
    storage_detail::validate_slot_bytes(bytes);
    if (bytes > std::size_t(std::numeric_limits<off_t>::max()) - record_header_bytes) {
        throw std::overflow_error("SSD record offset overflow");
    }
    return bytes + record_header_bytes;
}
std::size_t SSDStorageBackend::allocation_size(std::size_t bytes, std::size_t alignment) {
    const auto record = allocation_size(bytes);
    if (!alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("SSD alignment must be a power of two");
    if (record > std::size_t(std::numeric_limits<off_t>::max()) - (alignment - 1))
        throw std::overflow_error("Aligned SSD record overflow");
    return (record + alignment - 1) / alignment * alignment;
}
SSDStorageBackend::SSDStorageBackend(std::size_t bytes, std::size_t budget,
                                     const std::string& directory, FileIOConfig io,
                                     SSDVerification verification) {
    const auto record = allocation_size(bytes);
    if (budget > std::size_t(std::numeric_limits<off_t>::max()))
        throw std::overflow_error("SSD file offset overflow");
    if (budget < record)
        throw std::invalid_argument(
            "SSD budget must include a full record header and payload slot");
    auto state = std::make_unique<Impl>();
    if (verification != SSDVerification::READ_BACK && verification != SSDVerification::ON_READ)
        throw std::invalid_argument("Invalid SSD verification policy");
    state->verification = verification;
    state->parallel_posix = io.mode == FileIOMode::POSIX;
    state->io_config = io;
    state->payload = bytes;
    state->record = record;
    state->max_slots = budget / record;
    if (directory.empty() || directory.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid SSD directory");
    if (io.file_shards != 1 && io.file_shards != 4)
        throw std::invalid_argument("SSD file shards must be 1 or 4");
    state->file_count = io.file_shards;
    for (std::size_t i = 0; i < state->file_count; ++i) {
        state->pending_file = directory + "/miniinfer-storage-" + std::to_string(i) + "-XXXXXX";
        auto& name = state->pending_file;
        const auto file = ::mkostemp(name.data(), O_CLOEXEC);
        if (file < 0)
            io_error("SSD private temporary file");
        state->files[i] = file;
        state->fd = state->files[0];
        if (::fchmod(file, 0600) != 0)
            io_error("SSD private file mode");
        const auto slots =
            state->max_slots / state->file_count + (i < state->max_slots % state->file_count);
        int resize;
        do {
            resize = ::ftruncate(file, static_cast<off_t>(slots * record));
        } while (resize < 0 && errno == EINTR);
        if (resize < 0)
            io_error("SSD initialize sparse address space");
        if (::unlink(name.c_str()) != 0)
            io_error("SSD unlink temporary file");
        state->pending_file.clear();
    }
    state->io = std::make_unique<FileIO>(io, state->fd);
    impl_ = std::move(state);
}
SSDStorageBackend::SSDStorageBackend(std::size_t bytes, std::size_t budget,
                                     std::unique_ptr<SSDDevice> device,
                                     SSDVerification verification, std::size_t read_piece_bytes) {
    if (!device)
        throw std::invalid_argument("SSD requires an owning device");
    const auto record = allocation_size(bytes, device->alignment());
    if (verification != SSDVerification::READ_BACK && verification != SSDVerification::ON_READ)
        throw std::invalid_argument("Invalid SSD verification policy");
    if (budget < record || budget > std::size_t(std::numeric_limits<off_t>::max()))
        throw std::invalid_argument("Invalid SSD device budget");
    const auto slots = budget / record;
    if (slots > device->capacity_bytes() / record)
        throw std::invalid_argument("SSD device region is smaller than budget");
    auto state = std::make_unique<Impl>();
    state->read_piece_bytes = read_piece_bytes;
    state->payload = bytes;
    state->record = record;
    state->max_slots = slots;
    state->verification = verification;
    state->device = std::move(device);
    impl_ = std::move(state);
}
SSDStorageBackend::~SSDStorageBackend() = default;
std::size_t SSDStorageBackend::slot_bytes() const noexcept {
    return impl_->payload;
}
std::size_t SSDStorageBackend::allocation_bytes() const noexcept {
    return impl_->record;
}
StorageHandle SSDStorageBackend::reserve() {
    IOTimer timer{impl_->reserve_ns};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->allocations.create([&](std::size_t slot) {
        auto allocation = std::make_shared<Impl::Allocation>();
        allocation->slot = slot;
        return allocation;
    }, impl_->max_slots);
}
void SSDStorageBackend::release(StorageHandle handle) {
    const auto allocation = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(allocation->mutex);
    Impl::check_live(*allocation);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->allocations.erase(handle);
    allocation->released = true;
}
StorageMemoryView SSDStorageBackend::memory_view(StorageHandle handle) const {
    const auto allocation = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(allocation->mutex);
    Impl::check_live(*allocation);
    return {nullptr, impl_->payload, DeviceType::UNKNOWN, -1};
}
void SSDStorageBackend::write(StorageHandle handle, std::size_t offset, const void* host,
                              std::size_t bytes) {
    IOTimer timer{impl_->write_ns};
    // 全局锁仅查表；记录锁保护这一槽的生命周期和完整 I/O，其他槽可并行。
    const auto entry = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(entry->mutex);
    Impl::check_live(*entry);
    auto& allocation = *entry;
    storage_detail::validate_io(impl_->payload, offset, host, bytes);
    if (!bytes)
        return;
    if (offset != 0 && offset != allocation.written)
        throw std::invalid_argument("SSD writes must start at zero or append");
    allocation.sealed = false;
    if (!offset) {
        allocation.written = 0;
        allocation.crc = ~0u;
    }
    try {
        impl_->write_range(host, bytes, impl_->offset(allocation) + record_header_bytes + offset);
    } catch (...) {
        allocation.written = 0;
        allocation.crc = ~0u;
        throw;
    }
    allocation.crc = extend_crc(allocation.crc, host, bytes);
    impl_->source_crc_bytes.fetch_add(bytes, std::memory_order_relaxed);
    allocation.written += bytes;
    impl_->written_bytes.fetch_add(bytes, std::memory_order_relaxed);
}
void SSDStorageBackend::publish(StorageHandle handle, const BlockMeta& meta) {
    IOTimer timer{impl_->publish_ns};
    // 全局锁仅查表；记录锁保护这一槽的生命周期和完整 I/O，其他槽可并行。
    const auto entry = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(entry->mutex);
    Impl::check_live(*entry);
    auto& allocation = *entry;
    if (!meta.id || !meta.version || !meta.size || meta.size > impl_->payload ||
        meta.size != allocation.written) {
        throw std::invalid_argument("SSD publication requires complete content and identity");
    }
    allocation.sealed = false;
    const auto header = record_header(meta, impl_->payload, ~allocation.crc);
    impl_->write_range(header.data(), header.size(), impl_->offset(allocation));
    // 默认回读核对记录头、输入 CRC 与文件内容；ON_READ 将校验延后到消费时。
    if (impl_->verification == SSDVerification::READ_BACK)
        impl_->checked_read(allocation, meta);
    allocation.meta = meta;
    allocation.sealed = true;
    impl_->record_writes.fetch_add(1, std::memory_order_relaxed);
}
void SSDStorageBackend::write_published(StorageHandle handle, const BlockMeta& meta,
                                        const void* host) {
    write_published_many({{handle, meta, host}});
}
void SSDStorageBackend::write_published_many(const std::vector<StorageWrite>& requests) {
    if (requests.empty())
        return;
    if (requests.size() > 64)
        throw std::invalid_argument("SSD write batch exceeds 64 blocks");
    IOTimer timer{impl_->write_ns};
    std::vector<std::size_t> order(requests.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](auto a, auto b) { return requests[a].handle < requests[b].handle; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (requests[order[i - 1]].handle == requests[order[i]].handle)
            throw std::invalid_argument("Duplicate SSD write handle");
    std::vector<std::shared_ptr<Impl::Allocation>> allocations(requests.size());
    std::vector<std::unique_lock<std::mutex>> locks;
    std::vector<Header> headers(requests.size());
    locks.reserve(requests.size());
    for (auto i : order) {
        allocations[i] = impl_->lookup(requests[i].handle);
        locks.emplace_back(allocations[i]->mutex);
        Impl::check_live(*allocations[i]);
        const auto& r = requests[i];
        storage_detail::validate_io(impl_->payload, 0, r.host, r.meta.size);
        if (!r.meta.id || !r.meta.version || !r.meta.size)
            throw std::invalid_argument("SSD publication requires complete content and identity");
    }
    // Validate every request before changing old records; locks outlive all shards.
    for (auto& a : allocations) {
        a->sealed = false;
        a->written = 0;
    }
    const auto shard = [&](std::size_t first, std::size_t end, FileIO* io, std::size_t lane) {
        std::vector<FileIORequest> ranges;
        ranges.reserve(end - first);
        std::vector<std::size_t> selected;
        for (auto i = first; i < end; ++i) {
            if (impl_->file_count > 1 && allocations[i]->slot % impl_->file_count != lane)
                continue;
            selected.push_back(i);
            const auto& r = requests[i];
            const auto crc = storage_detail::crc32c(r.host, r.meta.size);
            impl_->source_crc_bytes.fetch_add(r.meta.size, std::memory_order_relaxed);
            allocations[i]->crc = ~crc;
            headers[i] = record_header(r.meta, impl_->payload, crc);
            ranges.push_back({const_cast<void*>(r.host), r.meta.size,
                              impl_->file_offset(impl_->offset(*allocations[i])), headers[i].data(),
                              headers[i].size()});
        }
        // One writev/SQE per record, without concatenating payloads or a second header write.
        if (impl_->device)
            impl_->device->write_many(ranges);
        else
            io->write_many(impl_->files[lane % impl_->file_count], ranges);
        for (auto i : selected) {
            auto& a = *allocations[i];
            const auto& r = requests[i];
            a.written = r.meta.size;
            impl_->written_bytes.fetch_add(r.meta.size, std::memory_order_relaxed);
            impl_->record_writes.fetch_add(1, std::memory_order_relaxed);
            if (impl_->verification == SSDVerification::READ_BACK) {
                IOTimer verifying{impl_->publish_ns};
                impl_->checked_read(a, r.meta);
            }
            a.meta = r.meta;
            a.sealed = true;
        }
    };
    if (impl_->device) {
        shard(0, requests.size(), nullptr, 0);
    } else if (requests.size() > 1) {
        impl_->prepare_parallel(true);
        const auto chunk = (requests.size() + 3) / 4;
        impl_->parallel[true].batch->run([&](std::size_t lane) {
            const auto first = impl_->file_count > 1 ? 0 : std::min(requests.size(), lane * chunk);
            const auto end = impl_->file_count > 1 ? requests.size()
                                                   : std::min(requests.size(), (lane + 1) * chunk);
            if (impl_->parallel_posix) {
                FileIO direct;
                shard(first, end, &direct, lane);
            } else {
                shard(first, end, impl_->parallel[true].lanes[lane].get(), lane);
            }
        });
    } else if (impl_->parallel_posix) {
        FileIO direct;
        shard(0, requests.size(), &direct, allocations[0]->slot % impl_->file_count);
    } else {
        shard(0, requests.size(), impl_->io.get(), allocations[0]->slot % impl_->file_count);
    }
}
void SSDStorageBackend::validate(StorageHandle handle, const BlockMeta& meta) const {
    // 全局锁仅查表；记录锁保护这一槽的生命周期和完整 I/O，其他槽可并行。
    const auto entry = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(entry->mutex);
    Impl::check_live(*entry);
    const auto& allocation = *entry;
    if (!allocation.sealed)
        throw std::logic_error("SSD record is not published");
    impl_->checked_read(allocation, meta);
}
void SSDStorageBackend::read_validated(StorageHandle handle, const BlockMeta& expected,
                                       void* host) {
    IOTimer timer{impl_->read_ns};
    // 全局锁仅查表；记录锁保护这一槽的生命周期和完整 I/O，其他槽可并行。
    const auto entry = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(entry->mutex);
    Impl::check_live(*entry);
    const auto& allocation = *entry;
    if (!allocation.sealed)
        throw std::logic_error("SSD record is not published");
    storage_detail::validate_io(impl_->payload, 0, host, expected.size);
    impl_->checked_read(allocation, expected, host, 0, expected.size);
}
void SSDStorageBackend::read_validated_many(const std::vector<StorageRead>& requests) {
    if (requests.empty())
        return;
    if (requests.size() > 64)
        throw std::invalid_argument("SSD read batch exceeds 64 blocks");
    IOTimer timer{impl_->read_ns};
    std::vector<std::size_t> order(requests.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](auto a, auto b) { return requests[a].handle < requests[b].handle; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (requests[order[i - 1]].handle == requests[order[i]].handle)
            throw std::invalid_argument("Duplicate SSD read handle");
    std::vector<std::shared_ptr<Impl::Allocation>> allocations(requests.size());
    std::vector<std::unique_lock<std::mutex>> locks;
    std::vector<Header> headers(requests.size());
    std::vector<FileIORequest> ranges;
    locks.reserve(requests.size());
    ranges.reserve(2 * requests.size());
    // 固定句柄顺序拿记录锁，避免两个并发批次反序访问时死锁。
    for (auto i : order) {
        allocations[i] = impl_->lookup(requests[i].handle);
        locks.emplace_back(allocations[i]->mutex);
        const auto& allocation = *allocations[i];
        Impl::check_live(allocation);
        if (!allocation.sealed)
            throw std::logic_error("SSD batch record is not published");
        storage_detail::validate_io(impl_->payload, 0, requests[i].host, requests[i].expected.size);
        ranges.push_back(
            {headers[i].data(), headers[i].size(), std::uint64_t(impl_->offset(allocation))});
        ranges.push_back({requests[i].host, requests[i].expected.size,
                          std::uint64_t(impl_->offset(allocation)) + record_header_bytes});
    }
    std::size_t parallel_bytes = 0;
    for (const auto& request : requests)
        parallel_bytes += std::min(request.expected.size, std::size_t(1024 * 1024));
    if (impl_->io && requests.size() > 1 &&
        (parallel_bytes >= 1024 * 1024 || impl_->file_count > 1)) {
        // FileIO 的原串行路径会拒绝重叠/溢出的目标；并发路径提交前同样核对，
        // 避免两个分片同时写一个 host 范围。
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const auto begin = reinterpret_cast<std::uintptr_t>(ranges[i].data);
            if (!ranges[i].bytes || ranges[i].bytes > UINTPTR_MAX - begin)
                throw std::invalid_argument("Invalid SSD batch destination");
            for (std::size_t j = 0; j < i; ++j) {
                const auto other = reinterpret_cast<std::uintptr_t>(ranges[j].data);
                if (begin < other + ranges[j].bytes && other < begin + ranges[i].bytes)
                    throw std::invalid_argument("Overlapping SSD batch destinations");
            }
        }
        impl_->prepare_parallel(false);
        const auto chunk = (requests.size() + 3) / 4;
        impl_->parallel[false].batch->run([&](std::size_t task) {
            const auto begin = std::min(requests.size(), task * chunk);
            const auto end = std::min(requests.size(), (task + 1) * chunk);
            if (!impl_->parallel_posix) {
                std::array<std::size_t, 64> selected{};
                std::size_t count = 0;
                for (auto j = impl_->file_count > 1 ? 0 : begin;
                     j < (impl_->file_count > 1 ? requests.size() : end); ++j) {
                    const auto i = order[j];
                    if (impl_->file_count > 1 && allocations[i]->slot % impl_->file_count != task)
                        continue;
                    selected[count++] = i;
                }
                std::vector<FileIORequest> part;
                part.reserve(2 * count);
                // Bound the CRC working set while retaining enough in-flight
                // reads for page-cache misses. No payload staging is allocated.
                constexpr std::size_t window_bytes = 1024 * 1024;
                for (std::size_t first = 0; first < count;) {
                    part.clear();
                    auto last = first;
                    std::size_t bytes = 0;
                    do {
                        const auto i = selected[last++];
                        const auto offset = impl_->file_offset(impl_->offset(*allocations[i]));
                        part.push_back({headers[i].data(), headers[i].size(), offset});
                        part.push_back({requests[i].host, requests[i].expected.size,
                                        offset + record_header_bytes});
                        bytes += std::min(requests[i].expected.size, window_bytes);
                    } while (last < count && bytes < window_bytes);
                    impl_->parallel[false].lanes[task]->read_many(
                        impl_->files[task % impl_->file_count], part);
                    for (auto j = first; j < last; ++j) {
                        const auto i = selected[j];
                        impl_->check_header(*allocations[i], requests[i].expected, headers[i]);
                        impl_->read_bytes.fetch_add(requests[i].expected.size,
                                                    std::memory_order_relaxed);
                        if (storage_detail::crc32c(requests[i].host, requests[i].expected.size) !=
                            get(headers[i], 48, 4))
                            throw std::system_error(EBADMSG, std::generic_category(), "SSD batch payload CRC32C mismatch");
                    }
                    first = last;
                }
                return;
            }
            for (auto i = task * chunk; i < end; ++i) {
                const auto& request = requests[i];
                impl_->checked_read(*allocations[i], request.expected, request.host, 0,
                                    request.expected.size);
            }
        });
        return;
    }
    if (impl_->device)
        impl_->device->read_many(ranges);
    else if (impl_->file_count > 1) {
        for (std::size_t i = 0; i < requests.size(); ++i)
            impl_->checked_read(*allocations[i], requests[i].expected, requests[i].host, 0,
                                requests[i].expected.size);
        return;
    } else
        impl_->io->read_many(impl_->fd, ranges);
    for (const auto& request : requests)
        impl_->read_bytes.fetch_add(request.expected.size, std::memory_order_relaxed);
    const auto verify = [&](std::size_t i) {
        impl_->check_header(*allocations[i], requests[i].expected, headers[i]);
        if (storage_detail::crc32c(requests[i].host, requests[i].expected.size) !=
            get(headers[i], 48, 4))
            throw std::system_error(EBADMSG, std::generic_category(), "SSD batch payload CRC32C mismatch");
    };
    for (std::size_t i = 0; i < requests.size(); ++i)
        verify(i);
}
void SSDStorageBackend::read(StorageHandle handle, std::size_t offset, void* host,
                             std::size_t bytes) {
    // 全局锁仅查表；记录锁保护这一槽的生命周期和完整 I/O，其他槽可并行。
    const auto entry = impl_->lookup(handle);
    std::lock_guard<std::mutex> use(entry->mutex);
    Impl::check_live(*entry);
    const auto& allocation = *entry;
    if (!allocation.sealed)
        throw std::logic_error("SSD record is not published");
    storage_detail::validate_io(allocation.meta.size, offset, host, bytes);
    impl_->checked_read(allocation, allocation.meta, host, offset, bytes);
}
void SSDStorageBackend::validate_stream(cudaStream_t stream) const {
    if (stream)
        throw std::invalid_argument("SSD does not accept a CUDA consumer stream");
}
void SSDStorageBackend::wait(cudaStream_t stream) const {
    validate_stream(stream);
}
BackendPoolStats SSDStorageBackend::pool_stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto active = impl_->allocations.size() * impl_->record,
               reserved = impl_->allocations.capacity() * impl_->record;
    // 已分配的逻辑槽和可复用槽；预设稀疏地址范围不代表这些槽已被分配。
    return {active, reserved - active, 0, reserved, true};
}
BackendIOStats SSDStorageBackend::io_stats() const {
    auto file = impl_->io ? impl_->io->stats() : FileIOStats{};
    {
        std::lock_guard<std::mutex> lock(impl_->parallel_mutex);
        for (const auto& direction : impl_->parallel) {
            for (const auto& lane : direction.lanes) {
                if (!lane)
                    continue;
                const auto s = lane->stats();
                file.read_operations += s.read_operations;
                file.write_operations += s.write_operations;
                file.read_bytes += s.read_bytes;
                file.write_bytes += s.write_bytes;
                file.submissions += s.submissions;
                file.completions += s.completions;
                file.completion_waits += s.completion_waits;
                file.peak_inflight = std::max(file.peak_inflight, s.peak_inflight);
                file.rings += s.rings;
            }
        }
    }
    return {impl_->reserve_ns.load(),
            impl_->read_ns.load(),
            impl_->write_ns.load(),
            impl_->publish_ns.load(),
            file,
            impl_->written_bytes.load(),
            impl_->read_bytes.load(),
            impl_->verification_bytes.load(),
            impl_->device ? impl_->device->stats() : SSDDeviceStats{},
            impl_->record_writes.load(),
            impl_->source_crc_bytes.load()};
}
} // namespace infer
