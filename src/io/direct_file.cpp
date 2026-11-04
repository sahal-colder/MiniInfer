#include "infer/io/direct_file.h"
#include "infer/io/safetensors.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#ifdef INFER_HAS_IO_URING
#include <liburing.h>
#endif

namespace infer::io {
namespace {
constexpr std::size_t huge_bytes = 2 * 1024 * 1024;
constexpr std::size_t max_depth = 256;
// Stable Linux UAPI values; allow building with pre-6.1 liburing headers.
constexpr unsigned coop_taskrun = 1U << 8, single_issuer = 1U << 12, defer_taskrun = 1U << 13;
void require(bool ok, int error, const char* operation) {
    if (!ok) throw std::system_error(error, std::generic_category(), operation);
}
void raise_memlock() {
    const rlimit unlimited{RLIM_INFINITY, RLIM_INFINITY};
    if (setrlimit(RLIMIT_MEMLOCK, &unlimited) == 0) return;
    rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0 || limit.rlim_cur == limit.rlim_max) return;
    limit.rlim_cur = limit.rlim_max;
    // The hard ceiling requires CAP_SYS_RESOURCE. Ordinary aligned I/O still
    // uses the caller's address when the process cannot increase it.
    setrlimit(RLIMIT_MEMLOCK, &limit);
}
struct Pool {
    void* data = MAP_FAILED;
    std::size_t bytes;
    bool hugetlb = false;
    explicit Pool(std::size_t size) : bytes((size + huge_bytes - 1) / huge_bytes * huge_bytes) {
        data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        hugetlb = data != MAP_FAILED;
        if (!hugetlb) {
            // Overmap and trim so the THP fallback also starts on a 2 MiB boundary.
            auto* mapping = static_cast<char*>(mmap(nullptr, bytes + huge_bytes,
                PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            require(mapping != MAP_FAILED, errno, "Map fixed I/O pool");
            const auto address = (reinterpret_cast<std::uintptr_t>(mapping) + huge_bytes - 1) & ~(huge_bytes - 1);
            data = reinterpret_cast<void*>(address);
            const auto prefix = static_cast<char*>(data) - mapping;
            if (prefix) munmap(mapping, prefix);
            munmap(static_cast<char*>(data) + bytes, huge_bytes - prefix);
            madvise(data, bytes, MADV_HUGEPAGE); // A hint, not a physical-contiguity guarantee.
        }
        std::memset(data, 0, bytes); // Fault before registration and timed I/O.
    }
    ~Pool() { if (data != MAP_FAILED) munmap(data, bytes); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
};
#ifdef INFER_HAS_IO_URING
struct Lane {
    Pool pool;
    io_uring ring{};
    bool live = false;
    unsigned flags = 0;
    bool pool_registered = false, external_registered = false, buffers_live = false;
    DirectFileBuffer window{};
    std::atomic<std::size_t> registration_failures{0}, window_registrations{0}, registered_bytes{0};
    DirectFileConfig config;
    std::atomic<std::uint64_t> submissions{0}, completions{0}, waits{0};
    std::atomic<std::uint64_t> fixed_ops{0}, ordinary_ops{0};
    std::atomic<std::uint64_t> read_ops{0}, write_ops{0}, read_bytes{0}, write_bytes{0};
    std::atomic<std::uint64_t> copied_read{0}, copied_write{0}, direct_read{0}, direct_write{0};
    std::atomic<std::size_t> peak{0};

    Lane(int fd, DirectFileConfig value) : pool(value.bounce_bytes), config(std::move(value)) {
        std::vector<iovec> buffers{{pool.data, pool.bytes}};
        for (const auto& buffer : config.buffers) buffers.push_back({buffer.data, buffer.bytes});
        const auto modern = coop_taskrun | single_issuer | defer_taskrun;
        const std::array<unsigned, 6> candidates{
            IORING_SETUP_SQPOLL | IORING_SETUP_IOPOLL, IORING_SETUP_SQPOLL,
            modern | IORING_SETUP_IOPOLL, modern, coop_taskrun, 0};
        int error = EOPNOTSUPP;
        for (auto candidate : candidates) {
            if ((!config.sqpoll && (candidate & IORING_SETUP_SQPOLL)) ||
                (!config.iopoll && (candidate & IORING_SETUP_IOPOLL))) continue;
            io_uring_params params{};
            params.flags = candidate;
            params.sq_thread_idle = 100; // Do not keep an idle worker polling indefinitely.
            auto rc = io_uring_queue_init_params(config.queue_depth, &ring, &params);
            if (rc < 0) { error = -rc; continue; }
            rc = io_uring_register_files(&ring, &fd, 1);
            if (rc < 0) {
                io_uring_queue_exit(&ring);
                throw std::system_error(-rc, std::generic_category(), "Register fixed file");
            }
            flags = candidate;
            if (flags & IORING_SETUP_IOPOLL) {
                // Probe device polling on an actual written sector without
                // requiring a successful buffer registration first.
                auto* sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(sqe, 0, pool.data, page_bytes, 0);
                sqe->flags |= IOSQE_FIXED_FILE;
                complete(1, false);
                io_uring_cqe* cqe = nullptr;
                if (io_uring_peek_cqe(&ring, &cqe) != 0) std::terminate();
                const auto result = cqe->res;
                io_uring_cq_advance(&ring, 1);
                if (result != int(page_bytes)) {
                    io_uring_unregister_files(&ring);
                    io_uring_queue_exit(&ring);
                    error = result < 0 ? -result : EIO;
                    require(error == EOPNOTSUPP || error == EINVAL || error == ENOSYS,
                            error, "Probe filesystem IOPOLL");
                    continue;
                }
            }
            rc = io_uring_register_buffers(&ring, buffers.data(), buffers.size());
            if (rc < 0 && (rc == -ENOMEM || rc == -EPERM || rc == -EAGAIN)) {
                ++registration_failures;
                // Preserve the remaining quota for caller payload windows.
                // RMW scratch can use ordinary direct I/O without a fixed slot.
            } else if (rc == 0) external_registered = !config.buffers.empty();
            if (rc < 0 && rc != -ENOMEM && rc != -EPERM && rc != -EAGAIN) {
                io_uring_unregister_files(&ring);
                io_uring_queue_exit(&ring);
                throw std::system_error(-rc, std::generic_category(), "Register fixed I/O buffers");
            }
            buffers_live = pool_registered = rc == 0;
            std::size_t bytes = pool_registered ? pool.bytes : 0;
            if (external_registered)
                for (const auto& buffer : config.buffers) bytes += buffer.bytes;
            registered_bytes.store(bytes, std::memory_order_relaxed);
            live = true;
            return;
        }
        throw std::system_error(error, std::generic_category(), "Initialize kernel I/O ring");
    }
    ~Lane() {
        if (!live) return;
        if (buffers_live) io_uring_unregister_buffers(&ring);
        io_uring_unregister_files(&ring);
        io_uring_queue_exit(&ring);
    }
    // Submit the entire wave and wait for its CQEs together. A short submission
    // or interrupted enter is retried without dropping accepted buffer borrows.
    void complete(unsigned count, bool account) {
        const bool fast = count == 1 && (flags & IORING_SETUP_SQPOLL);
        auto rc = fast ? io_uring_submit(&ring) : io_uring_submit_and_wait(&ring, count);
        if (account) { ++submissions; if (!fast) ++waits; }
        while (rc == -EINTR || rc == -EAGAIN || rc == -EBUSY || rc == -ENOMEM ||
               (!(flags & IORING_SETUP_SQPOLL) && io_uring_sq_ready(&ring))) {
            if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY && rc != -ENOMEM)
                std::terminate(); // Unknown in-flight ownership cannot be returned to the caller.
            rc = io_uring_submit_and_wait(&ring, count);
            if (account) { ++submissions; ++waits; }
        }
        if (rc < 0) std::terminate();
        // SQPOLL can return after publishing the SQEs; waiting still needs enter
        // on an empty CQ, and IOPOLL always needs kernel completion polling.
        while (io_uring_cq_ready(&ring) < count) {
            io_uring_cqe* cqe = nullptr;
            rc = io_uring_wait_cqe_nr(&ring, &cqe, count);
            if (account) ++waits;
            if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY && rc != -ENOMEM)
                std::terminate();
        }
    }
    int registered(void* data, std::size_t bytes) const {
        const auto address = reinterpret_cast<std::uintptr_t>(data);
        if (window.bytes) {
            const auto begin = reinterpret_cast<std::uintptr_t>(window.data);
            if (address >= begin && address - begin <= window.bytes &&
                bytes <= window.bytes - (address - begin)) return 0;
        }
        if (!external_registered) return -1;
        for (std::size_t i = 0; i < config.buffers.size(); ++i) {
            const auto begin = reinterpret_cast<std::uintptr_t>(config.buffers[i].data);
            if (address >= begin && address - begin <= config.buffers[i].bytes &&
                bytes <= config.buffers[i].bytes - (address - begin)) return int(i + 1);
        }
        return -1;
    }
    void register_single(const FileIORequest& request) {
        if (request.bytes > 4 * 1024 * 1024 || registered(request.data, request.bytes) >= 0) return;
        const auto address = reinterpret_cast<std::uintptr_t>(request.data);
        // Only declared buffers promise a stable mapping beyond this call.
        // Retaining a fixed binding for a temporary caller allocation is unsafe.
        if (!std::any_of(config.buffers.begin(), config.buffers.end(), [&](const auto& buffer) {
            const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
            return address >= begin && address - begin <= buffer.bytes &&
                   request.bytes <= buffer.bytes - (address - begin);
        })) return;
        // Every previous wave is drained. Release the old window and optional
        // RMW registration first, reserving the available quota for payload.
        if (buffers_live) {
            const auto rc = io_uring_unregister_buffers(&ring);
            require(rc == 0, -rc, "Release fixed buffer window");
        }
        buffers_live = pool_registered = false;
        window = {};
        registered_bytes.store(0, std::memory_order_relaxed);
        iovec buffer{request.data, request.bytes};
        const auto rc = io_uring_register_buffers(&ring, &buffer, 1);
        if (rc == 0) {
            buffers_live = true;
            window = {request.data, request.bytes};
            registered_bytes.store(request.bytes, std::memory_order_relaxed);
            window_registrations.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        registration_failures.fetch_add(1, std::memory_order_relaxed);
        require(rc == -ENOMEM || rc == -EPERM || rc == -EAGAIN, -rc, "Register fixed buffer window");
    }
    struct Part {
        void* buffer;
        std::size_t bytes, done = 0;
        std::uint64_t offset;
        int index;
    };
    void wave(bool write, Part* parts, std::size_t count, bool payload = true) {
        std::size_t remaining = count;
        while (remaining) {
            unsigned prepared = 0, fixed = 0;
            for (std::size_t i = 0; i < count; ++i) {
                auto& part = parts[i];
                if (part.done == part.bytes) continue;
                auto* sqe = io_uring_get_sqe(&ring);
                if (!sqe) std::terminate();
                auto* buffer = static_cast<char*>(part.buffer) + part.done;
                const auto bytes = unsigned(part.bytes - part.done);
                if (part.index >= 0) {
                    if (write) io_uring_prep_write_fixed(sqe, 0, buffer, bytes, part.offset + part.done, part.index);
                    if (!write) io_uring_prep_read_fixed(sqe, 0, buffer, bytes, part.offset + part.done, part.index);
                }
                if (part.index < 0) {
                    if (write) io_uring_prep_write(sqe, 0, buffer, bytes, part.offset + part.done);
                    if (!write) io_uring_prep_read(sqe, 0, buffer, bytes, part.offset + part.done);
                }
                sqe->flags |= IOSQE_FIXED_FILE;
                sqe->user_data = i;
                ++prepared;
                fixed += part.index >= 0;
            }
            complete(prepared, true);
            fixed_ops.fetch_add(fixed, std::memory_order_relaxed);
            ordinary_ops.fetch_add(prepared - fixed, std::memory_order_relaxed);
            peak.store(std::max(peak.load(std::memory_order_relaxed), std::size_t(prepared)), std::memory_order_relaxed);
            (write ? write_ops : read_ops).fetch_add(prepared, std::memory_order_relaxed);
            unsigned head, consumed = 0;
            io_uring_cqe* cqe;
            int error = 0;
            std::uint64_t completed_bytes = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                if (cqe->user_data >= count) std::terminate();
                auto& part = parts[cqe->user_data];
                const auto result = cqe->res;
                if (result <= 0 || std::size_t(result) > part.bytes - part.done || result % page_bytes) {
                    if (!error) error = result < 0 ? -result : EIO;
                } else {
                    part.done += result;
                    completed_bytes += result;
                    if (part.done == part.bytes) --remaining;
                }
                ++consumed;
            }
            io_uring_cq_advance(&ring, consumed);
            completions.fetch_add(consumed, std::memory_order_relaxed);
            (write ? write_bytes : read_bytes).fetch_add(completed_bytes, std::memory_order_relaxed);
            require(!error, error, "Direct I/O completion"); // All accepted SQEs are drained.
        }
        if (payload) {
            std::uint64_t bytes = 0;
            for (std::size_t i = 0; i < count; ++i) bytes += parts[i].bytes;
            (write ? direct_write : direct_read).fetch_add(bytes, std::memory_order_relaxed);
        }
    }
    void aligned(bool write, const FileIORequest* requests, std::size_t count) {
        if (count == 1 && !external_registered) register_single(requests[0]);
        std::array<Part, max_depth> parts;
        std::size_t first = 0, done = 0;
        while (first < count) {
            std::size_t n = 0;
            while (first < count && n < config.queue_depth) {
                const auto& r = requests[first];
                auto* caller = static_cast<char*>(r.data) + done;
                const auto whole = std::min(r.bytes - done, std::size_t(0x7ffff000));
                // Existing batch parallelism needs one SQE per block. Only a
                // singleton can be striped; registration never changes its size.
                const auto bytes = count == 1 ? std::min(whole, config.stripe_bytes) : whole;
                parts[n++] = {caller, bytes, 0, r.offset + done, registered(caller, bytes)};
                done += bytes;
                if (done == r.bytes) { ++first; done = 0; }
            }
            wave(write, parts.data(), n);
        }
    }
    void run(bool write, const FileIORequest* requests, std::size_t count) {
        // Aligned requests never visit the RMW loop or allocate request vectors.
        if (std::all_of(requests, requests + count, [](const auto& r) {
            return !r.prefix_bytes && !((r.offset | r.bytes | reinterpret_cast<std::uintptr_t>(r.data)) % page_bytes);
        })) { aligned(write, requests, count); return; }
        std::array<FileIORequest, 128> aligned_requests;
        std::size_t aligned_count = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& r = requests[i];
            if (!r.prefix_bytes && !((r.offset | r.bytes | reinterpret_cast<std::uintptr_t>(r.data)) % page_bytes)) {
                aligned_requests[aligned_count++] = r;
                continue;
            }
            for (std::size_t done = 0; done < r.prefix_bytes + r.bytes;) {
                const auto offset = r.offset + done;
                const auto prefix = std::size_t(offset % page_bytes);
                const auto bytes = std::min(r.prefix_bytes + r.bytes - done, page_bytes - prefix);
                Part part{pool.data, page_bytes, 0, offset - prefix, pool_registered ? 0 : -1};
                if (!write || prefix || bytes != page_bytes) wave(false, &part, 1, false);
                auto* buffer = static_cast<char*>(pool.data) + prefix;
                if (write) {
                    const auto head = done < r.prefix_bytes ? std::min(bytes, r.prefix_bytes - done) : 0;
                    if (head) std::memcpy(buffer, static_cast<const char*>(r.prefix) + done, head);
                    if (bytes > head) std::memcpy(buffer + head, static_cast<char*>(r.data) + done + head - r.prefix_bytes, bytes - head);
                    part.done = 0;
                    wave(true, &part, 1, false);
                } else std::memcpy(static_cast<char*>(r.data) + done, buffer, bytes);
                (write ? copied_write : copied_read).fetch_add(bytes, std::memory_order_relaxed);
                done += bytes;
            }
        }
        if (aligned_count) aligned(write, aligned_requests.data(), aligned_count);
    }
};
#endif
} // namespace

struct DirectFile::Impl {
    int fd = -1;
    std::size_t capacity;
    DirectFileConfig config;
#ifdef INFER_HAS_IO_URING
    mutable std::mutex control;
    std::vector<std::shared_ptr<Lane>> lanes;
    std::shared_ptr<Lane> lane() {
        struct Cache { const Impl* owner = nullptr; std::weak_ptr<Lane> lane; };
        thread_local std::vector<Cache> caches;
        for (const auto& cache : caches)
            if (cache.owner == this) if (auto value = cache.lane.lock()) return value;
        // Initialization only: each worker creates its own SINGLE_ISSUER ring.
        auto value = std::make_shared<Lane>(fd, config);
        { std::lock_guard<std::mutex> lock(control); lanes.push_back(value); }
        const auto empty = std::find_if(caches.begin(), caches.end(), [](const auto& cache) {
            return cache.lane.expired();
        });
        if (empty != caches.end()) *empty = {this, value};
        else caches.push_back({this, value});
        return value;
    }
#endif
    Impl(const std::string& directory, std::size_t bytes, DirectFileConfig value)
        : capacity(bytes), config(std::move(value)) {
        if (!capacity || capacity % page_bytes || capacity > INT64_MAX ||
            !config.queue_depth || config.queue_depth > max_depth ||
            !config.stripe_bytes || config.stripe_bytes % page_bytes || config.stripe_bytes > 4 * 1024 * 1024 ||
            config.bounce_bytes < page_bytes || config.bounce_bytes > max_depth * huge_bytes ||
            config.buffers.size() > 1023 || directory.empty() || directory.find('\0') != std::string::npos)
            throw std::invalid_argument("Invalid direct-file configuration");
        for (const auto& buffer : config.buffers)
            if (!buffer.data || !buffer.bytes || buffer.bytes > UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(buffer.data))
                throw std::invalid_argument("Invalid fixed-buffer range");
#ifndef INFER_HAS_IO_URING
        throw std::runtime_error("DirectFile requires liburing");
#else
        config.queue_depth = max_depth;
        raise_memlock();
        std::string name = directory + "/infer-direct-XXXXXX";
        const auto temporary = mkostemp(name.data(), O_CLOEXEC);
        require(temporary >= 0, errno, "Create direct cache");
        fd = open(("/proc/self/fd/" + std::to_string(temporary)).c_str(), O_RDWR | O_CLOEXEC | O_DIRECT | O_NOATIME);
        const auto error = errno;
        unlink(name.c_str()); close(temporary);
        require(fd >= 0, error, "Open direct cache");
        try {
            require(direct_alignment(fd).bytes == page_bytes, EOPNOTSUPP, "Direct-cache alignment");
            if (fallocate(fd, 0, 0, capacity) != 0)
                throw std::system_error(errno, std::generic_category(), "Preallocate direct cache");
            posix_fadvise(fd, 0, capacity, POSIX_FADV_RANDOM);
            posix_fadvise(fd, 0, capacity, POSIX_FADV_NOREUSE);
            Pool zeros(huge_bytes);
            // fallocate can leave unwritten extents. Materialize them outside
            // the hot path so measured writes really overwrite written blocks.
            for (std::size_t done = 0; done < capacity;) {
                const auto n = pwrite(fd, zeros.data, std::min(zeros.bytes, capacity - done), done);
                if (n < 0 && errno == EINTR) continue;
                require(n > 0, n < 0 ? errno : EIO, "Initialize direct extents");
                done += n;
            }
        } catch (...) { close(fd); fd = -1; throw; }
#endif
    }
    ~Impl() {
#ifdef INFER_HAS_IO_URING
        lanes.clear(); // Unregister before closing fd or releasing borrowed buffers.
#endif
        if (fd >= 0) close(fd);
    }
};
DirectFile::DirectFile(const std::string& directory, std::size_t capacity, DirectFileConfig config)
    : impl_(std::make_unique<Impl>(directory, capacity, std::move(config))) {}
DirectFile::~DirectFile() = default;
std::size_t DirectFile::alignment() const noexcept { return page_bytes; }
std::uint64_t DirectFile::capacity_bytes() const noexcept { return impl_->capacity; }
void DirectFile::transfer(bool write, const FileIORequest* requests, std::size_t count) {
    if (!count) return;
    if (!requests || count > 128) throw std::invalid_argument("Invalid direct-file batch");
    for (std::size_t i = 0; i < count; ++i) {
        const auto& r = requests[i];
        const auto address = reinterpret_cast<std::uintptr_t>(r.data);
        if (!r.data || !r.bytes || r.bytes > UINTPTR_MAX - address || r.offset > impl_->capacity ||
            r.prefix_bytes > impl_->capacity - r.offset || r.bytes > impl_->capacity - r.offset - r.prefix_bytes ||
            (r.prefix_bytes && (!write || !r.prefix || r.prefix_bytes > UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(r.prefix))))
            throw std::invalid_argument("Direct device extent exceeds capacity");
        for (std::size_t j = 0; j < i; ++j) {
            const auto begin = write ? r.offset : address;
            const auto other = write ? requests[j].offset : reinterpret_cast<std::uintptr_t>(requests[j].data);
            const auto bytes = write ? r.prefix_bytes + r.bytes : r.bytes;
            const auto other_bytes = requests[j].bytes + (write ? requests[j].prefix_bytes : 0);
            if (begin < other + other_bytes && other < begin + bytes)
                throw std::invalid_argument("Overlapping direct-file destinations");
        }
    }
#ifdef INFER_HAS_IO_URING
    impl_->lane()->run(write, requests, count);
#endif
}
void DirectFile::read(void* data, std::size_t bytes, std::uint64_t offset) {
    const FileIORequest request{data, bytes, offset}; transfer(false, &request, 1);
}
void DirectFile::write(const void* data, std::size_t bytes, std::uint64_t offset) {
    const FileIORequest request{const_cast<void*>(data), bytes, offset}; transfer(true, &request, 1);
}
void DirectFile::read_many(const std::vector<FileIORequest>& r) { transfer(false, r.data(), r.size()); }
void DirectFile::write_many(const std::vector<FileIORequest>& r) { transfer(true, r.data(), r.size()); }
FileIOStats DirectFile::file_io_stats() const {
    FileIOStats result;
#ifdef INFER_HAS_IO_URING
    std::lock_guard<std::mutex> lock(impl_->control);
    result.uring_available = result.calibrated = result.registered_file = true;
    result.selected = FileIOMode::IO_URING;
    result.queue_depth = impl_->config.queue_depth;
    result.rings = impl_->lanes.size();
    for (const auto& lane : impl_->lanes) {
        result.read_operations += lane->read_ops.load(); result.write_operations += lane->write_ops.load();
        result.read_bytes += lane->read_bytes.load(); result.write_bytes += lane->write_bytes.load();
        result.submissions += lane->submissions.load(); result.completions += lane->completions.load();
        result.completion_waits += lane->waits.load();
        result.peak_inflight = std::max(result.peak_inflight, lane->peak.load());
        result.setup_flags |= lane->flags;
        result.registered_buffer_bytes += lane->registered_bytes.load();
        result.buffer_window_registrations += lane->window_registrations.load();
        result.fixed_buffer_operations += lane->fixed_ops.load();
        result.ordinary_buffer_operations += lane->ordinary_ops.load();
        result.buffer_registration_failures += lane->registration_failures.load();
        result.hugetlb_pools += lane->pool.hugetlb;
    }
#endif
    return result;
}
SSDDeviceStats DirectFile::stats() const {
    SSDDeviceStats result;
#ifdef INFER_HAS_IO_URING
    std::lock_guard<std::mutex> lock(impl_->control);
    for (const auto& lane : impl_->lanes) {
        result.dma_reserved_bytes += lane->pool.bytes;
        result.read_bytes += lane->read_bytes.load(); result.written_bytes += lane->write_bytes.load();
        result.completed_ios += lane->completions.load();
        result.copied_read_bytes += lane->copied_read.load(); result.copied_write_bytes += lane->copied_write.load();
        result.direct_read_bytes += lane->direct_read.load(); result.direct_written_bytes += lane->direct_write.load();
    }
#endif
    return result;
}
std::unique_ptr<SSDDevice> open_direct_file_device(const std::string& directory, std::size_t capacity, std::size_t depth) {
    DirectFileConfig config;
    config.queue_depth = depth;
    return std::make_unique<DirectFile>(directory, capacity, std::move(config));
}
} // namespace infer::io
