#include "infer/storage/file_io.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/uio.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#ifdef INFER_HAS_IO_URING
#include <liburing.h>
#endif

namespace infer {
namespace {
std::size_t transfer_bytes(const FileIORequest& r) {
    if (r.prefix_bytes > SIZE_MAX - r.bytes)
        throw std::invalid_argument("FileIO prefix size overflow");
    return r.prefix_bytes + r.bytes;
}
int vectors(const FileIORequest& r, std::size_t done, iovec* out) {
    int count = 0;
    if (done < r.prefix_bytes) {
        out[count++] = {const_cast<char*>(static_cast<const char*>(r.prefix)) + done,
                        r.prefix_bytes - done};
        done = 0;
    } else {
        done -= r.prefix_bytes;
    }
    if (done < r.bytes)
        out[count++] = {static_cast<char*>(r.data) + done, r.bytes - done};
    return count;
}
void validate(const std::vector<FileIORequest>& requests, bool write) {
    if (requests.size() > 128)
        throw std::invalid_argument("FileIO batch exceeds 128 ranges");
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& r = requests[i];
        const auto address = reinterpret_cast<std::uintptr_t>(r.data);
        const auto prefix = reinterpret_cast<std::uintptr_t>(r.prefix);
        const auto bytes = transfer_bytes(r);
        if (!r.data || !r.bytes || r.bytes > UINTPTR_MAX - address ||
            (r.prefix_bytes && (!write || !r.prefix || r.prefix_bytes > UINTPTR_MAX - prefix)) ||
            r.offset > std::uint64_t(std::numeric_limits<off_t>::max()) ||
            bytes > std::uint64_t(std::numeric_limits<off_t>::max()) - r.offset)
            throw std::invalid_argument("Invalid FileIO range");
        for (std::size_t j = 0; j < i; ++j) {
            const auto other =
                write ? requests[j].offset : reinterpret_cast<std::uintptr_t>(requests[j].data);
            const auto begin = write ? r.offset : address;
            const auto other_bytes = write ? transfer_bytes(requests[j]) : requests[j].bytes;
            if (begin < other + other_bytes && other < begin + (write ? bytes : r.bytes))
                throw std::invalid_argument("Overlapping FileIO destinations");
        }
    }
}
void read_adjacent(int fd, const std::vector<FileIORequest>& requests, std::size_t first,
                   std::size_t end) {
    std::array<iovec, 128> parts;
    std::size_t done = 0;
    while (first < end) {
        std::size_t left = INT_MAX;
        int count = 0;
        for (auto i = first; i < end && left; ++i) {
            const auto skip = i == first ? done : 0;
            const auto bytes = std::min(requests[i].bytes - skip, left);
            parts[count++] = {static_cast<char*>(requests[i].data) + skip, bytes};
            left -= bytes;
        }
        const auto n = ::preadv(fd, parts.data(), count, requests[first].offset + done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), "FileIO POSIX");
        }
        if (!n)
            throw std::runtime_error("FileIO unexpected EOF");
        auto remaining = std::size_t(n);
        while (remaining) {
            const auto bytes = std::min(remaining, requests[first].bytes - done);
            remaining -= bytes;
            done += bytes;
            if (done == requests[first].bytes) {
                ++first;
                done = 0;
            }
        }
    }
}
void posix_io(int fd, const std::vector<FileIORequest>& requests, bool write) {
    for (std::size_t first = 0; first < requests.size(); ++first) {
        if (!write) {
            auto end = first + 1;
            while (end < requests.size() &&
                   requests[end - 1].offset + requests[end - 1].bytes == requests[end].offset)
                ++end;
            if (end > first + 1) {
                read_adjacent(fd, requests, first, end);
                first = end - 1;
                continue;
            }
        }
        const auto& r = requests[first];
        std::size_t done = 0;
        const auto total = transfer_bytes(r);
        while (done < total) {
            const auto size = std::min(total - done, std::size_t(INT_MAX));
            ssize_t n;
            if (write && r.prefix_bytes) {
                iovec parts[2];
                const auto count = vectors(r, done, parts);
                // Limit each syscall as for the scalar path, including large payloads.
                std::size_t left = INT_MAX;
                for (int i = 0; i < count; ++i) {
                    parts[i].iov_len = std::min(parts[i].iov_len, left);
                    left -= parts[i].iov_len;
                }
                n = ::pwritev(fd, parts, count, r.offset + done);
            } else {
                auto* address = static_cast<char*>(r.data) + done;
                n = write ? ::pwrite(fd, address, size, r.offset + done)
                          : ::pread(fd, address, size, r.offset + done);
            }
            if (n < 0) {
                if (errno == EINTR)
                    continue;
                throw std::system_error(errno, std::generic_category(), "FileIO POSIX");
            }
            if (!n)
                throw std::runtime_error(write ? "FileIO write made no progress"
                                               : "FileIO unexpected EOF");
            done += std::size_t(n);
        }
    }
}
bool unavailable(int error) {
    return error == ENOSYS || error == EOPNOTSUPP || error == EPERM || error == EINVAL;
}
} // namespace

struct FileIO::Impl {
    struct Job {
        int fd;
        bool write;
        std::vector<FileIORequest> requests;
        std::promise<void> promise;
    };
    static constexpr std::size_t max_jobs = 64;
    FileIOConfig config;
    FileIOStats counters;
    mutable std::mutex stats_mutex;
    std::mutex ring_mutex, queue_mutex;
    std::condition_variable work, space;
    std::deque<std::unique_ptr<Job>> queued;
    std::thread worker;
    std::size_t accepted = 0;
    bool stopping = false;
    std::exception_ptr broken;
#ifdef INFER_HAS_IO_URING
    io_uring ring{};
    bool ring_live = false;
    int registered_fd = -1;
#endif

    Impl(FileIOConfig value, int fd) : config(value) {
        if (!config.queue_depth || config.queue_depth > 64)
            throw std::invalid_argument("FileIO queue depth must be 1..64");
        if (config.mode != FileIOMode::AUTO && config.mode != FileIOMode::POSIX &&
            config.mode != FileIOMode::IO_URING)
            throw std::invalid_argument("Unknown FileIO mode");
        counters.calibrated = true;
        if (config.mode == FileIOMode::POSIX)
            return;
#ifdef INFER_HAS_IO_URING
        const int rc = io_uring_queue_init(static_cast<unsigned>(config.queue_depth), &ring, 0);
        if (rc < 0) {
            if (config.mode == FileIOMode::IO_URING)
                throw std::system_error(-rc, std::generic_category(), "io_uring_queue_init");
            return;
        }
        ring_live = true;
        if (fd >= 0 && io_uring_register_files(&ring, &fd, 1) == 0) {
            registered_fd = fd;
            counters.registered_file = true;
        }
        counters.uring_available = true;
        counters.selected = FileIOMode::IO_URING;
        counters.queue_depth = config.queue_depth;
        counters.rings = 1;
#else
        (void)fd;
        if (config.mode == FileIOMode::IO_URING)
            throw std::runtime_error("io_uring backend is unavailable in this build");
#endif
    }
    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stopping = true;
        }
        work.notify_all();
        space.notify_all();
        if (worker.joinable())
            worker.join();
#ifdef INFER_HAS_IO_URING
        close_ring();
#endif
    }
#ifdef INFER_HAS_IO_URING
    void close_ring() noexcept {
        if (ring_live) {
            io_uring_queue_exit(&ring);
            ring_live = false;
        }
    }
    // Only this call owns SQ/CQ. Existing I/O workers execute their batches
    // directly, without forwarding through another thread or notification fd.
    void uring_io(int fd, const std::vector<FileIORequest>& requests, bool write) {
        std::array<std::size_t, 128> done{};
        std::array<std::array<iovec, 2>, 128> parts{};
        for (std::size_t first = 0; first < requests.size();) {
            const auto end = std::min(requests.size(), first + config.queue_depth);
            std::size_t remaining = end - first;
            while (remaining) {
                std::size_t prepared = 0;
                for (auto i = first; i < end; ++i) {
                    const auto& r = requests[i];
                    if (done[i] == transfer_bytes(r))
                        continue;
                    auto* sqe = io_uring_get_sqe(&ring);
                    if (!sqe)
                        std::terminate();
                    const bool fixed = fd >= 0 && fd == registered_fd;
                    const auto count =
                        unsigned(std::min(transfer_bytes(r) - done[i], std::size_t(INT_MAX)));
                    if (write && r.prefix_bytes) {
                        const auto n = vectors(r, done[i], parts[i].data());
                        std::size_t left = INT_MAX;
                        for (int j = 0; j < n; ++j) {
                            parts[i][j].iov_len = std::min(parts[i][j].iov_len, left);
                            left -= parts[i][j].iov_len;
                        }
                        io_uring_prep_writev(sqe, fixed ? 0 : fd, parts[i].data(), n,
                                             r.offset + done[i]);
                    } else {
                        auto* address = static_cast<char*>(r.data) + done[i];
                        if (write)
                            io_uring_prep_write(sqe, fixed ? 0 : fd, address, count,
                                                r.offset + done[i]);
                        else
                            io_uring_prep_read(sqe, fixed ? 0 : fd, address, count,
                                               r.offset + done[i]);
                    }
                    if (fixed)
                        sqe->flags |= IOSQE_FIXED_FILE;
                    if (config.force_async && count >= 1024 * 1024)
                        sqe->flags |= IOSQE_ASYNC;
                    sqe->user_data = i;
                    ++prepared;
                }
                int error = 0;
                std::size_t submitted = 0;
                while (submitted < prepared) {
                    const int rc = io_uring_submit(&ring);
                    if (rc == -EINTR)
                        continue;
                    if (rc <= 0) {
                        error = rc < 0 ? -rc : EIO;
                        break;
                    }
                    submitted += std::size_t(rc);
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    ++counters.submissions;
                }
                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    counters.peak_inflight = std::max(counters.peak_inflight, submitted);
                    (write ? counters.write_operations : counters.read_operations) += submitted;
                }
                for (std::size_t completed = 0; completed < submitted; ++completed) {
                    io_uring_cqe* cqe = nullptr;
                    int rc = io_uring_peek_cqe(&ring, &cqe);
                    if (rc == -EAGAIN) {
                        {
                            std::lock_guard<std::mutex> lock(stats_mutex);
                            ++counters.completion_waits;
                        }
                        do {
                            rc = io_uring_wait_cqe(&ring, &cqe);
                        } while (rc == -EINTR || rc == -EAGAIN || rc == -EBUSY || rc == -ENOMEM);
                    }
                    // The private ring stays open until every accepted I/O
                    // completes. Never return its buffers after an unknown CQ.
                    if (rc < 0)
                        std::terminate();
                    const auto i = cqe->user_data;
                    const int result = cqe->res;
                    io_uring_cqe_seen(&ring, cqe);
                    if (i < first || i >= end)
                        std::terminate();
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);
                        ++counters.completions;
                        if (result > 0)
                            (write ? counters.write_bytes : counters.read_bytes) += result;
                    }
                    if (result <= 0 ||
                        std::size_t(result) > transfer_bytes(requests[i]) - done[i]) {
                        if (!error)
                            error = result < 0 ? -result : EIO;
                    } else {
                        done[i] += std::size_t(result);
                        if (done[i] == transfer_bytes(requests[i]))
                            --remaining;
                    }
                }
                if (error) {
                    if (submitted < prepared) {
                        // No SQPOLL consumer: after submitted CQEs are drained,
                        // close to discard SQEs that never reached the kernel.
                        close_ring();
                        broken = std::make_exception_ptr(std::system_error(
                            error, std::generic_category(), "FileIO submission failed"));
                    }
                    throw std::system_error(error, std::generic_category(), "FileIO io_uring");
                }
            }
            first = end;
        }
    }
#endif
    void transfer(int fd, const std::vector<FileIORequest>& requests, bool write) {
        if (requests.empty())
            return;
        std::lock_guard<std::mutex> ring_lock(ring_mutex);
#ifdef INFER_HAS_IO_URING
        if (broken)
            std::rethrow_exception(broken);
        if (ring_live) {
            try {
                uring_io(fd, requests, write);
                return;
            } catch (const std::system_error& error) {
                if (config.mode != FileIOMode::AUTO || !unavailable(error.code().value()))
                    throw;
                // All submitted CQEs have been drained. Retrying ordinary
                // offset-based file operations preserves caller ownership.
                close_ring();
                broken = nullptr;
                std::lock_guard<std::mutex> lock(stats_mutex);
                counters.selected = FileIOMode::POSIX;
                counters.uring_available = false;
                counters.registered_file = false;
                counters.queue_depth = 1;
                counters.rings = 0;
            }
        }
#endif
        posix_io(fd, requests, write);
    }
    std::future<void> submit(int fd, std::vector<FileIORequest> requests, bool write) {
        validate(requests, write);
        auto job = std::make_unique<Job>();
        job->fd = fd;
        job->write = write;
        job->requests = std::move(requests);
        auto future = job->promise.get_future();
        std::unique_lock<std::mutex> lock(queue_mutex);
        space.wait(lock, [&] { return stopping || accepted < max_jobs; });
        if (stopping)
            throw std::runtime_error("FileIO is stopping");
        // Create the worker before accepting any buffers: failed thread
        // creation cannot strand an accepted operation in the queue.
        if (!worker.joinable())
            worker = std::thread([this] { run(); });
        queued.push_back(std::move(job));
        ++accepted;
        lock.unlock();
        work.notify_one();
        return future;
    }
    void run() noexcept {
        for (;;) {
            std::unique_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                work.wait(lock, [&] { return stopping || !queued.empty(); });
                if (queued.empty())
                    return;
                job = std::move(queued.front());
                queued.pop_front();
            }
            try {
                transfer(job->fd, job->requests, job->write);
                job->promise.set_value();
            } catch (...) {
                job->promise.set_exception(std::current_exception());
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                --accepted;
            }
            space.notify_all();
        }
    }
};

FileIO::FileIO(FileIOConfig config, int fd) : impl_(std::make_unique<Impl>(config, fd)) {}
FileIO::~FileIO() = default;
std::future<void> FileIO::read_many_async(int fd, std::vector<FileIORequest> requests) {
    return impl_->submit(fd, std::move(requests), false);
}
std::future<void> FileIO::write_many_async(int fd, std::vector<FileIORequest> requests) {
    return impl_->submit(fd, std::move(requests), true);
}
void FileIO::read_many(int fd, const std::vector<FileIORequest>& requests) {
    validate(requests, false);
    impl_->transfer(fd, requests, false);
}
void FileIO::write_many(int fd, const std::vector<FileIORequest>& requests) {
    validate(requests, true);
    impl_->transfer(fd, requests, true);
}
FileIOStats FileIO::stats() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->counters;
}
} // namespace infer
