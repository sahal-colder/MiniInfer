#pragma once
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace infer {
enum class FileIOMode { AUTO, POSIX, IO_URING };
struct FileIOConfig {
    FileIOMode mode = FileIOMode::POSIX;
    std::size_t queue_depth = 8; // 上限，不是必须达到的并发；不分配额外 payload 缓冲。
    bool force_async = false;    // Optional kernel-worker dispatch for large ranges.
    std::size_t file_shards = 1; // SSD backend: 1 or 4 private files, same total slot budget.
};
struct FileIORequest {
    void* data = nullptr;
    std::size_t bytes = 0;
    std::uint64_t offset = 0;
    // Optional write-only prefix at offset; payload follows without a staging copy.
    const void* prefix = nullptr;
    std::size_t prefix_bytes = 0;
};
struct FileIOStats {
    bool uring_available = false;
    bool calibrated = false;
    FileIOMode selected = FileIOMode::POSIX;
    std::size_t queue_depth = 1;
    std::uint64_t calibration_ns = 0;
    bool registered_file = false;
    std::uint64_t read_operations = 0, write_operations = 0;
    std::uint64_t read_bytes = 0, write_bytes = 0;
    std::uint64_t submissions = 0, completions = 0, completion_waits = 0;
    std::size_t peak_inflight = 0;
    std::size_t rings = 0;
    std::uint64_t fixed_buffer_operations = 0, ordinary_buffer_operations = 0;
    std::size_t buffer_registration_failures = 0, buffer_window_registrations = 0;
    unsigned setup_flags = 0;
    std::size_t registered_buffer_bytes = 0, hugetlb_pools = 0;
};

// Bounded asynchronous file reads/writes. Callers retain fd and buffers until
// the future completes; synchronous methods drain the entire accepted batch.
// 默认 POSIX 直接完成读取，不建 ring、不在业务请求中重复读数据做基准。
// AUTO prefers an available ring, without reading user data for calibration.
// 不改变 page-cache/fsync 语义，也不把文件完成事件当作 CUDA 完成事件。
// An optional registered_fd must keep referring to the same open file until
// destruction. Sync calls execute directly on the caller's I/O worker. Async
// calls use a bounded lazy worker; neither path uses notification descriptors.
class FileIO {
  public:
    explicit FileIO(FileIOConfig config = {}, int registered_fd = -1);
    ~FileIO();
    FileIO(const FileIO&) = delete;
    FileIO& operator=(const FileIO&) = delete;
    void read_many(int fd, const std::vector<FileIORequest>& requests);
    void write_many(int fd, const std::vector<FileIORequest>& requests);
    std::future<void> read_many_async(int fd, std::vector<FileIORequest> requests);
    std::future<void> write_many_async(int fd, std::vector<FileIORequest> requests);
    FileIOStats stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace infer
