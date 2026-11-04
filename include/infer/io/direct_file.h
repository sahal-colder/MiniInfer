#pragma once
#include "infer/storage/ssd_device.h"
namespace infer::io {
struct DirectFileBuffer {
    void* data;
    std::size_t bytes;
};
struct DirectFileConfig {
    std::size_t queue_depth = 256; // DirectFile reserves at least 256 SQEs.
    std::size_t stripe_bytes = 4 * 1024 * 1024;
    std::size_t bounce_bytes = 2 * 1024 * 1024; // RMW scratch only; aligned requests never copy.

    bool sqpoll = true;
    bool iopoll = true;
    // Borrowed buffers are registered once per worker. Keep them alive until
    // the device is destroyed. If the full table cannot fit, singleton requests
    // reuse a <=4 MiB fixed window. Other unregistered aligned ranges use
    // ordinary READ/WRITE on the same pointer.
    std::vector<DirectFileBuffer> buffers;
};
// Private, preallocated O_DIRECT file. Each calling thread owns its ring;
// callers serialize overlapping destinations and partial-sector writes.
class DirectFile final : public SSDDevice {
  public:
    DirectFile(const std::string& directory, std::size_t capacity_bytes,
               DirectFileConfig config = {});
    ~DirectFile() override;
    std::size_t alignment() const noexcept override;
    std::uint64_t capacity_bytes() const noexcept override;
    void read(void* data, std::size_t bytes, std::uint64_t offset) override;
    void write(const void* data, std::size_t bytes, std::uint64_t offset) override;
    void read_many(const std::vector<FileIORequest>& requests) override;
    void write_many(const std::vector<FileIORequest>& requests) override;
    // Concrete, allocation-free request entry for storage workers/benchmarks.
    void transfer(bool write, const FileIORequest* requests, std::size_t count);
    FileIOStats file_io_stats() const override;
    SSDDeviceStats stats() const override;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::unique_ptr<SSDDevice> open_direct_file_device(const std::string& directory,
    std::size_t capacity_bytes, std::size_t queue_depth = 8);
}
