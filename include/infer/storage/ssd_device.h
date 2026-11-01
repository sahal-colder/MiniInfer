#pragma once

#include "infer/storage/file_io.h"
#include <memory>
#include <string>

namespace infer {

struct SSDDeviceStats {
    // Usable DMA buffer bytes; EAL allocator padding is included in the arena.
    std::size_t dma_reserved_bytes = 0;
    // Entire process-owned EAL heap, including DMA, mempools and unused space.
    std::size_t environment_bytes = 0;
    std::uint64_t read_bytes = 0, written_bytes = 0, completed_ios = 0;
};

// Byte-range adapter below the SSD record/CRC layer. Calls finish all accepted
// I/O before returning or throwing. The record owner serializes overlapping
// writes; alignment() keeps distinct records in independent physical sectors.
class SSDDevice {
  public:
    virtual ~SSDDevice() = default;
    virtual std::size_t alignment() const noexcept = 0;
    virtual std::uint64_t capacity_bytes() const noexcept = 0;
    virtual void read(void* destination, std::size_t bytes, std::uint64_t offset) = 0;
    virtual void write(const void* source, std::size_t bytes, std::uint64_t offset) = 0;
    virtual void read_many(const std::vector<FileIORequest>& requests) = 0;
    virtual void write_many(const std::vector<FileIORequest>& requests) {
        for (const auto& request : requests) {
            if (request.prefix_bytes)
                write(request.prefix, request.prefix_bytes, request.offset);
            write(request.data, request.bytes, request.offset + request.prefix_bytes);
        }
    }
    virtual SSDDeviceStats stats() const = 0;
};

struct SpdkBdevConfig {
    std::string json_config; // SPDK app JSON: explicit bdev creation/attachment.
    std::string bdev_name;   // Empty disables SPDK; no implicit device selection.
    std::string reactor_mask = "0x1";
    std::uint64_t offset_bytes = 0, capacity_bytes = 0; // Explicit, exclusive region.
    std::size_t queue_depth = 8;
    std::size_t io_buffer_bytes = 256 * 1024;
    std::size_t dma_budget_bytes = 2 * 1024 * 1024;
    int environment_mib = 256; // Includes DMA buffers; separate from CPU KV budget.
    bool no_huge = true;       // File/malloc testing. Direct PCI NVMe needs hugepages.
    bool no_pci = true;
};

bool spdk_available() noexcept;
// EAL retains its reusable arena until process exit, including after a backend
// closes. Report that process resource independently of live logical blocks.
std::size_t spdk_environment_bytes() noexcept;
// One owning SPDK app per process. Only its reactor accesses bdev/channel APIs;
// callers use a bounded submission queue and retain their buffers until completion.
std::unique_ptr<SSDDevice> open_spdk_device(const SpdkBdevConfig& config);

} // namespace infer
