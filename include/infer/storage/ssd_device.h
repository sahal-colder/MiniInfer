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
    std::uint64_t copied_read_bytes = 0, copied_write_bytes = 0;
    std::uint64_t direct_read_bytes = 0, direct_written_bytes = 0;
};

// A device-owned DMA allocation. Its owner keeps the SPDK environment alive.
// Transfer buffers also own their CUDA registration. Finish all DMA before
// dropping the final reference; externally added registrations remain caller-owned.
class SSDDmaBuffer {
  public:
    virtual ~SSDDmaBuffer() = default;
    virtual void* data() const noexcept = 0;
    virtual std::size_t size() const noexcept = 0;
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
    virtual FileIOStats file_io_stats() const { return {}; }
    // Filesystem devices return empty. SPDK page-rounds and charges live leases
    // plus its fixed bounce slots to dma_budget_bytes; exhaustion throws.
    virtual std::shared_ptr<SSDDmaBuffer> allocate_dma_buffer(std::size_t) { return {}; }
    virtual bool owns_dma_range(const void*, std::size_t) const { return false; }
    // Owns CUDA registration as well as DMA memory. Empty when unsupported.
    virtual std::shared_ptr<SSDDmaBuffer> allocate_transfer_buffer(std::size_t) { return {}; }
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
    bool no_huge = false;      // Dedicated PCI NVMe; file/malloc tests opt out explicitly.
    bool no_pci = false;
    std::string iova_mode = "auto"; // va: unprivileged VFIO/IOMMU; pa: physical addresses.
};

bool spdk_available() noexcept;
// EAL retains its reusable arena until process exit, including after a backend
// closes. Report that process resource independently of live logical blocks.
std::size_t spdk_environment_bytes() noexcept;
// One SPDK app lifecycle per process. Leases keep its reactor alive after the
// device closes; after the final lease/device closes, reopening needs a new
// process (EAL reinitialization cannot migrate its original main-thread TLS).
// Only the reactor accesses bdev/channel APIs; callers retain I/O buffers until
// completion and share its bounded submission slots.
std::unique_ptr<SSDDevice> open_spdk_device(const SpdkBdevConfig& config);

} // namespace infer
