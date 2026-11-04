#pragma once
#include "infer/io/safetensors.h"
#include "infer/storage/storage_block.h"
#include <memory>
#include <string>
#include <vector>
namespace infer::io {
struct GpuRead { std::uint64_t file_offset; void* destination; std::size_t bytes; };
struct LoadStats {
    std::string backend, fallback_reason;
    std::uint64_t disk_bytes = 0, tensor_bytes = 0, dma_calls = 0, read_operations = 0;
    std::size_t staging_bytes = 0, alignment = page_bytes;
    bool registered_buffers = false, registered_file = false, alignment_queried = false;
    // Selecting cuFile is not proof that a particular filesystem used P2P.
    bool p2p_verified = false;
};
struct WeightIOConfig {
    std::size_t chunk_bytes = 4 * 1024 * 1024;
    int device = 0;
    bool try_cufile = true;
};
class StorageBackend {
  public:
    virtual ~StorageBackend() = default;
    // Input file, GPU destinations and stream remain live until return.
    // Success means all reads and DMA completed; error drains accepted work.
    virtual LoadStats read(const Safetensors&, std::vector<GpuRead>, cudaStream_t) = 0;
};
std::unique_ptr<StorageBackend> make_weight_backend(WeightIOConfig config = {});
// One GPU allocation, 256-byte-aligned tensor views, no full CPU weight copy.
class DeviceWeights {
  public:
    DeviceWeights(const Safetensors&, StorageBackend&, cudaStream_t, int device = 0);
    ~DeviceWeights();
    DeviceWeights(const DeviceWeights&) = delete;
    DeviceWeights& operator=(const DeviceWeights&) = delete;
    const void* data(std::string_view name) const;
    std::size_t allocated_bytes() const noexcept;
    const LoadStats& stats() const noexcept;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace infer::io
