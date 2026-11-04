#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "infer/storage/ssd_device.h"
namespace mini_infer {
enum class ModelKVMode { AUTO, CONTIGUOUS, PAGED };
struct PagedModelConfig {
    std::size_t page_bytes=64*1024, tokens_per_page=0;
    std::size_t resident_layers=2; // Auto GPU budget: this many max-context layers; 0 keeps all.
    // Zero derives bounded capacities from the model's maximum context.
    std::size_t gpu_bytes=0, host_bytes=0, ssd_bytes=0;
    std::string ssd_directory="/tmp";
    bool prefetch=true, prefix_cache=true;
    infer::FileIOConfig ssd_io{infer::FileIOMode::IO_URING};
    infer::SpdkBdevConfig ssd_spdk;
    bool ssd_direct=true, ssd_read_back=false;
    // Two streaming slots, or persistent per-layer slots when resident_layers
    // is zero and the GPU budget covers them. Prefix publication flushes KV.
    bool enable_async_kv_pipeline=false;
};
struct ModelOptions {
    ModelKVMode kv_mode=ModelKVMode::AUTO; // CUDA: paged; CPU: contiguous.
    PagedModelConfig paged;
    bool enable_cuda_graph=false; // Requires the fixed-slot async KV pipeline.
};
struct PagedModelStats {
    std::uint64_t computed_tokens=0,prefix_hit_tokens=0,last_prefix_hit_tokens=0;
    std::uint64_t demand_fault_pages=0,prefetched_pages=0,prefetch_windows=0;
    std::uint64_t gpu_evicted_pages=0,ssd_spilled_pages=0,prefix_evictions=0,prefetch_skipped=0;
    std::uint64_t layer_calls=0,prefetch_wait_ns=0;
    std::size_t page_bytes=0,tokens_per_page=0,pages=0;
    std::size_t gpu_limit_bytes=0,host_limit_bytes=0,ssd_limit_bytes=0;
    std::size_t gpu_reserved_bytes=0,host_reserved_bytes=0,ssd_reserved_bytes=0;
    std::size_t gpu_peak_bytes=0,host_peak_bytes=0,ssd_peak_bytes=0,staging_bytes=0;
    std::uint64_t ssd_read_bytes=0,ssd_written_bytes=0,ssd_record_writes=0,uring_submissions=0;
    std::uint64_t host_read_bytes=0,host_written_bytes=0;
    infer::SSDDeviceStats ssd_device;
};
}
