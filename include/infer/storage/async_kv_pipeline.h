#pragma once

#include "infer/utils/cuda_check.h"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>

#if INFER_ENABLE_CUDA
namespace infer {

struct LayerRead {
    std::uint64_t generation;
    std::size_t layer, pages;
    std::byte* destination;
};

struct ReadDone {
    std::uint64_t generation;
    std::size_t layer;
    std::exception_ptr error;
};

// The caller owns storage callbacks and all job buffers until retirement.
// A single scheduler submits work and consumes completions.
class AsyncKVPipeline {
  public:
    struct Config {
        std::size_t layers, max_pages, page_bytes, tokens_per_page, row_bytes;
        int device;
        std::size_t max_device_bytes = 0; // Zero derives the capacity from max_pages.
        bool gpu_resident = false; // One persistent slot per layer; write back at prefix publication.
    };
    struct LayerView {
        float* key;
        float* value;
        const float* const* pages;
        std::size_t tokens_per_page, row_elements;
    };
    using Read = std::function<void(const LayerRead&)>;
    using Write = std::function<void(std::size_t layer, std::size_t position,
                                     const std::byte* key, const std::byte* value)>;
    using Compute = std::function<void(std::size_t layer, LayerView, cudaStream_t)>;

    AsyncKVPipeline(Config config, Read read, Write write);
    ~AsyncKVPipeline();
    AsyncKVPipeline(const AsyncKVPipeline&) = delete;
    AsyncKVPipeline& operator=(const AsyncKVPipeline&) = delete;

    void begin_token(std::size_t position, std::size_t count = 1);
    // Streaming mode completes KV writebacks. Resident mode queues kernels on
    // compute_stream; download/retirement fence them, flush_resident persists KV.
    bool tick(const Compute& compute);
    cudaStream_t compute_stream() const noexcept;
    // Stable addresses for graph capture before a token is admitted. key/value
    // identify row zero; replay must address the current row on the device.
    LayerView graph_slot(std::size_t slot) const;
    bool gpu_resident() const noexcept;
    std::size_t layer_count() const noexcept;
    void flush_resident();

    // Submit a copy of compute output to a caller-owned pinned buffer.
    void download(void* pinned, const void* device, std::size_t bytes);
    bool poll_download();

    // Stop admitting work and fence already queued CUDA operations. Continue
    // polling until all buffers and storage callbacks have retired.
    void begin_retire();
    bool poll_retired();
    std::size_t reserved_device_bytes() const noexcept;
    std::size_t staging_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infer
#endif
