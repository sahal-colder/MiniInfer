#pragma once

#include "infer/config.h"
#if !INFER_ENABLE_NATIVE
#error "Build Infer with INFER_BUILD_NATIVE=ON to use the native backend."
#endif

#include "infer/core/allocator.h"
#include "infer/storage/kv_offload_cache.h"
#include "mini_infer/tensor/mini_infer_kv_cache.h"

namespace infer::native {

// Inject the same infer pool into native Tensors, KV caches or LlamaModel.
// Copy/zero retain mini_infer's synchronization contract.
using AllocatorAdapter = mini_infer::DeviceAllocator;

// Synchronous checkpoints of the native contiguous FP32 KV layout. Caller slot
// IDs still address layers; records hold up to 256-token-equivalent bytes.
// Tail records retain exact lengths and use actual CPU size classes.
// cpu_slots retains the previous rounded layer-byte budget, shared by page slots.
// This does not turn native attention into paged attention.
// The caller owns slot lifetimes; operations on this adapter are serialized.
class KVCacheStorage {
  public:
    KVCacheStorage(const mini_infer::KVCache& layout, KVOffloadConfig config);
    void save(const mini_infer::KVCache& cache, std::size_t first_slot);
    // Stage and validate all valid prefixes before changing cache. GPU copy failures
    // may leave a partial restore; reset the cache before using it again.
    void restore(mini_infer::KVCache& cache, std::size_t first_slot);
    KVOffloadStats stats() const { return storage_.stats(); }

  private:
    struct Layer {
        int32_t length = -1;   // Unwritten or an incomplete save; zero is a saved empty layer.
        std::size_t pages = 0; // Includes possibly written pages after a failed batch.
    };
    void check(const mini_infer::KVCache& cache, std::size_t first_slot) const;
    int32_t layers_, capacity_, dim_;
    std::size_t elements_, slots_, pages_per_layer_;
    KVOffloadCache storage_;
    std::vector<Layer> checkpoints_;
};

} // namespace infer::native
