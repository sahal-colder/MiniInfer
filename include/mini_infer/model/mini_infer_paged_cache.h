#pragma once
#include "mini_infer/model/mini_infer_model_options.h"
#include "mini_infer/loader/mini_infer_mmap.h"
#include "mini_infer/tensor/mini_infer_tensor.h"
#include "infer/storage/paged_kv.h"
namespace mini_infer {
// Single-request scheduler. PrefixIndex survives reset; consumers remain pinned
// by PageLease until all kernels on the explicit compute stream complete.
class PagedModelCache {
  public:
    struct Layer {
        std::vector<infer::PagedKV::PageLease> leases;
        Tensor key,value;
        const float* const* pages=nullptr;
        std::size_t page_count=0,tokens_per_page=0;
        void finish() { infer::PagedKV::finish_many(leases); }
    };
    PagedModelCache(const ModelConfig&,PagedModelConfig,std::shared_ptr<DeviceAllocator>);
    ~PagedModelCache();
    std::size_t begin_request(const int32_t* tokens,std::size_t count);
    void begin_token(int32_t token,int32_t position);
    Layer prepare_layer(std::size_t layer,int32_t position);
    void prefetch_next(std::size_t layer,std::size_t protected_layer);
    void finish_step();
    void reset(bool discard_prefix=false);
    cudaStream_t stream() const noexcept;
    PagedModelStats stats() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
