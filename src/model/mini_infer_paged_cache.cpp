#include "mini_infer/model/mini_infer_paged_cache.h"
#include "mini_infer/base/checked_math.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
namespace mini_infer {
namespace {
using Location=infer::StorageLocation;
std::size_t multiply(std::size_t a,std::size_t b) { return detail::checked_mul(a,b,"Paged model capacity overflow"); }
std::size_t add(std::size_t a,std::size_t b) { return detail::checked_add(a,b,"Paged model capacity overflow"); }
}
struct PagedModelCache::Impl {
    ModelConfig model;
    PagedModelConfig options;
    std::size_t kv_dim=0,tokens=0,max_pages=0;
    infer::StorageConfig config;
    std::unique_ptr<infer::PagedKV> cache;
    infer::PagedKV::Sequence sequence;
    std::vector<int32_t> history;
    std::vector<std::uint64_t> used;
    std::uint64_t clock=0;
    std::size_t published=0;
    bool prefix_dirty=false,prefix_clean=true;
    cudaStream_t compute=nullptr;
    void* host_table=nullptr;
    Tensor device_table;
    struct Pending { std::size_t layer;std::deque<infer::PagedKV::Ticket> jobs; };
    std::optional<Pending> pending;
    PagedModelStats counts;
    Impl(const ModelConfig& c,PagedModelConfig value,std::shared_ptr<DeviceAllocator> allocator)
        : model(c),options(std::move(value)),used(c.layer_num,0) {
        if(!allocator || allocator->device_type()!=DeviceType::CUDA || c.dim<=0 || c.seq_len<=0 ||
           c.layer_num<=0 || c.head_num<=0 || c.kv_head_num<=0 || c.dim%c.head_num || c.head_num%c.kv_head_num ||
           (options.page_bytes!=64*1024 && options.page_bytes!=256*1024))
            throw std::invalid_argument("Paged model needs CUDA, valid geometry and 64/256 KiB pages");
        kv_dim=multiply(c.kv_head_num,c.dim/c.head_num);
        const auto capacity=options.page_bytes/multiply(2* sizeof(float),kv_dim);
        tokens=options.tokens_per_page;
        if(!tokens) {
            tokens=1;
            while(tokens<=std::min<std::size_t>(16,capacity)/2) tokens*=2;
        }
        if(!capacity || !tokens || tokens>capacity || multiply(tokens,kv_dim)%4)
            throw std::invalid_argument("KV tokens/page must fit and preserve 16-byte row alignment");
        max_pages=1+(std::size_t(c.seq_len)-1)/tokens;
        config.block_bytes=options.page_bytes;
        check_cuda(cudaGetDevice(&config.gpu_device),"Paged model device");
        config.cpu_memory=infer::HostMemoryKind::PINNED;
        config.retain_ssd_copy=true;
        config.ssd_direct=options.ssd_direct && options.ssd_spdk.bdev_name.empty();
        config.ssd_io=options.ssd_io;
        config.ssd_spdk=options.ssd_spdk;
        config.ssd_verification=options.ssd_read_back ? infer::SSDVerification::READ_BACK : infer::SSDVerification::ON_READ;
        config.ssd_directory=options.ssd_directory;
        const auto layer=multiply(max_pages,options.page_bytes);
        config.budget.gpu_bytes=options.gpu_bytes ? options.gpu_bytes : multiply(options.resident_layers ? std::min<std::size_t>(c.layer_num,options.resident_layers) : std::size_t(c.layer_num),layer);
        config.budget.cpu_bytes=options.host_bytes ? options.host_bytes : multiply(add(c.layer_num,2*max_pages),options.page_bytes);
        config.budget.ssd_bytes=options.ssd_bytes ? options.ssd_bytes : multiply(multiply(c.layer_num,max_pages+1),options.page_bytes+4096);
        if(config.budget.gpu_bytes<options.page_bytes || config.budget.cpu_bytes<multiply(c.layer_num,options.page_bytes))
            throw std::invalid_argument("KV GPU budget needs one page; host budget needs one all-layer page group");
        cache=std::make_unique<infer::PagedKV>(config,c.layer_num,tokens,"llama-model-instance/f32/rope10000/causal");
        try {
            check_cuda(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking),"Paged compute stream");
            check_cuda(cudaHostAlloc(&host_table,multiply(max_pages,sizeof(void*)),cudaHostAllocPortable),"Paged pinned table");
            const auto table_bytes=multiply(max_pages,sizeof(void*));
            if(table_bytes>std::size_t(INT32_MAX)) throw std::length_error("KV page table exceeds Tensor dimensions");
            device_table=Tensor(DataType::INT8,{static_cast<int32_t>(table_bytes)},allocator);
        } catch(...) { cleanup();throw; }
    }
    void drain() {
        if(!pending) return;
        const auto start=std::chrono::steady_clock::now();
        std::exception_ptr error;
        while(!pending->jobs.empty()) {
            auto job=std::move(pending->jobs.front());pending->jobs.pop_front();
            try {job.wait();}catch(...) {if(!error) error=std::current_exception();}
        }
        pending.reset();
        counts.prefetch_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        if(error) std::rethrow_exception(error);
    }
    void cleanup() noexcept {
        try {drain();}catch(...) { pending.reset(); }
        if(compute) cudaStreamSynchronize(compute);
        if(host_table) {cudaFreeHost(host_table);host_table=nullptr;}
        if(compute) {cudaStreamDestroy(compute);compute=nullptr;}
    }
    ~Impl() { cleanup(); }
    std::vector<std::size_t> select(std::size_t layer,Location source) const {
        std::vector<std::size_t> result;
        for(std::size_t p=0;p<sequence.page_count();++p) {
            const auto meta=cache->metadata(sequence,p,layer);
            if(meta.location==source && meta.state==infer::BlockState::RESIDENT) result.push_back(p);
        }
        return result;
    }
    std::vector<std::size_t> missing(std::size_t layer) const {
        std::vector<std::size_t> result;
        for(std::size_t p=0;p<sequence.page_count();++p)
            if(cache->metadata(sequence,p,layer).location!=Location::GPU) result.push_back(p);
        return result;
    }
    void move(std::size_t layer,Location source,Location target) {
        const auto indices=select(layer,source);
        for(std::size_t first=0;first<indices.size();first+=64) {
            const std::vector<std::size_t> group(indices.begin()+first,indices.begin()+std::min(indices.size(),first+64));
            auto ticket=cache->prefetch_pages(sequence,layer,target,group);ticket.wait();
            if(source==Location::GPU) counts.gpu_evicted_pages+=group.size();
            if(target==Location::SSD) counts.ssd_spilled_pages+=group.size();
        }
    }
    void discard_inactive_prefixes() {
        if(prefix_clean) return;
        counts.prefix_evictions+=cache->evict_prefixes();prefix_dirty=true;prefix_clean=true;
    }
    std::vector<std::size_t> candidates(std::size_t exclude_a,std::size_t exclude_b) const {
        std::vector<std::size_t> result;
        for(std::size_t layer=0;layer<used.size();++layer)
            if(layer!=exclude_a && layer!=exclude_b) result.push_back(layer);
        std::stable_sort(result.begin(),result.end(),[&](auto a,auto b){return used[a]<used[b];});
        return result;
    }
    bool host_room(std::size_t bytes,std::size_t exclude_a,std::size_t exclude_b) {
        if(bytes>config.budget.cpu_bytes) return false;
        const auto fits=[&]{return cache->storage().stats(Location::CPU).reserved_bytes<=config.budget.cpu_bytes-bytes;};
        if(fits()) return true;
        discard_inactive_prefixes();
        for(auto layer:candidates(exclude_a,exclude_b)) {
            if(fits()) break;
            move(layer,Location::CPU,Location::SSD);
        }
        return fits();
    }
    bool gpu_room(std::size_t target,std::size_t protected_layer,bool optional) {
        const auto required=multiply(missing(target).size(),options.page_bytes);
        const auto full=multiply(sequence.page_count(),options.page_bytes);
        if(full>config.budget.gpu_bytes) {
            if(optional) return false;
            throw std::length_error("GPU KV budget cannot hold one layer of the current context");
        }
        if(optional && protected_layer<used.size()) {
            const auto protected_bytes=multiply(select(protected_layer,Location::GPU).size(),options.page_bytes);
            if(full>config.budget.gpu_bytes-protected_bytes) return false;
        }
        const auto fits=[&]{return cache->storage().stats(Location::GPU).reserved_bytes<=config.budget.gpu_bytes-required;};
        if(fits()) return true;
        discard_inactive_prefixes();
        for(auto victim:candidates(target,protected_layer)) {
            if(fits()) break;
            const auto bytes=multiply(select(victim,Location::GPU).size(),options.page_bytes);
            if(!bytes) continue;
            const bool in_host=host_room(bytes,target,protected_layer);
            move(victim,Location::GPU,in_host ? Location::CPU : Location::SSD);
        }
        return fits();
    }
    void submit_layer(std::size_t layer,bool ahead) {
        auto indices=missing(layer);
        if(ahead) counts.prefetched_pages+=indices.size();else counts.demand_fault_pages+=indices.size();
        pending.emplace(Pending{layer,{}});
        try {
            for(std::size_t first=0;first<indices.size();first+=64) {
                if(pending->jobs.size()==config.max_async_migrations) {
                    pending->jobs.front().wait();pending->jobs.pop_front();
                }
                const std::vector<std::size_t> group(indices.begin()+first,indices.begin()+std::min(indices.size(),first+64));
                pending->jobs.push_back(cache->prefetch_pages(sequence,layer,Location::GPU,group));
                if(ahead) ++counts.prefetch_windows;
            }
        } catch(...) {try{drain();}catch(...){}throw;}
        if(!ahead) drain();
    }
};
PagedModelCache::PagedModelCache(const ModelConfig& c,PagedModelConfig options,std::shared_ptr<DeviceAllocator> allocator)
    : impl_(std::make_unique<Impl>(c,std::move(options),std::move(allocator))) {}
PagedModelCache::~PagedModelCache()=default;
std::size_t PagedModelCache::begin_request(const int32_t* input,std::size_t count) {
    auto& s=*impl_;
    if(!s.history.empty()) throw std::logic_error("Paged request was not reset");
    s.drain();s.prefix_clean=false;
    s.sequence=s.options.prefix_cache ? s.cache->match(std::vector<int32_t>(input,input+count-1)) : infer::PagedKV::Sequence{};
    s.history.assign(input,input+s.sequence.matched_tokens);
    s.counts.last_prefix_hit_tokens=s.sequence.matched_tokens;
    s.counts.prefix_hit_tokens+=s.sequence.matched_tokens;
    s.published=s.sequence.page_count();
    return s.sequence.matched_tokens;
}
void PagedModelCache::begin_token(int32_t token,int32_t position) {
    auto& s=*impl_;
    if(position<0 || std::size_t(position)!=s.history.size()) throw std::logic_error("Non-contiguous paged token admission");
    if(std::size_t(position)%s.tokens==0) {
        s.drain();
        if(!s.host_room(multiply(s.model.layer_num,s.options.page_bytes),s.used.size(),s.used.size()))
            throw std::bad_alloc();
        s.cache->append_page(s.sequence,Location::CPU);
        const auto page=s.sequence.page_count()-1;
        for(int layer=0;layer<s.model.layer_num;++layer) {
            auto lease=s.cache->acquire(s.sequence,page,layer,infer::AccessMode::WRITE);
            std::memset(lease.writable_data(),0,lease.bytes());lease.finish();
        }
    }
    s.history.push_back(token);++s.counts.computed_tokens;
}
PagedModelCache::Layer PagedModelCache::prepare_layer(std::size_t layer,int32_t position) {
    auto& s=*impl_;
    s.drain();
    if(!s.gpu_room(layer,s.used.size(),false)) throw std::bad_alloc();
    if(!s.missing(layer).empty()) s.submit_layer(layer,false);
    s.used.at(layer)=++s.clock;++s.counts.layer_calls;
    Layer result;result.page_count=s.sequence.page_count();result.tokens_per_page=s.tokens;
    result.leases.reserve(result.page_count);
    auto** host=static_cast<const float**>(s.host_table);
    for(std::size_t page=0;page<result.page_count;++page) {
        const bool current=page==std::size_t(position)/s.tokens;
        result.leases.push_back(s.cache->acquire(s.sequence,page,layer,
            current ? infer::AccessMode::WRITE : infer::AccessMode::READ,s.compute));
        host[page]=static_cast<const float*>(result.leases.back().data());
        if(current) {
            auto* base=static_cast<float*>(result.leases.back().writable_data());
            const auto row=std::size_t(position)%s.tokens*s.kv_dim;
            result.key=Tensor::borrow(DataType::FLOAT32,{static_cast<int32_t>(s.kv_dim)},base+row,DeviceType::CUDA);
            result.value=Tensor::borrow(DataType::FLOAT32,{static_cast<int32_t>(s.kv_dim)},base+s.tokens*s.kv_dim+row,DeviceType::CUDA);
        }
    }
    check_cuda(cudaMemcpyAsync(s.device_table.ptr<int8_t>(),s.host_table,result.page_count*sizeof(void*),cudaMemcpyHostToDevice,s.compute),"Paged table upload");
    result.pages=reinterpret_cast<const float* const*>(s.device_table.ptr<int8_t>());
    return result;
}
void PagedModelCache::prefetch_next(std::size_t layer,std::size_t protected_layer) {
    auto& s=*impl_;
    if(!s.options.prefetch || layer>=s.used.size()) return;
    if(s.pending) throw std::logic_error("Previous layer prefetch was not consumed");
    if(s.missing(layer).empty()) return;
    if(!s.gpu_room(layer,protected_layer,true)) {++s.counts.prefetch_skipped;return;}
    s.submit_layer(layer,true);
}
void PagedModelCache::finish_step() {
    auto& s=*impl_;s.drain();
    const auto complete=s.history.size()/s.tokens;
    if(s.options.prefix_cache && (complete>s.published || s.prefix_dirty)) {
        s.cache->publish_prefix(s.sequence,s.history,complete);s.published=complete;s.prefix_dirty=false;
    }
}
void PagedModelCache::reset(bool discard_prefix) {
    auto& s=*impl_;
    std::exception_ptr error;
    try {s.drain();}catch(...) {error=std::current_exception();}
    const auto completed=cudaStreamSynchronize(s.compute);
    s.sequence=infer::PagedKV::Sequence{};s.history.clear();s.published=0;s.counts.last_prefix_hit_tokens=0;
    if(discard_prefix || error) s.cache->evict_prefixes();
    check_cuda(completed,"Reset paged consumers");
    if(error) std::rethrow_exception(error);
}
cudaStream_t PagedModelCache::stream() const noexcept {return impl_->compute;}
PagedModelStats PagedModelCache::stats() const {
    const auto& s=*impl_;auto result=s.counts;
    const auto gpu=s.cache->storage().stats(Location::GPU),host=s.cache->storage().stats(Location::CPU),disk=s.cache->storage().stats(Location::SSD);
    const auto io=s.cache->storage().io_stats(Location::SSD);
    result.page_bytes=s.options.page_bytes;result.tokens_per_page=s.tokens;result.pages=s.sequence.page_count();
    result.gpu_limit_bytes=gpu.limit_bytes;result.host_limit_bytes=host.limit_bytes;result.ssd_limit_bytes=disk.limit_bytes;
    result.gpu_reserved_bytes=gpu.reserved_bytes;result.host_reserved_bytes=host.reserved_bytes;result.ssd_reserved_bytes=disk.reserved_bytes;
    result.gpu_peak_bytes=gpu.peak_reserved_bytes;result.host_peak_bytes=host.peak_reserved_bytes;result.ssd_peak_bytes=disk.peak_reserved_bytes;
    result.staging_bytes=s.cache->storage().staging_stats().reserved_bytes+s.max_pages*sizeof(void*)+4096;
    result.ssd_read_bytes=io.payload_read_bytes;result.ssd_written_bytes=io.payload_written_bytes;result.ssd_record_writes=io.record_writes;
    result.uring_submissions=io.file_io.submissions;result.ssd_device=io.device;return result;
}
}
