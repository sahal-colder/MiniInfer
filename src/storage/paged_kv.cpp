#include "infer/storage/paged_kv.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
namespace infer {
struct PagedKV::Page {
    std::shared_ptr<TieredStorage> storage;
    std::vector<BlockId> layers;
    bool sealed = false;
    std::vector<std::int32_t> tokens;
    ~Page() {
        for (auto id : layers) {
            try {
                if (!storage->erase(id)) std::fputs("Paged KV retained: page is busy or completion unknown\n",stderr);
            } catch (const std::exception& e) { std::fprintf(stderr,"Paged KV cleanup retained: %s\n",e.what()); }
        }
    }
};
struct PagedKV::Prefix {
    struct Span {
        const std::int32_t* pointer; std::size_t count;
        const std::int32_t* data() const { return pointer; }
        std::size_t size() const { return count; }
    };
    struct Less {
        using is_transparent=void;
        template<class A,class B> bool operator()(const A& a,const B& b) const {
            return std::lexicographical_compare(a.data(),a.data()+a.size(),b.data(),b.data()+b.size());
        }
    };
    struct Node {
        std::shared_ptr<Page> page;
        std::map<std::vector<std::int32_t>, std::unique_ptr<Node>,Less> children;
    };
    Node root;
    mutable std::mutex mutex; // Prefix/control plane only; kernels hold PageLease addresses.
};
PagedKV::PageLease::PageLease(std::shared_ptr<Page> page, BlockLease lease)
    : owner_(std::move(page)), lease_(std::move(lease)) {}
PagedKV::Ticket::Ticket(std::vector<std::shared_ptr<Page>> owners,
                        std::future<std::vector<MigrationResult>> future)
    : owners_(std::move(owners)), future_(std::move(future)) {}
PagedKV::Ticket::~Ticket() { if (future_.valid()) future_.wait(); }
void PagedKV::Ticket::wait() {
    if (!future_.valid()) return;
    const auto results = future_.get();
    for (const auto& result : results) {
        if (result.error) std::rethrow_exception(result.error);
        if (!result.committed) throw StorageBusy("Paged KV migration did not commit");
    }
}
PagedKV::PagedKV(StorageConfig config,std::size_t layers,std::size_t tokens,std::string cache_namespace)
    : layers_(layers), tokens_(tokens), bytes_(config.block_bytes), namespace_(std::move(cache_namespace)),
      prefix_(std::make_unique<Prefix>()) {
    if (!layers || !tokens || namespace_.empty() || (bytes_ != 64*1024 && bytes_ != 256*1024))
        throw std::invalid_argument("Paged KV needs a namespace, layers, tokens and 64/256 KiB blocks");
    config.cpu_memory=HostMemoryKind::PINNED;
    storage_=std::make_shared<TieredStorage>(std::move(config));
}
PagedKV::~PagedKV() = default;
PagedKV::Sequence PagedKV::match(const std::vector<std::int32_t>& tokens) const {
    Sequence sequence;
    sequence.pages.reserve(tokens.size()/tokens_);
    std::lock_guard<std::mutex> lock(prefix_->mutex);
    const auto* node=&prefix_->root;
    for (std::size_t start=0; tokens.size()-start>=tokens_; start+=tokens_) {
        // ALLOCATION_FREE_PREFIX_LOOKUP: transparent comparison borrows input
        // tokens; no temporary vector allocation for every radix edge.
        const auto found=node->children.find(Prefix::Span{tokens.data()+start,tokens_});
        if (found==node->children.end()) break;
        node=found->second.get();
        // ZERO_COPY_PREFIX: clone ownership and the block table, never KV bytes.
        sequence.pages.push_back(node->page);
        sequence.matched_tokens+=tokens_;
    }
    return sequence;
}
void PagedKV::append_page(Sequence& sequence,StorageLocation location) {
    if (location!=StorageLocation::GPU && location!=StorageLocation::CPU)
        throw std::invalid_argument("New KV pages require CPU/GPU memory");
    auto page=std::make_shared<Page>(); page->storage=storage_;
    page->layers.reserve(layers_);
    for (std::size_t layer=0;layer<layers_;++layer) page->layers.push_back(storage_->create(bytes_,location));
    sequence.pages.push_back(std::move(page));
}
PagedKV::PageLease PagedKV::acquire(const Sequence& sequence,std::size_t index,std::size_t layer,
                                  AccessMode mode,cudaStream_t stream) {
    auto page=sequence.pages.at(index);
    if (page->storage.get()!=storage_.get()) throw std::invalid_argument("KV page belongs to another namespace");
    if (mode==AccessMode::WRITE && page->sealed) throw std::logic_error("Shared prefix pages are immutable");
    return PageLease(page,storage_->acquire(page->layers.at(layer),mode,stream));
}
void PagedKV::publish(const Sequence& sequence,const std::vector<std::int32_t>& tokens) {
    publish_prefix(sequence,tokens,sequence.pages.size());
}
BlockMeta PagedKV::metadata(const Sequence& sequence,std::size_t index,std::size_t layer) const {
    const auto& page=sequence.pages.at(index);
    if (page->storage.get()!=storage_.get()) throw std::invalid_argument("KV namespace mismatch");
    return storage_->metadata(page->layers.at(layer));
}
void PagedKV::finish_many(std::vector<PageLease>& leases) {
    std::vector<BlockLease*> group;group.reserve(leases.size());
    for (auto& lease:leases) group.push_back(&lease.lease_);
    BlockLease::finish_group(group);
}
void PagedKV::publish_prefix(const Sequence& sequence,const std::vector<std::int32_t>& tokens,std::size_t count) {
    if (count>sequence.pages.size() || count>tokens.size()/tokens_)
        throw std::invalid_argument("Only complete token pages may be published");
    // Single scheduler owns mutable Sequence/Page state. Prefix metadata lock
    // does not span any CUDA/IO wait or kernel execution.
    for (std::size_t i=0;i<count;++i) {
        const auto& page=sequence.pages[i];
        if (page->storage.get()!=storage_.get()) throw std::invalid_argument("KV namespace mismatch");
        if (page->sealed && !std::equal(page->tokens.begin(),page->tokens.end(),tokens.begin()+i*tokens_))
            throw std::invalid_argument("Published KV pages cannot be relabeled with different tokens");
        for (auto id:page->layers) {
            const auto meta=storage_->metadata(id);
            if (!meta.version || meta.lease_count || meta.state!=BlockState::RESIDENT)
                throw StorageBusy("KV page has uncommitted/in-flight layers");
        }
    }
    std::lock_guard<std::mutex> lock(prefix_->mutex);
    auto* node=&prefix_->root;
    for (std::size_t i=0;i<count;++i) {
        std::vector<std::int32_t> edge(tokens.begin()+i*tokens_,tokens.begin()+(i+1)*tokens_);
        sequence.pages[i]->tokens=edge;
        sequence.pages[i]->sealed=true;
        auto found=node->children.find(edge);
        if (found==node->children.end()) {
            auto child=std::make_unique<Prefix::Node>(); child->page=sequence.pages[i];
            found=node->children.emplace(std::move(edge),std::move(child)).first;
        }
        node=found->second.get();
    }
}
PagedKV::Ticket PagedKV::prefetch_layer(const Sequence& sequence,std::size_t layer,StorageLocation target,
                                      std::size_t first,std::size_t count) {
    if (layer>=layers_ || first>sequence.pages.size() || count>64)
        throw std::out_of_range("KV prefetch layer/window");
    std::vector<std::size_t> indices;
    const auto end=first+std::min(count,sequence.pages.size()-first);
    for(auto i=first;i<end;++i) indices.push_back(i);
    return prefetch_pages(sequence,layer,target,indices);
}
PagedKV::Ticket PagedKV::prefetch_pages(const Sequence& sequence,std::size_t layer,StorageLocation target,
                                      const std::vector<std::size_t>& indices) {
    if(layer>=layers_ || indices.size()>64) throw std::out_of_range("KV migration layer/batch");
    std::vector<BlockId> ids;
    std::vector<std::shared_ptr<Page>> owners;
    for(std::size_t i=0;i<indices.size();++i) {
        if(i && indices[i]<=indices[i-1]) throw std::invalid_argument("KV page indices must increase");
        const auto& page=sequence.pages.at(indices[i]);
        if(page->storage.get()!=storage_.get()) throw std::invalid_argument("KV namespace mismatch");
        const auto meta=storage_->metadata(page->layers[layer]);
        if(!meta.version) throw std::logic_error("Uninitialized KV layers cannot migrate");
        if(meta.state!=BlockState::RESIDENT) throw StorageBusy("KV prefetch source is in flight");
        if(meta.location==target) continue;
        ids.push_back(page->layers[layer]);owners.push_back(page);
    }
    if(ids.empty()) return Ticket({},{});
    return Ticket(std::move(owners),storage_->migrate_batch_async(ids,target));
}
std::size_t PagedKV::evict_prefixes() {
    decltype(prefix_->root.children) retired;
    {
        std::lock_guard<std::mutex> lock(prefix_->mutex);
        retired.swap(prefix_->root.children);
    }
    return retired.size(); // last owners reclaim outside the prefix lock
}
} // namespace infer
