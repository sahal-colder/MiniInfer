#pragma once
#include "infer/storage/tiered_storage.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace infer {
// One instance is one model/adapter/position+mask namespace. Only complete,
// immutable token pages enter the block-radix tree. Partial tails stay private.
// One scheduler owns mutations/acquire/migration admission; match may run
// concurrently under the short prefix-index lock. Tickets execute asynchronously.
class PagedKV {
    struct Page;
    struct Prefix;
  public:
    struct Sequence {
        Sequence() = default;
        Sequence(Sequence&&) noexcept = default;
        Sequence& operator=(Sequence&&) noexcept = default;
        Sequence(const Sequence&) = delete;
        Sequence& operator=(const Sequence&) = delete;
        std::size_t matched_tokens = 0;
        std::size_t page_count() const noexcept { return pages.size(); }
      private:
        friend class PagedKV;
        std::vector<std::shared_ptr<Page>> pages;
    };
    class PageLease {
        std::shared_ptr<Page> owner_; // destroyed after lease_, preserving the page
        BlockLease lease_;
        friend class PagedKV;
        PageLease(std::shared_ptr<Page>, BlockLease);
      public:
        PageLease(PageLease&&) noexcept = default;
        PageLease& operator=(PageLease&&) = delete;
        const void* data() const noexcept { return lease_.data(); }
        void* writable_data() const { return lease_.writable_data(); }
        std::size_t bytes() const noexcept { return lease_.size(); }
        void finish() { lease_.finish(); }
    };
    class Ticket {
        std::vector<std::shared_ptr<Page>> owners_;
        std::future<std::vector<MigrationResult>> future_;
        friend class PagedKV;
        Ticket(std::vector<std::shared_ptr<Page>>, std::future<std::vector<MigrationResult>>);
      public:
        Ticket(Ticket&&) noexcept = default;
        ~Ticket(); // drains before releasing the pages, even if caller drops it
        void wait();
    };
    PagedKV(StorageConfig, std::size_t layers, std::size_t tokens_per_page, std::string cache_namespace);
    ~PagedKV();
    PagedKV(const PagedKV&) = delete;
    PagedKV& operator=(const PagedKV&) = delete;
    Sequence match(const std::vector<std::int32_t>& tokens) const;
    void append_page(Sequence&, StorageLocation = StorageLocation::GPU);
    PageLease acquire(const Sequence&, std::size_t page, std::size_t layer,
                      AccessMode, cudaStream_t = nullptr);
    void publish(const Sequence&, const std::vector<std::int32_t>& tokens);
    void publish_prefix(const Sequence&, const std::vector<std::int32_t>& tokens,std::size_t complete_pages);
    BlockMeta metadata(const Sequence&,std::size_t page,std::size_t layer) const;
    // One authoritative completion wait, then commit/release distinct same-stream leases.
    static void finish_many(std::vector<PageLease>&);
    // Per-layer prefetch/offload. At most 64 pages per bounded migration job.
    // Aligned SSD backends coalesce adjacent physical records within that job.
    // Resident targets are skipped; a busy source raises StorageBusy.
    Ticket prefetch_layer(const Sequence&, std::size_t layer,
                          StorageLocation target = StorageLocation::GPU,
                          std::size_t first_page = 0, std::size_t max_pages = 64);
    Ticket prefetch_pages(const Sequence&,std::size_t layer,StorageLocation,
                          const std::vector<std::size_t>& ascending_pages);
    std::size_t evict_prefixes(); // active Sequence/PageLease/Ticket references survive
    TieredStorage& storage() noexcept { return *storage_; }
  private:
    std::shared_ptr<TieredStorage> storage_;
    std::size_t layers_, tokens_, bytes_;
    std::string namespace_;
    std::unique_ptr<Prefix> prefix_;
};
} // namespace infer
