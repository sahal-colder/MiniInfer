#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/cpu_thread_cache.h"

#include <algorithm>
#include <cstdio>
#include <list>
#include <mutex>
#include <new>
#include <stdexcept>
#include <sys/mman.h>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace infer {
using namespace cpu_detail;

class CachingCPUDeviceAllocator::Impl : public std::enable_shared_from_this<Impl> {
    struct Entry {
        Impl* identity;
        std::weak_ptr<Impl> pool;
        std::shared_ptr<ThreadCache> cache;
        Entry(Impl* p, std::weak_ptr<Impl> w, std::shared_ptr<ThreadCache> c)
            : identity(p), pool(std::move(w)), cache(std::move(c)) {}
        ~Entry() {
            if (auto p = pool.lock())
                p->retire(cache.get());
        }
    };
    struct Registry {
        std::list<Entry> entries;
        Registry() noexcept { registry_pointer_ = this; }
        // 此后才析构的其他 thread_local 对象仍可能归还 Buffer；那时只走中央池。
        ~Registry() {
            tls_exiting_ = true;
            registry_pointer_ = nullptr;
        }
    };
    static thread_local bool tls_exiting_;
    static thread_local Registry* registry_pointer_;
    static thread_local Registry registry_;

    // 中央 mutex 串行化维护者；冻结只覆盖短元数据操作，不等待用户的 CPU/GPU 工作。
    class FrozenAll {
      public:
        explicit FrozenAll(const std::vector<std::shared_ptr<ThreadCache>>& caches)
            : caches_(caches) {
            for (auto& c : caches_)
                c->freeze();
        }
        ~FrozenAll() {
            for (auto& c : caches_)
                c->thaw();
        }
        FrozenAll(const FrozenAll&) = delete;
        FrozenAll& operator=(const FrozenAll&) = delete;

      private:
        const std::vector<std::shared_ptr<ThreadCache>>& caches_;
    };

  public:
    Impl(std::shared_ptr<DeviceAllocator> upstream, CachingCPUAllocatorConfig config)
        : config_(config), upstream_(std::move(upstream)),
          cpu_(dynamic_cast<CPUDeviceAllocator*>(upstream_.get())),
          pinned_(dynamic_cast<PinnedHostAllocator*>(upstream_.get())) {
        if (!upstream_ || upstream_->device_type() != DeviceType::CPU)
            throw std::invalid_argument("CPU pool needs a CPU upstream allocator.");
        if (config_.mapped_block_bytes &&
            (config_.enable_thread_cache || !cpu_ ||
             config_.mapped_block_bytes < 256 * 1024 || config_.mapped_block_bytes % 4096 ||
             config_.mapped_block_bytes != capacities[size_class(config_.mapped_block_bytes)]))
            throw std::invalid_argument("Mapped batches require a central pageable size class");
        if (config_.enable_thread_cache &&
            (!config_.thread_cache_batch_size ||
             config_.thread_cache_batch_size > config_.thread_cache_blocks_per_bin ||
             config_.thread_cache_blocks_per_bin > ThreadCache::max_owned_blocks))
            throw std::invalid_argument(
                "CPU thread cache requires 1 <= batch <= per-bin limit <= 256.");
    }
    ~Impl() {
        // 与其他 allocator 一样，不允许析构与成员调用并发；Buffer 会延长公有池寿命。
        // TLS Entry 的 weak_ptr 此时已过期；其索引只会被销毁，不再解引用 Block。
        for (const auto& entry : blocks_) {
            try {
                release_upstream(entry.first, entry.second.capacity);
            } catch (...) {
                std::fputs("CPU pool: upstream release failed during destruction.\n", stderr);
            }
        }
    }

    void* allocate(std::size_t bytes, bool aligned_io = false) {
        const auto bin = size_class(bytes);
        ThreadCache* cache = bin < small_bin_count ? find_cache() : nullptr;
        {
            LocalAccess access(cache);
            if (access) {
                const auto available = suitable(cache->free, bin, aligned_io);
                if (available != bin_count) {
                    Block* block = cache->free.pop(available);
                    activate(block);
                    ++cache->reuse_count;
                    ++cache->allocations;
                    return block->ptr;
                }
            }
        } // 在等待中央锁以前退出本地操作，避免 maintenance 与 owner 相互等待。
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.central_allocations;
        if (!cache && bin < small_bin_count && config_.enable_thread_cache && !tls_exiting_)
            cache = register_cache();
        {
            FrozenCache frozen(cache);
            // 本地尝试失败可能只是 stats/trim 暂时冻结；拿锁后再查一次。
            const auto local = cache ? suitable(cache->free, bin, aligned_io) : bin_count;
            if (local != bin_count) {
                Block* block = cache->free.pop(local);
                activate(block);
                ++stats_.reuse_count;
                return block->ptr;
            }
            const auto available = suitable(free_, bin, aligned_io);
            if (available != bin_count) {
                Block* block = free_.pop(available);
                if (can_attach(cache, block)) {
                    move_to_thread(block, cache);
                    // 连同本次取用的块，至多从相同桶搬运 batch 个已有块。
                    for (std::size_t n = 1; n < config_.thread_cache_batch_size; ++n) {
                        Block* extra = free_.front(available);
                        if (!extra || !can_attach(cache, extra) ||
                            cache->free.count(available) >= config_.thread_cache_blocks_per_bin)
                            break;
                        free_.pop(available);
                        move_to_thread(extra, cache);
                        cache->free.push(extra);
                    }
                    ++stats_.thread_cache_refills;
                }
                activate(block);
                ++stats_.reuse_count;
                return block->ptr;
            }
        }

        const auto capacity = capacities[bin];
        if (capacity > config_.max_reserved_bytes)
            throw std::bad_alloc();
        if (stats_.reserved_bytes > config_.max_reserved_bytes - capacity) {
            FrozenAll frozen(caches_);
            drain_thread_caches();
            trim_central();
            if (stats_.reserved_bytes > config_.max_reserved_bytes - capacity)
                throw std::bad_alloc();
        }
        if (capacity > std::numeric_limits<std::size_t>::max() - stats_.reserved_bytes)
            throw std::overflow_error("CPU pool reserved size overflow.");
        if (capacity == config_.mapped_block_bytes) {
            try {
                return allocate_mapped(bin, capacity);
            } catch (const std::bad_alloc&) {
                if (!cached_bytes_.load(std::memory_order_relaxed))
                    throw;
                {
                    FrozenAll frozen(caches_);
                    drain_thread_caches();
                    trim_central();
                }
                return allocate_mapped(bin, capacity);
            }
        }
        // 先备好哈希节点和桶容量，再申请上游内存。节点提取后不会收缩桶数组；
        // 恢复同一个节点时大小不超过刚才 emplace 的大小，无 rehash/元数据分配。
        // 这样避免了 cudaMallocHost 成功后登记失败，再遇 cudaFreeHost 失败的双重故障。
        auto node = blocks_.extract(blocks_.emplace(nullptr, Block{}).first);
        void* ptr;
        try {
            ptr = upstream_->allocate(capacity);
        } catch (const std::bad_alloc&) {
            if (!cached_bytes_.load(std::memory_order_relaxed))
                throw;
            {
                FrozenAll frozen(caches_);
                drain_thread_caches();
                trim_central();
            }
            ptr = upstream_->allocate(capacity); // 只在确有空闲缓存时 trim 后重试一次。
        }
        // 上游策略遵守 DeviceAllocator 合约：成功返回独占、对齐、非空地址。
        node.key() = ptr;
        node.mapped() = Block{ptr, capacity, bin, true, nullptr, nullptr, nullptr};
        Block* block = &blocks_.insert(std::move(node)).position->second;
        ++stats_.upstream_allocations;
        stats_.reserved_bytes += capacity;
        usage_.add(*block);
        FrozenCache frozen(cache);
        if (can_attach(cache, block))
            move_to_thread(block, cache);
        return ptr;
    }

    void* allocate_io(std::size_t bytes) {
        if (!cpu_ && !pinned_)
            throw std::logic_error("Aligned I/O allocation requires a built-in CPU upstream");
        return allocate(io_allocation_size(bytes), true);
    }

    void release(void* ptr) {
        if (!ptr)
            return;
        ThreadCache* cache = find_cache();
        {
            LocalAccess access(cache);
            if (access) {
                if (Block* block = cache->find(ptr)) {
                    check_active(block);
                    if (cache->free.count(block->bin) < config_.thread_cache_blocks_per_bin &&
                        reserve_cache(block->capacity)) {
                        cache_block(block);
                        ++cache->releases;
                        return;
                    }
                }
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.central_releases;
        const auto it = blocks_.find(ptr);
        if (it == blocks_.end())
            throw std::invalid_argument("CPU pool: pointer is not owned by this pool.");
        Block* block = &it->second;
        // owner 只会在中央锁下变化；获得它的 gate 后才访问 active/链表/计数。
        cache = block->owner;
        FrozenCache frozen(cache);
        check_active(block);
        if (reserve_cache(block->capacity)) {
            if (cache && cache->free.count(block->bin) >= config_.thread_cache_blocks_per_bin) {
                spill(cache, block->bin, config_.thread_cache_batch_size);
                ++stats_.thread_cache_spills;
            }
            cache_block(block);
        } else {
            // 先释放成功，再脱索引和改账；上游抛错时仍能用原地址重试 release。
            release_upstream(ptr, block->capacity);
            usage(block).remove(*block);
            if (cache)
                cache->detach(block);
            stats_.reserved_bytes -= block->capacity;
            blocks_.erase(it);
        }
    }

    CPUMemoryPoolStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        FrozenAll frozen(caches_);
        auto result = stats_;
        auto add = [&](const Usage& u) {
            result.active_bytes += u.active_bytes;
            result.cached_bytes += u.cached_bytes;
            result.active_blocks += u.active_blocks;
            result.cached_blocks += u.cached_blocks;
        };
        add(usage_);
        for (auto& cache : caches_) {
            add(cache->usage);
            result.thread_cached_bytes += cache->usage.cached_bytes;
            result.thread_cached_blocks += cache->usage.cached_blocks;
            result.reuse_count += cache->reuse_count;
            result.thread_cache_allocations += cache->allocations;
            result.thread_cache_releases += cache->releases;
        }
        return result;
    }
    void trim() {
        std::lock_guard<std::mutex> lock(mutex_);
        FrozenAll frozen(caches_);
        drain_thread_caches();
        trim_central();
    }

  private:
    ThreadCache* find_cache() const noexcept {
        // release 的首次跨线程调用也不能初始化有析构器的 TLS 容器；某些运行库
        // 注册 TLS 析构器会 malloc。只有 allocate 的慢路径显式创建 Registry。
        if (!config_.enable_thread_cache || !registry_pointer_)
            return nullptr;
        for (const auto& entry : registry_pointer_->entries)
            if (entry.identity == this && !entry.pool.expired())
                return entry.cache.get();
        return nullptr;
    }
    ThreadCache* register_cache() {
        // 调用者持中央锁。清除过期槽，不保留死池元数据；弱引用防止地址复用 ABA。
        auto& entries = registry_.entries;
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->pool.expired())
                it = entries.erase(it);
            else
                ++it;
        }
        auto cache = std::make_shared<ThreadCache>();
        caches_.push_back(cache);
        try {
            entries.emplace_back(this, weak_from_this(), cache);
        } catch (...) {
            caches_.pop_back();
            throw;
        }
        return cache.get();
    }
    void retire(ThreadCache* cache) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        FrozenCache frozen(cache);
        for (std::size_t bin = 0; bin < small_bin_count; ++bin)
            spill(cache, bin, cache->free.count(bin));
        // 线程退出时 active 块可能已经交给其他线程，所有权索引必须迁回中央池。
        for (auto& head : cache->index) {
            while (head)
                move_to_central(head);
        }
        stats_.reuse_count += cache->reuse_count;
        stats_.thread_cache_allocations += cache->allocations;
        stats_.thread_cache_releases += cache->releases;
        auto it = std::find_if(caches_.begin(), caches_.end(),
                               [&](const auto& c) { return c.get() == cache; });
        caches_.erase(it); // TLS Entry 仍持 shared_ptr，FrozenCache 析构时 cache 尚存活。
    }
    static void check_active(const Block* block) {
        if (!block->active)
            throw std::invalid_argument("CPU pool: block already released.");
    }
    std::size_t suitable(const FreeBins& bins, std::size_t bin, bool aligned_io) const noexcept {
        auto available = bins.first_nonempty(bin);
        while (aligned_io && available != bin_count && capacities[available] % io_alignment)
            available = bins.first_nonempty(available + 1);
        if (available == bin_count)
            return bin_count;
        const auto waste = capacities[available] - capacities[bin];
        return waste <= config_.max_reuse_waste && waste <= capacities[bin] / 256 ? available
                                                                                : bin_count;
    }
    Usage& usage(Block* block) noexcept { return block->owner ? block->owner->usage : usage_; }
    bool can_attach(const ThreadCache* cache, const Block* block) const noexcept {
        return cache && block->bin < small_bin_count &&
               cache->owned_blocks < ThreadCache::max_owned_blocks;
    }
    void move_to_thread(Block* block, ThreadCache* cache) noexcept {
        usage_.remove(*block);
        cache->attach(block);
        cache->usage.add(*block);
    }
    void move_to_central(Block* block) noexcept {
        auto* cache = block->owner;
        cache->usage.remove(*block);
        cache->detach(block);
        usage_.add(*block);
    }
    bool reserve_cache(std::size_t bytes) noexcept {
        // 全局硬上限：仅用原子容量预订，不拿中央锁；并发快路径不会各自超额缓存。
        auto cached = cached_bytes_.load(std::memory_order_relaxed);
        while (bytes <= config_.max_cached_bytes - cached) {
            if (cached_bytes_.compare_exchange_weak(cached, cached + bytes,
                                                    std::memory_order_relaxed))
                return true;
        }
        return false;
    }
    void activate(Block* block) noexcept {
        auto& u = usage(block);
        u.remove(*block);
        block->active = true;
        u.add(*block);
        cached_bytes_.fetch_sub(block->capacity, std::memory_order_relaxed);
    }
    void cache_block(Block* block) noexcept {
        auto& u = usage(block);
        u.remove(*block);
        block->active = false;
        u.add(*block);
        (block->owner ? block->owner->free : free_).push(block);
    }
    void spill(ThreadCache* cache, std::size_t bin, std::size_t count) noexcept {
        while (count--) {
            Block* block = cache->free.pop(bin);
            move_to_central(block);
            free_.push(block);
        }
    }
    void drain_thread_caches() noexcept {
        for (auto& cache : caches_)
            for (std::size_t bin = 0; bin < small_bin_count; ++bin)
                spill(cache.get(), bin, cache->free.count(bin));
    }
    void trim_central() {
        for (std::size_t bin = 0; bin < bin_count; ++bin) {
            while (Block* block = free_.front(bin)) {
                void* ptr = block->ptr;
                release_upstream(ptr, block->capacity); // On failure the block stays indexed.
                free_.pop(bin);
                usage_.remove(*block);
                cached_bytes_.fetch_sub(block->capacity, std::memory_order_relaxed);
                stats_.reserved_bytes -= block->capacity;
                blocks_.erase(ptr);
            }
        }
    }

    void* allocate_mapped(std::size_t bin, std::size_t capacity) {
        constexpr std::size_t alignment = 2 * 1024 * 1024;
        const auto count = std::min(
            {std::size_t(8), 1 + (alignment - 1) / capacity,
             (config_.max_reserved_bytes - stats_.reserved_bytes) / capacity,
             1 + (config_.max_cached_bytes - cached_bytes_.load(std::memory_order_relaxed)) /
                     capacity});
        // Prepare all metadata before mmap; node insertion then cannot allocate.
        blocks_.reserve(blocks_.size() + count);
        std::array<decltype(blocks_)::node_type, 8> nodes;
        for (std::size_t i = 0; i < count; ++i)
            nodes[i] = blocks_.extract(blocks_.emplace(nullptr, Block{}).first);
        const auto bytes = count * capacity;
        const auto padding = bytes >= alignment ? alignment : 0;
        if (bytes > SIZE_MAX - padding)
            throw std::bad_alloc();
        auto* mapping = mmap(nullptr, bytes + padding, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED)
            throw std::bad_alloc();
        const auto address = reinterpret_cast<std::uintptr_t>(mapping);
        const auto aligned = padding ? (address + alignment - 1) & ~(alignment - 1) : address;
        auto* data = reinterpret_cast<unsigned char*>(aligned);
        const auto prefix = aligned - address;
        const auto suffix = padding - prefix;
        if (prefix && munmap(mapping, prefix)) {
            munmap(mapping, bytes + padding);
            throw std::bad_alloc();
        }
        if (suffix && munmap(data + bytes, suffix)) {
            munmap(data, bytes + suffix);
            throw std::bad_alloc();
        }
#ifdef MADV_HUGEPAGE
        if (bytes >= alignment)
            (void)madvise(data, bytes, MADV_HUGEPAGE);
#endif
        // Central-only mode keeps cache headroom stable while this lock is held.
        (void)reserve_cache((count - 1) * capacity);
        for (std::size_t i = 0; i < count; ++i) {
            nodes[i].key() = data + i * capacity;
            nodes[i].mapped() =
                Block{data + i * capacity, capacity, bin, i == 0, nullptr, nullptr, nullptr};
            auto* block = &blocks_.insert(std::move(nodes[i])).position->second;
            usage_.add(*block);
            if (i)
                free_.push(block);
        }
        stats_.reserved_bytes += bytes;
        stats_.mapped_reserved_bytes += bytes;
        ++stats_.mapped_allocations;
        return data;
    }
    void release_upstream(void* ptr, std::size_t capacity) {
        if (capacity == config_.mapped_block_bytes) {
            // Slices are page aligned and independently owned, including after
            // neighboring slices have been trimmed from the original mapping.
            if (munmap(ptr, capacity))
                throw std::system_error(errno, std::generic_category(), "CPU pool munmap");
            stats_.mapped_reserved_bytes -= capacity;
            return;
        }
        upstream_->release(ptr);
        ++stats_.upstream_releases;
    }
    const CachingCPUAllocatorConfig config_;
    const std::shared_ptr<DeviceAllocator> upstream_;
    CPUDeviceAllocator* const cpu_;
    PinnedHostAllocator* const pinned_;
    mutable std::mutex mutex_;
    std::unordered_map<void*, Block> blocks_;
    FreeBins free_;
    Usage usage_;
    CPUMemoryPoolStats stats_;
    std::vector<std::shared_ptr<ThreadCache>> caches_;
    std::atomic<std::size_t> cached_bytes_{0};
};

thread_local bool CachingCPUDeviceAllocator::Impl::tls_exiting_ = false;
thread_local CachingCPUDeviceAllocator::Impl::Registry*
    CachingCPUDeviceAllocator::Impl::registry_pointer_ = nullptr;
thread_local CachingCPUDeviceAllocator::Impl::Registry CachingCPUDeviceAllocator::Impl::registry_;

CachingCPUDeviceAllocator::CachingCPUDeviceAllocator(CachingCPUAllocatorConfig config)
    : CachingCPUDeviceAllocator(std::make_shared<CPUDeviceAllocator>(), config) {}
CachingCPUDeviceAllocator::CachingCPUDeviceAllocator(std::shared_ptr<DeviceAllocator> upstream,
                                                     CachingCPUAllocatorConfig config)
    : DeviceAllocator(DeviceType::CPU), impl_(std::make_shared<Impl>(std::move(upstream), config)) {
}
CachingCPUDeviceAllocator::~CachingCPUDeviceAllocator() = default;
void* CachingCPUDeviceAllocator::allocate(std::size_t size, cudaStream_t /*stream*/) const {
    return impl_->allocate(size);
}
void* CachingCPUDeviceAllocator::allocate_io(std::size_t size) const {
    return impl_->allocate_io(size);
}
void CachingCPUDeviceAllocator::release(void* ptr, cudaStream_t /*stream*/) const {
    impl_->release(ptr);
}
CPUMemoryPoolStats CachingCPUDeviceAllocator::stats() const { return impl_->stats(); }
void CachingCPUDeviceAllocator::trim() const { impl_->trim(); }

} // namespace infer
