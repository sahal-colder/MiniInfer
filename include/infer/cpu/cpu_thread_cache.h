#pragma once

#include "infer/cpu/cpu_block.h"
#include <atomic>

namespace infer::cpu_detail {

// 本线程通过 gate 独占访问；中央维护者先持池 mutex，再 freeze 等待本地访问退出。
struct ThreadCache {
    static constexpr std::size_t index_size = 64;
    // 上限约束本线程元数据及哈希冲突链规模，包含 active 与 cached 块。
    static constexpr std::size_t max_owned_blocks = 256;
    std::array<Block*, index_size> index{};
    std::size_t owned_blocks = 0;
    FreeBins free;
    Usage usage;
    std::size_t reuse_count = 0, allocations = 0, releases = 0;
    // 不等待中央维护：失败则退回中央慢路径。leave 保留维护者已设置的 bit1。
    bool try_enter() noexcept;
    void leave() noexcept;
    void freeze() noexcept;
    void thaw() noexcept;
    // 只比较地址，不解引用用户指针或其前缀。
    Block* find(void* ptr) const noexcept;
    // 只更新索引、owner 和 owned_blocks；Usage 由调用方转账。
    void attach(Block* block) noexcept;
    void detach(Block* block) noexcept;

  private:
    // bit0：本地访问中；bit1：中央维护已请求。独立缓存行避免与元数据伪共享。
    alignas(64) std::atomic<unsigned> gate_{0};
    static std::size_t hash(void* ptr) noexcept;
};

// 本线程快路径的非阻塞 gate 凭证。
class LocalAccess {
  public:
    explicit LocalAccess(ThreadCache* cache) noexcept
        : cache_(cache && cache->try_enter() ? cache : nullptr) {}
    ~LocalAccess() {
        if (cache_)
            cache_->leave();
    }
    explicit operator bool() const noexcept { return cache_ != nullptr; }
    LocalAccess(const LocalAccess&) = delete;
    LocalAccess& operator=(const LocalAccess&) = delete;

  private:
    ThreadCache* cache_;
};

// 调用方须持中央 mutex；同一 cache 不可嵌套，否则内层析构会提前解冻。
class FrozenCache {
  public:
    explicit FrozenCache(ThreadCache* cache) noexcept : cache_(cache) {
        if (cache_)
            cache_->freeze();
    }
    ~FrozenCache() {
        if (cache_)
            cache_->thaw();
    }
    FrozenCache(const FrozenCache&) = delete;
    FrozenCache& operator=(const FrozenCache&) = delete;

  private:
    ThreadCache* cache_;
};

} // namespace infer::cpu_detail
