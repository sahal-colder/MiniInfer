#pragma once

#include <cstddef>
#include <limits>

namespace infer {

struct CachingCPUAllocatorConfig {
    // 中央池 + 所有线程空闲链表的总上限，不限制 active 内存；0 禁用缓存。
    std::size_t max_cached_bytes = 64ULL * 1024 * 1024;
    // 跨桶借用的余量还必须 <= 请求桶容量的 50%。
    std::size_t max_reuse_waste = 1ULL * 1024 * 1024;
    bool enable_thread_cache = true;
    // <= 4 KiB 才进入 TLS；一次交换已有块，不预分配 batch 个上游块。
    std::size_t thread_cache_batch_size = 8;
    std::size_t thread_cache_blocks_per_bin = 32;
    // Hard bound on active + cached; discard cached size classes before growing.
    std::size_t max_reserved_bytes = std::numeric_limits<std::size_t>::max();
    // Optional pageable central-pool class, mapped in bounded ~2 MiB batches.
    // Every returned or cached slice is charged to max_reserved_bytes.
    std::size_t mapped_block_bytes = 0;
};

struct CPUMemoryPoolStats {
    // stats() 暂停各线程的短元数据操作，获得一致快照（不统计用户数据读写）。
    // reserved == active + cached；cached 包含中央缓存和线程缓存。
    std::size_t active_bytes = 0;
    std::size_t cached_bytes = 0;
    std::size_t reserved_bytes = 0;
    std::size_t active_blocks = 0;
    std::size_t cached_blocks = 0;
    std::size_t upstream_allocations = 0;
    std::size_t upstream_releases = 0;
    std::size_t reuse_count = 0;
    std::size_t thread_cached_bytes = 0;
    std::size_t thread_cached_blocks = 0;
    std::size_t thread_cache_allocations = 0;
    std::size_t thread_cache_releases = 0;
    // 只有 allocate/release 慢路径计数；stats/trim/线程退出不计入。
    std::size_t central_allocations = 0;
    std::size_t central_releases = 0;
    std::size_t thread_cache_refills = 0;
    std::size_t thread_cache_spills = 0;
    std::size_t mapped_allocations = 0; // mmap batches, separate from upstream allocator calls.
    std::size_t mapped_reserved_bytes = 0; // Live mapped slices, including cached slices.
};

} // namespace infer
