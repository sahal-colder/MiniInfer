#pragma once
#include "infer/storage/storage_backend.h"
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infer {

struct KVOffloadConfig {
    std::size_t payload_bytes = 0;
    std::size_t logical_slots = 0; // 外部调度器持有的稳定槽编号，允许覆盖重用。
    std::size_t cpu_slots = 0;     // 1..logical_slots，预算按取整后的物理槽计费。
    std::string ssd_directory = "/tmp";
    FileIOConfig ssd_io;
    bool ssd_direct = false;
    std::size_t write_concurrency = 2; // 有界 CPU→SSD worker 数；1 可强制串行对照。
    std::size_t prefetch_depth = 8;    // 排队 + 执行中的预取上限；0 禁用。
    bool retain_ssd_copy = true;       // 预取后再次下放同版本时复用文件副本。
    HostMemoryKind cpu_memory = HostMemoryKind::PAGEABLE;
    SSDVerification ssd_verification = default_ssd_verification;
    std::uint32_t min_reuse_count = 1;
    bool prompt_only = false;
    bool compact_slots = false;       // 复用 CPU 池细分桶；默认保留原 2 的幂布局。
    std::size_t batch_promotions = 0; // 每批最多提升的重复 SSD 热块；0 保留扫描行为，绝不驱逐。
    bool overlap_writeback = false;   // 有界下放期间先填充已有空槽，返回前排空所有迁移。
    SpdkBdevConfig ssd_spdk;
    bool protect_hot_cpu = false; // Protect up to half the CPU slots after repeated reads.
    bool variable_payloads = false; // Exact tails; fixed-size callers retain their layout.
    // Unset: generic offload uses malloc; native checkpoints map CPU sources only.
    // Explicit false retains malloc reuse for warm allocation workloads.
    std::optional<bool> mapped_cpu_pool;
    bool enable_ssd = true;
    std::uint32_t promote_hits = 2; // 0 disables demand promotion.
    bool promote_evict = false; // Demand promotion may replace a cold CPU block.
};
struct KVOffloadMeta {
    BlockId id = invalid_block_id;
    std::size_t size = 0;
    StorageLocation location = StorageLocation::CPU;
    std::uint64_t version = 0;
    bool ssd_copy = false; // A valid same-version retained copy, or the SSD primary.
};
// 内容身份、token 前缀命中次数由上层调度器提供；Storage 不猜测模型/layout/hash。
struct KVStoreHint {
    std::uint32_t reuse_count = 1;
    bool is_prompt = true;
};
struct KVOffloadStats {
    std::uint64_t writes = 0, cpu_hits = 0, ssd_hits = 0, demotions = 0, promotions = 0;
    std::size_t populated_slots = 0, slot_bytes = 0;
    std::size_t cpu_reserved_bytes = 0, cpu_limit_bytes = 0, cpu_pool_reserved_bytes = 0;
    std::size_t ssd_reserved_bytes = 0, ssd_limit_bytes = 0;
    // 单调时钟累计值；write_ns 按公开调用计时，批写包括预降级且只计一次。
    // 并行操作可能重叠，不能直接相加当作端到端时间。
    std::uint64_t read_ns = 0, write_ns = 0, lock_wait_ns = 0, demotion_ns = 0;
    std::uint64_t ssd_reserve_ns = 0, ssd_read_ns = 0, ssd_write_ns = 0, ssd_publish_ns = 0;
    FileIOStats file_io;
    std::uint64_t admission_skips = 0, invalidations = 0;
    std::uint64_t prefetch_submitted = 0, prefetch_deduplicated = 0, prefetch_completed = 0;
    std::uint64_t prefetch_skipped = 0, prefetch_failed = 0, prefetch_rejected = 0;
    std::uint64_t prefetch_hits = 0, prefetch_unused = 0;
    std::size_t prefetch_inflight = 0, prefetch_peak = 0;
    std::size_t ssd_replica_bytes = 0, cpu_pending_cleanup_bytes = 0, ssd_pending_cleanup_bytes = 0;
    std::uint64_t writer_wait_ns = 0, prefetch_queue_ns = 0, prefetch_ns = 0;
    std::uint64_t ssd_payload_written_bytes = 0, ssd_payload_read_bytes = 0,
                  ssd_verification_read_bytes = 0;
    std::uint64_t ssd_record_writes = 0, ssd_source_crc_bytes = 0;
    std::uint64_t overlapped_writes = 0;
    std::size_t protected_cpu_slots = 0;
    std::uint64_t hot_promotions = 0;
    SSDDeviceStats ssd_device;
    std::uint64_t ssd_copied_read_bytes = 0, ssd_copied_write_bytes = 0;
    std::uint64_t ssd_direct_read_bytes = 0, ssd_direct_written_bytes = 0;
};

// GPU 层由调用方持有；本类使用 TieredStorage 管理 CPU/SSD 副本。
// write 先进入 CPU，满载后写回 SSD。可选热点保护优先下放冷块；重复 SSD 需求可替换冷块。
// 外部调度器决定逻辑槽的有效期；内部只改变位置，不静默淘汰任何已写槽。
// 写入串行，同槽操作互斥；不同槽读取可并发，I/O 不占元数据锁。
// 调用方必须持有有效 payload 大小的主机缓冲，DMA 已完成。
// 不提供掉电持久化。析构不能与外部调用并发。
class KVOffloadCache {
  public:
    // CPU slot capacity after the configured rounding, without an SSD record header.
    static std::size_t allocation_size(std::size_t payload_bytes, bool compact_slots = false);
    explicit KVOffloadCache(KVOffloadConfig config,
                            std::unique_ptr<StorageBackend> cpu = {},
                            std::unique_ptr<StorageBackend> ssd = {});
    ~KVOffloadCache();
    KVOffloadCache(const KVOffloadCache&) = delete;
    KVOffloadCache& operator=(const KVOffloadCache&) = delete;
    void write(std::size_t slot, const void* data);
    // 一批持有一次写入权，各槽独立提交；后项失败保留已完成项，可重试。
    // 预降级允许不同 SSD 槽并行写；无关槽读取不等待整批写入。
    void write_many(const std::vector<std::size_t>& slots, const std::vector<void*>& buffers);
    void write_many(const std::vector<std::size_t> &slots, const std::vector<void *> &buffers,
                    const std::vector<std::size_t> &sizes);
    void read(std::size_t slot, void* data);
    // Capacity is checked while protecting the same slot version; returns its actual length.
    std::size_t read(std::size_t slot, void* data, std::size_t capacity);
    KVOffloadMeta metadata(std::size_t slot) const; // Missing slot has id == 0.
    bool prefetch(std::size_t slot, bool allow_eviction = false);
    void read_many(const std::vector<std::size_t>& slots, const std::vector<void*>& buffers);
    // 提前调度 SSD→CPU，满载时下放一个无读者的 LRU 槽。同版本请求共用 future。
    // true 表示完成时已在 CPU；false 表示缺失、版本已变或没有可用 victim。
    // 满队列抛 StorageBusy；I/O 失败由 get() 抛出。析构排空已接受任务。
    std::shared_future<bool> prefetch_async(std::size_t slot);
    // 推测性提示：只使用空闲 CPU 容量，不驱逐热块、不等待写入权。
    std::shared_future<bool> prefetch_if_space_async(std::size_t slot);
    // 显式失效，等待同槽正在进行的读写；排队中的旧预取不会复活该槽。
    // 不存在时 false；失败保留资源记录和预算，可重试。
    bool erase(std::size_t slot);
    bool contains(std::size_t slot) const;
    bool should_store(KVStoreHint hint) const noexcept;
    // 拒绝准入时使旧槽失效，返回 false，避免逻辑槽重用后读到旧内容。
    bool write_if_admitted(std::size_t slot, const void* data, KVStoreHint hint);
    KVOffloadStats stats() const;
    // Device staging shares the SPDK DMA budget; filesystem backends return empty.
    std::shared_ptr<SSDDmaBuffer> allocate_dma_buffer(std::size_t bytes);

  private:
    std::shared_future<bool> submit_prefetch(std::size_t slot, bool allow_eviction);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace infer
