// 实现 Block 的 split/coalesce，以及 bin 计算、查找和双向链表增删。
#include "infer/cuda/cuda_block.h"
#include "infer/cuda/cuda_memory_pool.h"
#include "infer/cuda/cuda_memory_stats.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infer::cuda_pool_detail {

// ---- Block 元数据与物理链表 ----

/**
 * @brief 判断一个块在切出指定容量后，剩余容量是否足够作为一个新的独立块。
 *
 * @param block           [in] 待检查的目标内存块（只读元数据，含 capacity 等）。
 * @param capacity        [in] 用户请求或需要切出的目标容量（字节数）。
 * @param min_split_bytes [in]
 * 允许切分的最小阈值（字节数）；若剩余空间小于该值，切分造成的碎片收益过低，不允许拆分。
 * @return true 允许拆分；false 剩余空间过小或配置禁止拆分。
 */
bool can_split(const Block& block, std::size_t capacity, std::size_t min_split_bytes) noexcept {
    return min_split_bytes && block.capacity >= capacity &&
           block.capacity - capacity >= min_split_bytes;
}

/**
 * @brief 将一个大块从逻辑上切分为两半：头部块（保留原地址）和尾部块（新生成元数据）。
 *
 * @param blocks   [in,out] 全局/池级别的地址索引映射表 (void* ->
 * Block)，用于登记新生成的尾块元数据。
 * @param bins     [in,out] 空闲容量桶索引数组 (Bins)，用于调整块所属的大小分类链表。
 * @param block    [in,out] 待拆分的原始块；切分后其容量缩减为 capacity，作为头部块保留。
 * @param capacity [in]     切分后头部块的目标容量（字节数）。
 * @return Block&  切分后新创建并插入物理链表、桶链表中的尾部块引用。
 */
Block& split_block(BlockMap& blocks, Bins& bins, Block& block, std::size_t capacity) {
    auto* tail_ptr = static_cast<char*>(block.ptr) + capacity;
    const auto tail_capacity = block.capacity - capacity;
    const unsigned tail_bin = free_bin(tail_capacity);
    const unsigned head_bin = free_bin(capacity);

    // 1. 先完成所有可能失败的计算和元数据分配，再触碰已有链表。
    // 插入失败时原块的地址、容量、epoch、链表和统计均不改变。
    // C++17 structured binding + try_emplace:
    // try_emplace 返回 pair(iterator, bool)，分别表示节点位置和是否插入成功。
    auto [it, inserted] = blocks.try_emplace(tail_ptr);
    if (!inserted) {
        throw std::logic_error("CUDA pool split metadata overlap");
    }
    Block& tail = it->second;
    tail.ptr = tail_ptr;
    tail.capacity = tail_capacity;
    tail.bin = tail_bin;
    tail.state = block.state;
    tail.owner = block.owner;
    tail.epoch = block.epoch;

    // 2. 从旧容量桶摘下头块，然后把尾块插入同一 segment 的物理链表。
    remove_bin(bins, block);
    tail.phys_prev = &block;
    tail.phys_next = block.phys_next;
    if (tail.phys_next) {
        tail.phys_next->phys_prev = &tail;
    }
    block.phys_next = &tail;
    block.capacity = capacity;
    block.bin = head_bin;

    // 3. 重新索引两块。以下和调用方的退休队列衔接都不分配内存、不抛异常。
    add_bin(bins, block);
    add_bin(bins, tail);
    return tail;
}

namespace {
/**
 * @brief 合并同一 Segment 内物理上相邻的两个 ready 状态空闲块。
 *
 * @param blocks [in,out] 全局/池地址索引表，合并后右块元数据会从中彻底销毁 (erase)。
 * @param ready  [in,out] ready 状态下的空闲容量桶索引链表，用于重排两块的大小归类。
 * @param left   [in,out] 物理地址靠前的块；吸收右块容量并更新物理/桶链表，作为合并后的主块保留。
 * @param right  [in,out] 物理地址靠后的块；合并后其元数据被脱钩并销毁。
 * @param stats  [in,out] 内存池统计指标；维护就绪块计数(--ready_blocks)与合并次数(++merges)。
 * @return Block& 合并后存活的左块引用（包含两者的总容量）。
 */
Block& merge(BlockMap& blocks, Bins& ready, Block& left, Block& right, CUDAMemoryPoolStats& stats) {
    // 两块来自同一次上游申请，容量和不超过该 segment 的原始容量。
    const auto combined = left.capacity + right.capacity;
    const auto bin = free_bin(combined); // 分桶可能抛出；必须在修改链表之前计算。

    remove_bin(ready, left);
    remove_bin(ready, right);
    left.capacity = combined;
    left.bin = bin;
    left.phys_next = right.phys_next;
    if (left.phys_next) {
        left.phys_next->phys_prev = &left;
    }
    // 先脱钩再销毁，不能留下指向已释放元数据的物理指针。
    blocks.erase(right.ptr);
    add_bin(ready, left);
    --stats.ready_blocks;
    ++stats.merges;
    return left;
}
} // namespace

/**
 * @brief 尝试对指定块进行双向相邻合并（左侧相邻 + 右侧相邻）。
 *
 * @param blocks [in,out] 全局/池地址索引表，透传给 merge 以便在合并成功时销毁被吸收块的元数据。
 * @param ready  [in,out] ready 状态桶集合，合并涉及的块均在此链表结构中脱钩与重挂。
 * @param block  [in,out] 触发合并的目标空闲块（通常刚归还为 ready 状态）。
 * @param stats  [in,out] 内存池统计信息对象，累计记录合并次数及存活块总数变化。
 */
void coalesce(BlockMap& blocks, Bins& ready, Block& block, CUDAMemoryPoolStats& stats) {
    // 左合并会销毁当前 block 节点，因此后续必须使用 merge 返回的存活节点。
    Block* current = &block;
    if (current->phys_prev && current->phys_prev->state == State::ready) {
        current = &merge(blocks, ready, *current->phys_prev, *current, stats);
    }
    if (current->phys_next && current->phys_next->state == State::ready) {
        merge(blocks, ready, *current, *current->phys_next, stats);
    }
}

/**
 * @brief 检查该块是否占据了一个完整的物理 Segment（即没有被任何切分）。
 *
 * @param block [in] 待测块（检查其物理前驱 phys_prev 与后继 phys_next 是否全为空）。
 * @return true 该块未被切割（是独立段）；false 该块由切分产生或属于切分链的一部分。
 */
bool whole_segment(const Block& block) noexcept {
    return !block.phys_prev && !block.phys_next;
}

// ---- 容量桶与桶内双向链表 ----

/**
 * @brief 计算给定字节数对应的向上对齐容量档位，以及对应的桶索引（用于分配时向上取整找桶）。
 *
 * @param bytes [in] 申请的最小内存大小（字节数）。
 * @return std::pair<unsigned, std::size_t> { 桶索引编号, 对齐后该桶保证容纳的基准容量 }。
 * @throw std::overflow_error 如果字节数超过池支持的最大 bin 范围。
 */
std::pair<unsigned, std::size_t> bucket(std::size_t bytes) {
    return size_classes::bucket(bytes, 8);
}

/**
 * @brief 计算一个空闲块在归还/存入时应放入哪个桶（向下归类，确保桶内所有块容量 >=
 * 该桶的承诺基准）。
 *
 * @param bytes [in] 空闲块的实际可用容量（字节数）。
 * @return unsigned 对应的空闲容量桶索引编号。
 */
unsigned free_bin(std::size_t bytes) {
    // Block 容量至少为 256 B。向下归类后，同桶任意块都不小于该桶请求容量。
    // 例如 6144 -> 1280 + 4864，4864 的尾块不能被放进 5120 的请求桶。
    // C++17 structured binding：把 bucket 返回的 pair 拆成两个变量。
    auto [bin, capacity] = bucket(bytes);
    return capacity == bytes ? bin : bin - 1;
}

std::pair<unsigned, std::size_t> allocation_bucket(std::size_t bytes,
                                                   const CachingCUDAAllocatorConfig& config) {
    const auto granularity = config.large_allocation_granularity;
    if (bytes < (16ULL << 20) || !granularity)
        return bucket(bytes);
    if (bytes > SIZE_MAX - (granularity - 1))
        throw std::overflow_error("CUDA pool large allocation overflow");
    const auto capacity = (bytes + granularity - 1) & ~(granularity - 1);
    return {free_bin(capacity), capacity};
}

/**
 * @brief 将块插入其所属 bin 索引的双向链表表头（LIFO 策略，提高缓存/时间局部性）。
 *
 * @param bins  [in,out] 目标桶数组，修改对应 bins[block.bin] 的头指针。
 * @param block [in,out] 待挂入的内存块元数据，将更新其内部的 prev 和 next 指针。
 */
void add_bin(Bins& bins, Block& block) noexcept {

    block.prev = nullptr;

    block.next = bins[block.bin];

    if (block.next)
        block.next->prev = &block;

    bins[block.bin] = &block;

    // 设置对应 bit
    bins.bitmap[block.bin / 64] |= (1ULL << (block.bin % 64));
}

/**
 * @brief 将块从其所在的容量桶双向链表中摘除（脱钩）。
 *
 * @param bins  [in,out] 目标桶数组，如果脱钩的是表头节点，则将更新 bins[block.bin]。
 * @param block [in,out] 待摘除的块；摘除后将其 prev、next 悬空指针清空为 nullptr。
 */
void remove_bin(Bins& bins, Block& block) noexcept {

    if (block.prev)
        block.prev->next = block.next;
    else
        bins[block.bin] = block.next;

    if (block.next)
        block.next->prev = block.prev;

    block.prev = nullptr;
    block.next = nullptr;

    // 如果整个桶空了
    if (bins[block.bin] == nullptr)
        bins.bitmap[block.bin / 64] &= ~(1ULL << (block.bin % 64));
}

/**
 * @brief 从空闲桶中查找可复用 Block。
 *
 * 使用 BinBitmap 快速跳过空桶：
 * bitmap 的每一位表示对应 bin 是否存在空闲块。
 * 通过 __builtin_ctzll() 利用 CPU bit-scan 指令定位下一个非空桶。
 *
 * 查找策略：
 * 1. 优先寻找可 split 或完全匹配的 Block；
 * 2. 大块允许有限容量浪费；
 * 3. 找不到返回 nullptr，由上层 cudaMalloc。
 */
Block* find(Bins& bins, unsigned bin, std::size_t capacity,
            const CachingCUDAAllocatorConfig& config) {
    const auto& bitmap = bins.bitmap;
    // 一个 uint64_t 管理 64 个 bin，定位当前 bitmap 段
    unsigned word = bin / 64;

    // 屏蔽比请求更小的 bin，只搜索 >= bin 的桶
    uint64_t mask = bitmap[word] & (~0ULL << (bin % 64));

    while (true) {

        if (mask) {

            // ctzll 返回最低位 1 的位置（CPU bit scan 指令）
            unsigned offset = __builtin_ctzll(mask);

            // bitmap 位置转换成真实 bin 编号
            unsigned candidate_bin = word * 64 + offset;

            // 拆分尾块向下分桶，同桶容量可能不同；桶头不匹配时继续查后继。
            for (auto* block = bins[candidate_bin]; block; block = block->next) {
                // Fine-grained large allocations can share a coarse search bin.
                if (block->capacity < capacity)
                    continue;
                // 可以切割或者大小完全匹配，直接复用
                if (can_split(*block, capacity, config.min_split_bytes) ||
                    block->capacity == capacity) {
                    return block;
                }

                // 大块允许一定浪费，避免过度 split
                if (capacity >= config.small_block_threshold &&
                    block->capacity - capacity <=
                        std::min(config.large_block_max_waste, capacity / 2)) {
                    return block;
                }
            }

            // 整个桶都不满足，删除最低位 1，继续寻找下一桶
            mask &= mask - 1;

        } else {

            // 当前 64 个 bin 没找到，检查下一组
            ++word;

            if (word >= bitmap.size()) {
                break;
            }

            mask = bitmap[word];
        }
    }

    return nullptr;
}
} // namespace infer::cuda_pool_detail
