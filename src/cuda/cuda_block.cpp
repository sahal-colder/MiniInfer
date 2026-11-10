#include "infer/cuda/cuda_block.h"
#include "infer/cuda/cuda_memory_pool.h"
#include "infer/cuda/cuda_memory_stats.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infer::cuda_pool_detail {

bool can_split(const Block& block, std::size_t capacity, std::size_t min_split_bytes) noexcept {
    return min_split_bytes && block.capacity >= capacity &&
           block.capacity - capacity >= min_split_bytes;
}

Block& split_block(BlockMap& blocks, Bins& bins, Block& block, std::size_t capacity) {
    auto* tail_ptr = static_cast<char*>(block.ptr) + capacity;
    const auto tail_capacity = block.capacity - capacity;
    const unsigned tail_bin = free_bin(tail_capacity);
    const unsigned head_bin = free_bin(capacity);

    // 先完成可能失败的计算和元数据分配，再触碰已有链表。
    // 插入失败时原块的地址、容量、epoch、链表和统计均不改变。
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

    // 尾块留在同一 segment，并继承 state/owner/epoch。
    remove_bin(bins, block);
    tail.phys_prev = &block;
    tail.phys_next = block.phys_next;
    if (tail.phys_next) {
        tail.phys_next->phys_prev = &tail;
    }
    block.phys_next = &tail;
    block.capacity = capacity;
    block.bin = head_bin;

    // 此后重新索引及调用方的退休队列衔接均不分配内存、不抛异常。
    add_bin(bins, block);
    add_bin(bins, tail);
    return tail;
}

namespace {
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

bool whole_segment(const Block& block) noexcept {
    return !block.phys_prev && !block.phys_next;
}

std::pair<unsigned, std::size_t> bucket(std::size_t bytes) {
    return size_classes::bucket(bytes, 8);
}

unsigned free_bin(std::size_t bytes) {
    // Block 容量至少为 256 B。向下归类后，同桶任意块都不小于该桶请求容量。
    // 例如 6144 -> 1280 + 4864，4864 的尾块不能被放进 5120 的请求桶。
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

void add_bin(Bins& bins, Block& block) noexcept {
    block.prev = nullptr;
    block.next = bins[block.bin];
    if (block.next)
        block.next->prev = &block;
    bins[block.bin] = &block;
    bins.bitmap[block.bin / 64] |= (1ULL << (block.bin % 64));
}

void remove_bin(Bins& bins, Block& block) noexcept {
    if (block.prev)
        block.prev->next = block.next;
    else
        bins[block.bin] = block.next;
    if (block.next)
        block.next->prev = block.prev;
    block.prev = nullptr;
    block.next = nullptr;
    if (bins[block.bin] == nullptr)
        bins.bitmap[block.bin / 64] &= ~(1ULL << (block.bin % 64));
}

Block* find(Bins& bins, unsigned bin, std::size_t capacity,
            const CachingCUDAAllocatorConfig& config) {
    const auto& bitmap = bins.bitmap;
    unsigned word = bin / 64;

    uint64_t mask = bitmap[word] & (~0ULL << (bin % 64));

    while (true) {
        if (mask) {
            unsigned offset = __builtin_ctzll(mask);

            unsigned candidate_bin = word * 64 + offset;

            // 拆分尾块向下分桶，同桶容量可能不同；桶头不匹配时继续查后继。
            for (auto* block = bins[candidate_bin]; block; block = block->next) {
                // Fine-grained large allocations can share a coarse search bin.
                if (block->capacity < capacity)
                    continue;
                if (can_split(*block, capacity, config.min_split_bytes) ||
                    block->capacity == capacity) {
                    return block;
                }

                if (capacity >= config.small_block_threshold &&
                    block->capacity - capacity <=
                        std::min(config.large_block_max_waste, capacity / 2)) {
                    return block;
                }
            }

            mask &= mask - 1;

        } else {
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
