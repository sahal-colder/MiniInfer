// 集中定义 Block 元数据、容量桶、物理拆分/合并与空闲桶链表。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace infer {
struct CachingCUDAAllocatorConfig;
struct CUDAMemoryPoolStats;

namespace cuda_pool_detail {
struct Stream; // 非拥有关系；本头文件不依赖 CUDA，也不操作 stream/event。
struct Block;
inline constexpr unsigned bin_count = sizeof(std::size_t) * 8 * 4;
using BinBitmap = std::array<std::uint64_t, (bin_count + 63) / 64>;
// 桶头与位图一起传递，split/merge/stream 迁移时不会遗漏位图更新。
struct Bins {
    std::array<Block*, bin_count> heads{};
    BinBitmap bitmap{};
    Block*& operator[](std::size_t bin) noexcept { return heads[bin]; }
    auto begin() noexcept { return heads.begin(); }
    auto end() noexcept { return heads.end(); }
};

enum class State {
    active,  // 已交给调用者；不能从空闲桶复用。
    pending, // CPU 已归还，GPU 未确认完成；仅原 stream 可直接复用。
    ready    // GPU 已完成；所有登记的 stream 均可复用。
};

struct Block {
    void* ptr{};
    std::size_t capacity{};  // 实际容量，参与分桶和显存统计。
    std::size_t requested{}; // 调用者请求量，可能小于 capacity。
    unsigned bin{};
    State state{State::active};
    Stream* owner{}; // 指向稳定的 stream map 节点，不拥有该节点。
    unsigned long long epoch{};

    // 三组链表彼此独立，不能用其中一组代替另一组。
    Block* prev{}; // 桶内双向链表：按容量寻找可复用块。
    Block* next{};
    Block* retire_prev{}; // 退休 FIFO：按 epoch 判断完成顺序。
    Block* retire_next{};
    Block* phys_prev{}; // 同一次 cudaMalloc 内的物理邻居。
    Block* phys_next{}; // 绝不跨 segment 合并，即使地址恰好相邻。
};

// map 节点拥有 Block；所有链表指针均为非拥有指针。
// unordered_map rehash 不使节点地址失效，因此扩表不会破坏这些链表。
using BlockMap = std::unordered_map<void*, Block>;

// 调用前 capacity <= block.capacity；min_split_bytes==0 表示禁用拆分。
bool can_split(const Block& block, std::size_t capacity, std::size_t min_split_bytes) noexcept;

// 调用前先通过 can_split。返回尾块，保留原 state/owner/epoch。
// 本函数处理元数据、物理链表、桶链表；调用者随后连接退休 FIFO 并更新 split 计数。
Block& split_block(BlockMap& blocks, Bins& bins, Block& block, std::size_t capacity);

// 只合并 ready 物理邻居；更新元数据、ready 桶及 ready_blocks/merges。
void coalesce(BlockMap& blocks, Bins& ready, Block& block, CUDAMemoryPoolStats& stats);

// 仅完整 segment 的首地址可交给 cudaFree，内部子块地址不可以。
bool whole_segment(const Block& block) noexcept;

// 请求容量向上取整到原几何档位，返回 {桶编号, 档位容量}；保留溢出检查。
std::pair<unsigned, std::size_t> bucket(std::size_t bytes);

// Large requests retain a coarse search bin but use a finer physical capacity.
std::pair<unsigned, std::size_t> allocation_bucket(std::size_t bytes,
                                                   const CachingCUDAAllocatorConfig& config);

// 已有空闲块向下分桶；split 后的非档位容量也不能被当成更大的块。
unsigned free_bin(std::size_t bytes);

// 侵入式链表只改 Block 指针，不分配主机内存；调用者须持有 Pool 的 mutex。
void add_bin(Bins& bins, Block& block) noexcept;
void remove_bin(Bins& bins, Block& block) noexcept;

// 保留原策略：从请求桶向上检查每桶头块，按 split 或允许浪费量决定是否复用。
Block* find(Bins& bins, unsigned bin, std::size_t capacity,
            const CachingCUDAAllocatorConfig& config);

} // namespace cuda_pool_detail
} // namespace infer
