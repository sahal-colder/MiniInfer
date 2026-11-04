#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace infer::cpu_detail {

// 内存对齐基准：64 字节对齐，适配现代 CPU 缓存行（Cache Line）及 AVX-512 SIMD 指令要求
constexpr std::size_t alignment = 64;

// 当前平台 std::size_t 的二进制位数（通常 32 位系统为 32，64 位系统为 64）
constexpr unsigned size_bits = std::numeric_limits<std::size_t>::digits;
static_assert(size_bits >= 32 && size_bits <= 64);

// 64-byte linear classes through 4 KiB, then 64 classes per octave.
constexpr std::size_t bin_count = 64 + 64 * (size_bits - 12);
constexpr std::size_t small_bin_count = 64;
constexpr std::size_t small_limit = 4096;
constexpr std::size_t word_count = (bin_count + 63) / 64;
constexpr auto make_capacities() {
    std::array<std::size_t, bin_count> result{};
    std::size_t index = 0;
    for (std::size_t n = 1; n <= 64; ++n) result[index++] = n * alignment;
    for (unsigned exponent = 12; exponent < size_bits; ++exponent) {
        const std::size_t base = std::size_t{1} << exponent;
        for (unsigned part = 1; part <= 64; ++part)
            result[index++] = exponent == size_bits - 1 && part == 64
                ? std::numeric_limits<std::size_t>::max() - (alignment - 1)
                : base + (base / 64) * part;
    }
    return result;
}

// 编译期常量表：记录每个 bin 索引对应的字节容量上限
inline constexpr auto capacities = make_capacities();

// 根据请求分配的字节数，计算其对应的分桶索引（Size Class Index）
std::size_t size_class(std::size_t bytes);

// 线程本地缓存（Thread Local Storage Cache）前向声明
struct ThreadCache;

/**
 * @brief 内存块元数据描述符（外部元数据设计，与实际数据载荷分离）
 */
struct Block {
    void* ptr = nullptr;      // 指向实际分配给用户的对齐内存起始地址
    std::size_t capacity = 0; // 该内存块的实际容量（字节）
    std::size_t bin = 0;      // 所属的分桶（Size Class）索引
    bool active = true;       // 状态标记：true 表示正在被用户使用，false 表示已回收处于空闲缓存中
    Block* next = nullptr;    // 空闲桶中的侵入式单向链表指针（用于 FreeBins 串联空闲块）
    Block* index_next = nullptr; // TLS 地址哈希索引冲突链指针；不读 ptr 前缀，安全拒绝外来指针。
    ThreadCache* owner =
        nullptr; // 指向当前持有或分配该块的线程缓存（TLS），用于跨线程释放识别或归属管理
};

/**
 * @brief 空闲内存块分桶管理器
 *
 * 中央池和 TLS 采用同一分桶规则；所有链表操作均基于 Block 节点指针，不触发任何动态内存分配。
 * 内部结合了单向链表数组与 64 位位图（nonempty_），支持 O(1) 级别查找可用空闲桶。
 */
class FreeBins {
  public:
    // 从指定的 bin 索引开始，利用位图向上查找第一个非空的桶索引（Best-Fit 查找）
    std::size_t first_nonempty(std::size_t bin) const noexcept;

    // 查看指定桶的链表头部内存块（不弹出）
    Block* front(std::size_t bin) const noexcept { return heads_[bin]; }

    // 获取指定桶中当前缓存的空闲内存块数量
    std::size_t count(std::size_t bin) const noexcept { return counts_[bin]; }

    // 将一个空闲内存块压入对应 bin 的链表头部，并更新计数和位图状态
    void push(Block* block) noexcept;

    // 从指定 bin 的链表头部弹出一个空闲内存块，并更新计数和位图状态
    Block* pop(std::size_t bin) noexcept;

  private:
    std::array<Block*, bin_count> heads_{};       // 每个桶的空闲块单向链表头指针
    std::array<std::size_t, bin_count> counts_{}; // 每个桶当前的空闲块数量统计
    std::array<std::uint64_t, word_count>
        nonempty_{}; // 非空状态位图：第 i 位为 1 表示第 i 个桶非空，加速查找
};

/**
 * @brief 内存池使用量统计结构
 */
struct Usage {
    std::size_t active_bytes = 0, cached_bytes = 0;   // 活跃（使用中）字节数 / 空闲缓存字节数
    std::size_t active_blocks = 0, cached_blocks = 0; // 活跃（使用中）块数量 / 空闲缓存块数量

    // 根据 block.active 状态，将该块的容量和数量累加到对应的统计项中
    void add(const Block& block) noexcept;
    // 根据 block.active 状态，从对应的统计项中扣除该块的容量和数量
    void remove(const Block& block) noexcept;
};

} // namespace infer::cpu_detail