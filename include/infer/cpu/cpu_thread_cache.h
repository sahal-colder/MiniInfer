#pragma once

#include "infer/cpu/cpu_block.h"
#include <atomic>

namespace infer::cpu_detail {

/**
 * @brief 线程本地缓存（TLS Cache）数据结构
 *
 * 核心职责：
 * 1. 维护当前线程独占的小内存（<= 4KiB）空闲桶（free）与地址索引表（index）。
 * 2. 提供极速的本地分配与释放快路径（无需争抢全局中央 Mutex，仅需一次无竞争的原子位操作）。
 * 3. 支持被持有全局中央 Mutex
 * 的其他线程安全“冻结（freeze）”，以便执行跨线程内存归还、统计汇总或强制瘦身（trim）。
 */
struct ThreadCache {
    // 固定大小的侵入式哈希表桶数量（64 个桶），全程无 malloc 动态分配
    static constexpr std::size_t index_size = 64;

    // 当前线程最多允许持有（管理）的 Block 元数据总数上限（256 个）
    // 作用：1. 限制单个线程占用的内存与元数据规模；
    //       2. 配合 index_size = 64，保证哈希表平均冲突链长（Load Factor）不超过 256 / 64 =
    //       4，实现近似 O(1) 的极速查找。
    static constexpr std::size_t max_owned_blocks = 256;

    // TLS 地址哈希索引表：通过 Block::index_next 拉链法解决冲突，用于根据 void* ptr
    // 快速查找到对应的 Block*
    std::array<Block*, index_size> index{};

    // 当前归属于本线程（owner == this）的 Block 总数（包含 active 使用中 + free 空闲缓存中）
    std::size_t owned_blocks = 0;

    // 当前线程私有的空闲内存分桶（通过 Block::next 串联空闲块）
    FreeBins free;

    // 当前线程私有的内存容量与块数量统计账本（active / cached）
    Usage usage;

    // 性能计数器：缓存命中复用次数、本线程总分配次数、本线程总释放次数
    std::size_t reuse_count = 0, allocations = 0, releases = 0;

    // ==================== 1. 房间门禁控制（并发同步） ====================

    // 【本线程自己调】尝试推门进入快路径（点亮 bit0）。
    // 如果看到 bit1 亮着（说明中央线程正在门外等着查账/清理），立刻放弃并返回
    // false，不傻等，直接去走加全局锁的慢路径。
    bool try_enter() noexcept;

    // 【本线程自己调】操作完本地缓存后出门（熄灭 bit0）。
    void leave() noexcept;

    // 【中央维护线程调（调用前必须已持有全局 mutex）】强制封锁并接管此线程的缓存：
    // 第 1 步：先点亮 bit1（挂出“正在维护”牌子，拦住本线程新的 try_enter
    // 请求，防止维护线程被饿死）。 第 2 步：如果 bit0
    // 还亮着（本线程刚好在屋里没出来），就在门口自旋等几纳秒，直到 bit0 熄灭后正式接管。
    void freeze() noexcept;

    // 【中央维护线程调（调用前必须已持有全局 mutex）】维护完毕，解封此线程缓存（熄灭
    // bit1），允许本线程恢复快路径访问。
    void thaw() noexcept;

    // ==================== 2. 内存户口本管理（哈希索引与归属） ====================

    // 【查户口】在 64 格的 index 哈希表里，顺着 Block::index_next 链表查找地址 ptr 对应的 Block
    // 卡片。 全程只比对地址数值，绝不解引用读取 ptr 前面的内存，遇到野指针或外来指针会安全返回
    // nullptr。
    Block* find(void* ptr) const noexcept;

    // 【上户口】将一个 Block 划归当前线程管辖：
    // 标记 block->owner = this，按地址算出 0~63 的槽位并头插进 index 哈希表，递增持有数
    // owned_blocks 并累加账本 usage。
    void attach(Block* block) noexcept;

    // 【销户口】将一个 Block 从当前线程移除（比如缓存满 256 个要退回中央池，或被 trim 清理时）：
    // 从 index 哈希表链表中摘除，清空 block->owner，递减持有数 owned_blocks 并扣减账本 usage。
    void detach(Block* block) noexcept;

  private:
    /**
     * @brief 房门口的“双灯电子门牌”（2-bit 原子状态变量，初始值为 0）
     *
     * 语法拆解：
     * - alignas(64)          : 强制按 64 字节对齐内存地址，让 gate_ 独占一个 CPU 缓存行（Cache
     * Line），避免和其他变量挤在一起互相拖慢速度（消除伪共享）。
     * - std::atomic<unsigned>:
     * 线程安全的无符号整数，保证本线程和中央线程同时修改它时绝对不会撞车（原子操作）。
     * - gate_{0}             : 变量名为 gate_，初始值设为 0（两个灯全灭）。
     *
     * 指示灯含义（只用二进制的最后两位）：
     * - 第 0 位 (bit0, 值为 1): “本线程正在屋里操作”指示灯。
     * - 第 1 位 (bit1, 值为 2): “中央线程请求维护封门”指示灯。
     */
    alignas(64) std::atomic<unsigned> gate_{0};

    // 哈希计算函数：把 64B 对齐的内存地址 ptr（右移 6 位去掉末尾的 0 后打散）转换成 0 ~ 63
    // 的数组下标。
    static std::size_t hash(void* ptr) noexcept;
};

/**
 * @brief 【本线程专用】快路径自动门卡（RAII 守卫：进作用域自动开灯，出作用域自动关灯，防止忘调
 * leave 导致死锁）
 *
 * 典型用法：
 *   if (LocalAccess access{tls_cache}) {
 *       // 抢门成功！安全地无锁操作 tls_cache->free（按大小拿块）或 tls_cache->index（按地址查块）
 *   } else {
 *       // 抢门失败（撞上中央线程在 freeze），直接放弃快路径，改走加全局锁的慢路径
 *   }
 */
class LocalAccess {
  public:
    // 构造时自动调用 try_enter() 尝试进屋；如果进不去，就把内部的 cache_ 指针设为 nullptr
    explicit LocalAccess(ThreadCache* cache) noexcept
        : cache_(cache && cache->try_enter() ? cache : nullptr) {}

    // 离开大括号析构时：如果之前成功进屋了（cache_ 不为空），自动调用 leave() 关掉 bit0
    ~LocalAccess() {
        if (cache_)
            cache_->leave();
    }

    // 支持直接写在 if (access) 里：如果 cache_ 不为空（进屋成功）就返回 true，否则返回 false
    explicit operator bool() const noexcept { return cache_ != nullptr; }

    // 禁止复制这张门卡，防止析构多次导致重复调用 leave()
    LocalAccess(const LocalAccess&) = delete;
    LocalAccess& operator=(const LocalAccess&) = delete;

  private:
    ThreadCache* cache_;
};

/**
 * @brief 【中央维护线程专用】缓存自动封条（RAII 守卫：创建时自动冻结目标线程缓存，销毁时自动解冻）
 *
 * 使用铁律：
 * 1. 必须在已经拿到全局中央锁（Central Mutex）的前提下才能创建此对象（保证同时只有一个维护者）。
 * 2. 严禁对同一个 ThreadCache 连续套两层 FrozenCache（否则第二次 freeze 会等第一次
 * thaw，把自己卡死）。
 */
class FrozenCache {
  public:
    // 构造时自动调用 freeze()：点亮 bit1 挂出封条，并等待目标线程从快路径彻底退出（bit0 归零）
    explicit FrozenCache(ThreadCache* cache) noexcept : cache_(cache) {
        if (cache_)
            cache_->freeze();
    }

    // 离开大括号析构时：自动调用 thaw() 撕下封条（熄灭 bit1），恢复目标线程的快路径访问
    ~FrozenCache() {
        if (cache_)
            cache_->thaw();
    }

    // 禁止复制封条，防止重复冻结/解冻
    FrozenCache(const FrozenCache&) = delete;
    FrozenCache& operator=(const FrozenCache&) = delete;

  private:
    ThreadCache* cache_;
};

} // namespace infer::cpu_detail