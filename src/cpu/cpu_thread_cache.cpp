#include "infer/cpu/cpu_thread_cache.h"
#include <thread>

namespace infer::cpu_detail {

bool ThreadCache::try_enter() noexcept {
    unsigned expected = 0;
    return gate_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                         std::memory_order_relaxed);
}
void ThreadCache::leave() noexcept {
    gate_.fetch_and(~1U, std::memory_order_release); // 保留中央线程设置的维护位。
}
void ThreadCache::freeze() noexcept {
    gate_.fetch_or(2U, std::memory_order_acquire);
    while (gate_.load(std::memory_order_acquire) & 1U)
        std::this_thread::yield();
}
void ThreadCache::thaw() noexcept {
    gate_.store(0, std::memory_order_release);
}

std::size_t ThreadCache::hash(void* ptr) noexcept {
    auto value = reinterpret_cast<std::uintptr_t>(ptr) >> 6;
    value ^= value >> 16;
    value ^= value >> 8;
    return value & (index_size - 1);
}
Block* ThreadCache::find(void* ptr) const noexcept {
    for (Block* block = index[hash(ptr)]; block; block = block->index_next)
        if (block->ptr == ptr)
            return block;
    return nullptr;
}
void ThreadCache::attach(Block* block) noexcept {
    auto& head = index[hash(block->ptr)];
    block->index_next = head;
    head = block;
    block->owner = this;
    ++owned_blocks;
}
void ThreadCache::detach(Block* block) noexcept {
    Block** link = &index[hash(block->ptr)];
    while (*link != block)
        link = &(*link)->index_next;
    *link = block->index_next;
    block->index_next = nullptr;
    block->owner = nullptr;
    --owned_blocks;
}

} // namespace infer::cpu_detail
