#include "infer/cpu/cpu_block.h"

#include <stdexcept>

namespace infer::cpu_detail {

std::size_t size_class(std::size_t bytes) {
    if (!bytes)
        throw std::invalid_argument("Cannot allocate zero bytes.");
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
        throw std::overflow_error("CPU pool allocation size overflow.");
    if (bytes <= 1024)
        return (bytes - 1) / alignment;
    const unsigned exponent = 63U - __builtin_clzll(static_cast<unsigned long long>(bytes - 1));
    const std::size_t base = std::size_t{1} << exponent;
    return 16 + (exponent - 10) * 4 + (bytes - base - 1) / (base / 4);
}

std::size_t FreeBins::first_nonempty(std::size_t bin) const noexcept {
    std::size_t word = bin / 64;
    std::uint64_t bits = nonempty_[word] & (~std::uint64_t{0} << (bin % 64));
    for (;;) {
        if (bits)
            return word * 64 + static_cast<unsigned>(__builtin_ctzll(bits));
        if (++word == word_count)
            return bin_count;
        bits = nonempty_[word];
    }
}
void FreeBins::push(Block* block) noexcept {
    const auto bin = block->bin;
    block->next = heads_[bin];
    heads_[bin] = block;
    ++counts_[bin];
    nonempty_[bin / 64] |= std::uint64_t{1} << (bin % 64);
}
Block* FreeBins::pop(std::size_t bin) noexcept {
    Block* block = heads_[bin];
    heads_[bin] = block->next;
    block->next = nullptr;
    --counts_[bin];
    if (!heads_[bin])
        nonempty_[bin / 64] &= ~(std::uint64_t{1} << (bin % 64));
    return block;
}
void Usage::add(const Block& block) noexcept {
    if (block.active) {
        active_bytes += block.capacity;
        ++active_blocks;
    } else {
        cached_bytes += block.capacity;
        ++cached_blocks;
    }
}
void Usage::remove(const Block& block) noexcept {
    if (block.active) {
        active_bytes -= block.capacity;
        --active_blocks;
    } else {
        cached_bytes -= block.capacity;
        --cached_blocks;
    }
}

} // namespace infer::cpu_detail
