#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace infer::size_classes {
// 64 subdivisions per octave bound geometric rounding to less than 1.563%.
inline constexpr unsigned parts = 64;
inline constexpr unsigned count = 64 + 64 * (sizeof(std::size_t) * 8 - 12);

inline std::pair<unsigned, std::size_t> bucket(std::size_t bytes, unsigned alignment_bits) {
    const auto alignment = std::size_t{1} << alignment_bits;
    const auto linear = alignment * parts;
    if (!bytes || bytes > SIZE_MAX - (alignment - 1))
        throw std::overflow_error("allocation size outside size classes");
    if (bytes <= linear)
        return {unsigned((bytes - 1) / alignment), (bytes + alignment - 1) & ~(alignment - 1)};
    const unsigned exponent = 63U - __builtin_clzll(bytes - 1);
    const auto base = std::size_t{1} << exponent;
    const auto step = base / parts;
    const auto part = (bytes - base - 1) / step;
    const auto capacity = part == parts - 1 && exponent == sizeof(std::size_t) * 8 - 1
        ? SIZE_MAX - (alignment - 1) : base + (part + 1) * step;
    return {unsigned(parts + (exponent - alignment_bits - 6) * parts + part), capacity};
}
}
