#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mini_infer::detail {
inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* message) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(message);
    }
    return a * b;
}
inline std::size_t checked_add(std::size_t a, std::size_t b, const char* message) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::overflow_error(message);
    }
    return a + b;
}
}  // namespace mini_infer::detail
