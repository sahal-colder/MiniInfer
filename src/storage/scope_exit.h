#pragma once

#include <utility>

namespace infer::storage_detail {
// Local cleanup for state protected by an existing lock; not an error boundary.
template <class Function> struct ScopeExit {
    Function function;
    explicit ScopeExit(Function value) : function(std::move(value)) {}
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ~ScopeExit() { function(); }
};
} // namespace infer::storage_detail
