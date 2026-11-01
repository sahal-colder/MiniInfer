#pragma once

#include "infer/storage/storage_backend.h"
#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

namespace infer::storage_detail {

// Handles carry a process-wide tag and a direct slot index. Reusing a slot never
// revives an old handle, and a handle from another backend cannot alias it.
// The caller serializes access; values must remain valid when the vector grows.
template <class T> class SlotTable {
    static constexpr unsigned index_bits = 24;
    static constexpr std::uint32_t no_slot = (1u << index_bits) - 1;
    struct Slot {
        T value{};
        StorageHandle handle = 0;
        std::uint32_t next = no_slot;
    };
    std::vector<Slot> slots_;
    std::uint32_t free_ = no_slot;
    std::size_t size_ = 0;

  public:
    template <class Factory>
    StorageHandle create(Factory make, std::size_t limit = no_slot) {
        static_assert(std::is_nothrow_move_assignable<T>::value);
        static_assert(std::is_nothrow_move_constructible<T>::value);
        const bool grow = free_ == no_slot;
        if (grow) {
            if (slots_.size() >= limit || slots_.size() == no_slot)
                throw std::bad_alloc();
            if (slots_.size() == slots_.capacity())
                slots_.reserve(std::min(std::size_t(no_slot),
                                       std::max(std::size_t(8), slots_.size() * 2)));
        }
        const auto tag = next_storage_handle();
        if (tag > (UINT64_MAX >> index_bits))
            throw std::overflow_error("Storage handle tags exhausted");
        const auto index = grow ? static_cast<std::uint32_t>(slots_.size()) : free_;
        const auto handle = (tag << index_bits) | index;
        // Metadata is already allocated; a failed factory leaves the free slot
        // available and publishes neither a handle nor a partially owned value.
        auto value = make(index);
        if (grow)
            slots_.push_back({std::move(value), handle, no_slot});
        else {
            auto& slot = slots_[index];
            slot.value = std::move(value);
            slot.handle = handle;
            free_ = slot.next;
        }
        ++size_;
        return handle;
    }
    const T& at(StorageHandle handle) const {
        const auto index = static_cast<std::uint32_t>(handle & no_slot);
        if (!handle || index >= slots_.size() || slots_[index].handle != handle)
            throw std::out_of_range("Invalid storage handle");
        return slots_[index].value;
    }
    void erase(StorageHandle handle) {
        (void)at(handle);
        const auto index = static_cast<std::uint32_t>(handle & no_slot);
        slots_[index].value = T{};
        slots_[index].handle = 0;
        slots_[index].next = free_;
        free_ = index;
        --size_;
    }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return slots_.size(); }
    template <class Function> void for_each(Function visit) const {
        for (const auto& slot : slots_)
            if (slot.handle)
                visit(slot.value);
    }
};
} // namespace infer::storage_detail
