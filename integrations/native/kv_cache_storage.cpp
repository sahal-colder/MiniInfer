#include "infer/integrations/native.h"
#include "mini_infer/base/checked_math.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <sys/mman.h>
#include <utility>

namespace infer::native {
namespace {
constexpr std::size_t page_tokens = 256;
struct RestoreBufferDelete {
    std::size_t mapped_bytes = 0;
    void operator()(float* data) const noexcept {
        if (mapped_bytes) {
            munmap(data, mapped_bytes);
            return;
        }
        delete[] data;
    }
};
auto restore_buffer(std::size_t elements) {
    if (!elements)
        return std::unique_ptr<float, RestoreBufferDelete>(nullptr, {});
    if (elements < (std::size_t(2) << 20) / sizeof(float))
        return std::unique_ptr<float, RestoreBufferDelete>(new float[elements], {});
    const auto bytes = elements * sizeof(float);
    auto* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED)
        throw std::bad_alloc();
    std::unique_ptr<float, RestoreBufferDelete> buffer(static_cast<float*>(data), {bytes});
#ifdef MADV_HUGEPAGE
    // A local best-effort hint: without it, first writes to a large fresh
    // buffer can fault once per 4 KiB during otherwise warm cache reads.
    (void)madvise(buffer.get(), bytes, MADV_HUGEPAGE);
#endif
    return buffer;
}
std::size_t page_count(int32_t length) {
    return (static_cast<std::size_t>(length) + page_tokens - 1) / page_tokens;
}
KVOffloadConfig configure(const mini_infer::KVCache& layout, KVOffloadConfig config) {
    if (config.logical_slots < static_cast<std::size_t>(layout.layer_num()))
        throw std::invalid_argument("KV checkpoint needs at least one slot per layer");
    if (!config.cpu_slots || config.cpu_slots > config.logical_slots)
        throw std::invalid_argument("KV checkpoint CPU slots must be 1..logical_slots");
    const auto cpu_bytes = mini_infer::detail::checked_mul(
        config.cpu_slots,
        KVOffloadCache::allocation_size(
            mini_infer::detail::checked_add(layout.byte_size() / layout.layer_num(),
                                            4 * sizeof(int32_t), "KV checkpoint payload overflow"),
            config.compact_slots),
        "KV checkpoint CPU budget overflow");
    // KVCache validates total byte size. Length/layout metadata stays in this
    // process-owned adapter, outside the payload's allocator size class.
    config.payload_bytes = layout.byte_size() / layout.layer_num() / layout.max_seq_len() *
                           std::min(page_tokens, static_cast<std::size_t>(layout.max_seq_len()));
    config.logical_slots = mini_infer::detail::checked_mul(
        config.logical_slots, page_count(layout.max_seq_len()), "KV checkpoint slot overflow");
    config.cpu_slots = std::min(
        config.logical_slots,
        cpu_bytes / KVOffloadCache::allocation_size(config.payload_bytes, config.compact_slots));
    config.variable_payloads = true;
    config.mapped_cpu_pool = config.mapped_cpu_pool.value_or(
        layout.device_type() == mini_infer::DeviceType::CPU);
    return config;
}
} // namespace

KVCacheStorage::KVCacheStorage(const mini_infer::KVCache &layout, KVOffloadConfig config)
    : layers_(layout.layer_num()), capacity_(layout.max_seq_len()), dim_(layout.kv_dim()),
      elements_(std::min(page_tokens, static_cast<std::size_t>(capacity_)) * dim_),
      slots_(config.logical_slots), pages_per_layer_(page_count(capacity_)),
      storage_(configure(layout, std::move(config))), checkpoints_(slots_) {}

void KVCacheStorage::check(const mini_infer::KVCache& cache, std::size_t first_slot) const {
    if (cache.layer_num() != layers_ || cache.max_seq_len() != capacity_ || cache.kv_dim() != dim_)
        throw std::invalid_argument("KV checkpoint layout mismatch");
    if (first_slot > slots_ - layers_)
        throw std::out_of_range("KV checkpoint slot range exceeds storage");
}

void KVCacheStorage::save(const mini_infer::KVCache& cache, std::size_t first_slot) {
    check(cache, first_slot);
    mini_infer::CPUDeviceAllocator copier;
    std::vector<float> window;
    std::vector<std::size_t> slots;
    std::vector<void*> buffers;
    std::vector<std::size_t> sizes;
    slots.reserve(64);
    buffers.reserve(64);
    sizes.reserve(64);
    for (int32_t layer = 0; layer < layers_; ++layer) {
        auto& checkpoint = checkpoints_[first_slot + layer];
        checkpoint.length = -1;
        const auto length = cache.valid_length(layer);
        const auto pages = page_count(length);
        if (pages) {
            const auto history = cache.history(layer);
            const auto elements = static_cast<std::size_t>(length) * dim_;
            const bool gpu = cache.device_type() == mini_infer::DeviceType::CUDA;
            auto payload = restore_buffer(gpu ? 2 * elements : 0);
            if (gpu) {
                copier.copy(history.first.ptr<float>(), payload.get(), elements * sizeof(float),
                            mini_infer::MemcpyKind::CUDA2CPU);
                copier.copy(history.second.ptr<float>(), payload.get() + elements,
                            elements * sizeof(float), mini_infer::MemcpyKind::CUDA2CPU);
            }
            const auto *keys = gpu ? payload.get() : history.first.ptr<float>();
            const auto *values = gpu ? payload.get() + elements : history.second.ptr<float>();
            const auto batch = std::min(std::size_t(64), pages);
            for (std::size_t first = 0; first < pages; first += batch) {
                slots.clear();
                buffers.clear();
                sizes.clear();
                for (std::size_t i = 0; i < std::min(batch, pages - first); ++i) {
                    // Pack the entire K prefix followed by V, then split bytes
                    // into pages. Restore can read straight into this layout.
                    const auto offset = (first + i) * 2 * elements_;
                    const auto count = std::min(2 * elements_, 2 * elements - offset);
                    const float *page =
                        offset < elements ? keys + offset : values + offset - elements;
                    if (!gpu && offset < elements && count > elements - offset) {
                        window.resize(count);
                        const auto keys_count = std::min(count, elements - offset);
                        std::memcpy(window.data(), keys + offset, keys_count * sizeof(float));
                        std::memcpy(window.data() + keys_count, values,
                                    (count - keys_count) * sizeof(float));
                        page = window.data();
                    }
                    slots.push_back((first_slot + layer) * pages_per_layer_ + first + i);
                    buffers.push_back(const_cast<float *>(page));
                    sizes.push_back(count * sizeof(float));
                }
                // A failed batch may have committed only some pages. Retain the
                // attempted range for later overwrite/cleanup, but do not publish length.
                checkpoint.pages = std::max(checkpoint.pages, first + slots.size());
                storage_.write_many(slots, buffers, sizes);
            }
        }
        for (auto page = pages; page < checkpoint.pages; ++page)
            storage_.erase((first_slot + layer) * pages_per_layer_ + page);
        checkpoint.pages = pages;
        checkpoint.length = length;
    }
}

void KVCacheStorage::restore(mini_infer::KVCache& cache, std::size_t first_slot) {
    check(cache, first_slot);
    std::vector<std::size_t> slots;
    std::vector<void*> buffers;
    slots.reserve(64);
    buffers.reserve(64);
    const auto begin = checkpoints_.begin() + first_slot;
    const auto end = begin + layers_;
    if (std::any_of(begin, end, [](const auto& layer) { return layer.length < 0; }))
        throw std::logic_error("KV checkpoint layer has not been completely saved");
    std::vector<std::size_t> offsets(static_cast<std::size_t>(layers_) + 1);
    for (int32_t layer = 0; layer < layers_; ++layer)
        offsets[layer + 1] =
            offsets[layer] + 2 * std::size_t(checkpoints_[first_slot + layer].length) * dim_;
    // All bytes are filled by validated reads before the target changes. One
    // allocation avoids per-layer initialization and a second full-prefix copy.
    auto payload = restore_buffer(offsets.back());
    for (int32_t layer = 0; layer < layers_; ++layer) {
        const auto& checkpoint = checkpoints_[first_slot + layer];
        if (!checkpoint.pages)
            continue;
        // Pages form one contiguous K prefix followed by V, without repacking.
        const auto batch = std::min(std::size_t(64), checkpoint.pages);
        for (std::size_t first = 0; first < checkpoint.pages; first += batch) {
            slots.clear();
            buffers.clear();
            for (std::size_t i = 0; i < std::min(batch, checkpoint.pages - first); ++i) {
                slots.push_back((first_slot + layer) * pages_per_layer_ + first + i);
                buffers.push_back(payload.get() + offsets[layer] + 2 * elements_ * (first + i));
            }
            storage_.read_many(slots, buffers);
        }
    }
    cache.reserve(std::max_element(begin, end, [](const auto &left, const auto &right) {
                      return left.length < right.length;
                  })->length);
    cache.reset();
    mini_infer::CPUDeviceAllocator copier;
    const auto kind = cache.device_type() == mini_infer::DeviceType::CUDA
                          ? mini_infer::MemcpyKind::CPU2CUDA
                          : mini_infer::MemcpyKind::CPU2CPU;
    for (int32_t layer = 0; layer < layers_; ++layer) {
        const auto length = checkpoints_[first_slot + layer].length;
        if (!length)
            continue;
        auto slices = cache.slice_kv_cache(layer, 0);
        const auto elements = static_cast<std::size_t>(length) * dim_;
        copier.copy(payload.get() + offsets[layer], slices.first.ptr<float>(),
                    elements * sizeof(float), kind);
        copier.copy(payload.get() + offsets[layer] + elements, slices.second.ptr<float>(),
                    elements * sizeof(float), kind);
        cache.commit(layer, 0, length);
    }
}

} // namespace infer::native
