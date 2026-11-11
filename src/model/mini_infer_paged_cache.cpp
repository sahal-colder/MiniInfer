#include "mini_infer/model/mini_infer_paged_cache.h"
#include "mini_infer/base/checked_math.h"
#include "infer/cuda/cuda_block.h"
#include "infer/cuda/cuda_memory_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>

namespace mini_infer {
namespace {

using Location = infer::StorageLocation;

std::size_t multiply(std::size_t a, std::size_t b) {
    return detail::checked_mul(a, b, "Paged model capacity overflow");
}

std::size_t add(std::size_t a, std::size_t b) {
    return detail::checked_add(a, b, "Paged model capacity overflow");
}

} // namespace

struct PagedModelCache::Impl {
    ModelConfig model;
    PagedModelConfig options;
    std::size_t kv_dim = 0, tokens_per_page = 0, max_pages = 0;
    infer::StorageConfig config;
    std::unique_ptr<infer::PagedKV> cache;
    infer::PagedKV::Sequence sequence;
    std::vector<int32_t> history;
    std::vector<std::uint64_t> layer_last_used;
    std::uint64_t use_clock = 0;
    std::size_t published = 0;
    bool prefix_dirty = false, prefix_clean = true;
    cudaStream_t compute = nullptr;
    void* host_table = nullptr;
    Tensor device_table;
    struct Pending {
        std::size_t layer;
        std::deque<infer::PagedKV::Ticket> jobs;
    };
    std::optional<Pending> pending;
    PagedModelStats counts;
    std::atomic<std::uint64_t> host_read_bytes{0}, host_written_bytes{0}, ssd_spilled_pages{0};
    bool reset_started = false;
    std::unique_ptr<infer::AsyncKVPipeline> pipeline;

    Impl(const ModelConfig& c, PagedModelConfig value, std::shared_ptr<DeviceAllocator> allocator)
        : model(c), options(std::move(value)), layer_last_used(c.layer_num, 0) {
        if (!allocator || allocator->device_type() != DeviceType::CUDA || c.dim <= 0 ||
            c.seq_len <= 0 || c.layer_num <= 0 || c.head_num <= 0 || c.kv_head_num <= 0 ||
            c.dim % c.head_num || c.head_num % c.kv_head_num ||
            (options.page_bytes != 64 * 1024 && options.page_bytes != 256 * 1024))
            throw std::invalid_argument(
                "Paged model needs CUDA, valid geometry and 64/256 KiB pages");
        kv_dim = multiply(c.kv_head_num, c.dim / c.head_num);
        const auto capacity = options.page_bytes / multiply(2 * sizeof(float), kv_dim);
        tokens_per_page = options.tokens_per_page;
        if (!tokens_per_page) {
            tokens_per_page = 1;
            while (tokens_per_page <= std::min<std::size_t>(16, capacity) / 2)
                tokens_per_page *= 2;
        }
        if (!capacity || !tokens_per_page || tokens_per_page > capacity ||
            multiply(tokens_per_page, kv_dim) % 4)
            throw std::invalid_argument(
                "KV tokens/page must fit and preserve 16-byte row alignment");
        max_pages = 1 + (std::size_t(c.seq_len) - 1) / tokens_per_page;
        config.block_bytes = options.page_bytes;
        check_cuda(cudaGetDevice(&config.gpu_device), "Paged model device");
        config.cpu_memory = infer::HostMemoryKind::PINNED;
        config.retain_ssd_copy = true;
        config.ssd_direct = options.ssd_direct && options.ssd_spdk.bdev_name.empty();
        config.ssd_io = options.ssd_io;
        config.ssd_spdk = options.ssd_spdk;
        config.ssd_verification = options.ssd_read_back ? infer::SSDVerification::READ_BACK
                                                        : infer::SSDVerification::ON_READ;
        config.ssd_directory = options.ssd_directory;
        const auto layer_bytes = multiply(max_pages, options.page_bytes);
        const auto resident_layers =
            options.resident_layers ? std::min<std::size_t>(c.layer_num, options.resident_layers)
                                    : std::size_t(c.layer_num);
        config.budget.gpu_bytes =
            options.gpu_bytes ? options.gpu_bytes : multiply(resident_layers, layer_bytes);
        config.budget.cpu_bytes =
            options.host_bytes ? options.host_bytes
                               : multiply(add(c.layer_num, 2 * max_pages), options.page_bytes);
        config.budget.ssd_bytes = options.ssd_bytes ? options.ssd_bytes
                                                    : multiply(multiply(c.layer_num, max_pages + 1),
                                                               options.page_bytes + 4096);
        if (config.budget.cpu_bytes < multiply(c.layer_num, options.page_bytes) ||
            (!options.enable_async_kv_pipeline && config.budget.gpu_bytes < options.page_bytes))
            throw std::invalid_argument(
                "KV GPU budget needs one page; host budget needs one all-layer page group");
        if (options.enable_async_kv_pipeline) config.budget.gpu_bytes = 0;
        cache = std::make_unique<infer::PagedKV>(config, c.layer_num, tokens_per_page,
                                                 "llama-model-instance/f32/rope10000/causal");
        if (options.enable_async_kv_pipeline) {
            bool gpu_resident = options.resident_layers == 0;
            if (gpu_resident && options.gpu_bytes) {
                const auto slot_bytes = add(layer_bytes, multiply(max_pages, sizeof(void*)));
                const auto reserved_bytes =
                    infer::cuda_pool_detail::allocation_bucket(slot_bytes, {}).second;
                gpu_resident = options.gpu_bytes >= multiply(c.layer_num, reserved_bytes);
            }
            pipeline = std::make_unique<infer::AsyncKVPipeline>(
                infer::AsyncKVPipeline::Config{
                    std::size_t(c.layer_num), max_pages, options.page_bytes, tokens_per_page,
                    kv_dim * sizeof(float), config.gpu_device, options.gpu_bytes, gpu_resident},
                [this](const infer::LayerRead& job) { read_layer(job); },
                [this](std::size_t layer, std::size_t position, std::size_t count,
                       const std::byte* key, const std::byte* value) {
                    write_rows(layer, position, count, key, value);
                });
            return;
        }
        try {
            check_cuda(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking),
                       "Paged compute stream");
            check_cuda(cudaHostAlloc(&host_table, multiply(max_pages, sizeof(void*)),
                                     cudaHostAllocPortable),
                       "Paged pinned table");
            const auto table_bytes = multiply(max_pages, sizeof(void*));
            if (table_bytes > std::size_t(INT32_MAX))
                throw std::length_error("KV page table exceeds Tensor dimensions");
            device_table = Tensor(DataType::INT8, {static_cast<int32_t>(table_bytes)}, allocator);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // Sequence ownership stays on the scheduler until all I/O callbacks retire.
    void read_layer(const infer::LayerRead& job) {
        constexpr std::size_t batch_size = 64;
        std::vector<infer::BlockId> ids;
        std::vector<void*> destinations;
        ids.reserve(std::min(job.pages, batch_size));
        destinations.reserve(ids.capacity());
        for (std::size_t first = 0; first < job.pages; first += batch_size) {
            const auto end = std::min(first + batch_size, job.pages);
            ids.clear();
            destinations.clear();
            for (auto page = first; page < end; ++page) {
                const auto meta = cache->metadata(sequence, page, job.layer);
                if (meta.location == Location::CPU)
                    host_read_bytes.fetch_add(options.page_bytes, std::memory_order_relaxed);
                ids.push_back(meta.id);
                destinations.push_back(job.destination + page * options.page_bytes);
            }
            cache->storage().read_to_host_many(ids, destinations, options.page_bytes);
        }
    }

    void write_rows(std::size_t layer, std::size_t first, std::size_t count, const std::byte* key,
                    const std::byte* value) {
        const auto row_bytes = kv_dim * sizeof(float);
        for (std::size_t written = 0; written < count;) {
            const auto position = first + written;
            const auto page = position / tokens_per_page;
            const auto row = position % tokens_per_page;
            const auto rows = std::min(count - written, tokens_per_page - row);
            // Admission may have spilled a destination page. Restore it once
            // per span and keep one write lease for both K and V.
            if (cache->metadata(sequence, page, layer).location == Location::SSD) {
                const auto fits = [&] {
                    return cache->storage().stats(Location::CPU).reserved_bytes <=
                           config.budget.cpu_bytes - options.page_bytes;
                };
                for (std::size_t victim = 0; victim < layer_last_used.size() && !fits(); ++victim) {
                    const auto pages = pages_at(victim, Location::CPU);
                    if (pages.empty()) continue;
                    cache->prefetch_pages(sequence, victim, Location::SSD, {pages.front()}).wait();
                    ssd_spilled_pages.fetch_add(1, std::memory_order_relaxed);
                }
                cache->prefetch_pages(sequence, layer, Location::CPU, {page}).wait();
            }
            auto lease = cache->acquire(sequence, page, layer, infer::AccessMode::WRITE);
            auto* dest = static_cast<std::byte*>(lease.writable_data()) + row * row_bytes;
            std::memcpy(dest, key + written * row_bytes, rows * row_bytes);
            std::memcpy(dest + tokens_per_page * row_bytes, value + written * row_bytes,
                        rows * row_bytes);
            lease.finish();
            host_written_bytes.fetch_add(2 * rows * row_bytes, std::memory_order_relaxed);
            written += rows;
        }
    }

    void wait_for_migrations() {
        if (!pending) return;
        const auto start = std::chrono::steady_clock::now();
        std::exception_ptr error;
        while (!pending->jobs.empty()) {
            auto job = std::move(pending->jobs.front());
            pending->jobs.pop_front();
            try {
                job.wait();
            } catch (...) {
                if (!error) error = std::current_exception();
            }
        }
        pending.reset();
        counts.prefetch_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
        if (error) std::rethrow_exception(error);
    }

    void cleanup() noexcept {
        try {
            wait_for_migrations();
        } catch (...) {
            pending.reset();
        }
        if (compute) cudaStreamSynchronize(compute);
        if (host_table) {
            cudaFreeHost(host_table);
            host_table = nullptr;
        }
        if (compute) {
            cudaStreamDestroy(compute);
            compute = nullptr;
        }
    }
    ~Impl() {
        pipeline.reset(); // Its worker borrows sequence and storage.
        cleanup();
    }
    std::vector<std::size_t> pages_at(std::size_t layer, Location source) const {
        std::vector<std::size_t> result;
        for (std::size_t p = 0; p < sequence.page_count(); ++p) {
            const auto meta = cache->metadata(sequence, p, layer);
            if (meta.location == source && meta.state == infer::BlockState::RESIDENT)
                result.push_back(p);
        }
        return result;
    }
    std::vector<std::size_t> pages_outside_gpu(std::size_t layer) const {
        std::vector<std::size_t> result;
        for (std::size_t p = 0; p < sequence.page_count(); ++p)
            if (cache->metadata(sequence, p, layer).location != Location::GPU) result.push_back(p);
        return result;
    }

    void migrate_layer_pages(std::size_t layer, Location source, Location target) {
        const auto indices = pages_at(layer, source);
        for (std::size_t first = 0; first < indices.size(); first += 64) {
            const std::vector<std::size_t> group(
                indices.begin() + first, indices.begin() + std::min(indices.size(), first + 64));
            auto ticket = cache->prefetch_pages(sequence, layer, target, group);
            ticket.wait();
            if (source == Location::GPU) counts.gpu_evicted_pages += group.size();
            if (target == Location::CPU) host_written_bytes += group.size() * options.page_bytes;
            if (target == Location::SSD)
                ssd_spilled_pages.fetch_add(group.size(), std::memory_order_relaxed);
        }
    }

    void discard_inactive_prefixes() {
        if (prefix_clean) return;
        counts.prefix_evictions += cache->evict_prefixes();
        prefix_dirty = true;
        prefix_clean = true;
    }

    std::vector<std::size_t> layers_by_last_use(std::size_t exclude_a,
                                                std::size_t exclude_b) const {
        std::vector<std::size_t> result;
        for (std::size_t layer = 0; layer < layer_last_used.size(); ++layer)
            if (layer != exclude_a && layer != exclude_b) result.push_back(layer);
        std::stable_sort(result.begin(), result.end(),
                         [&](auto a, auto b) { return layer_last_used[a] < layer_last_used[b]; });
        return result;
    }

    bool ensure_host_room(std::size_t bytes, std::size_t exclude_a, std::size_t exclude_b) {
        if (bytes > config.budget.cpu_bytes) return false;
        const auto fits = [&] {
            return cache->storage().stats(Location::CPU).reserved_bytes <=
                   config.budget.cpu_bytes - bytes;
        };
        if (fits()) return true;
        discard_inactive_prefixes();
        for (auto layer : layers_by_last_use(exclude_a, exclude_b)) {
            if (fits()) break;
            migrate_layer_pages(layer, Location::CPU, Location::SSD);
        }
        return fits();
    }

    bool ensure_gpu_room(std::size_t target, std::size_t protected_layer, bool prefetch_only) {
        const auto required = multiply(pages_outside_gpu(target).size(), options.page_bytes);
        const auto full = multiply(sequence.page_count(), options.page_bytes);
        if (full > config.budget.gpu_bytes) {
            if (prefetch_only) return false;
            throw std::length_error("GPU KV budget cannot hold one layer of the current context");
        }
        if (prefetch_only && protected_layer < layer_last_used.size()) {
            const auto protected_bytes =
                multiply(pages_at(protected_layer, Location::GPU).size(), options.page_bytes);
            if (full > config.budget.gpu_bytes - protected_bytes) return false;
        }
        const auto fits = [&] {
            return cache->storage().stats(Location::GPU).reserved_bytes <=
                   config.budget.gpu_bytes - required;
        };
        if (fits()) return true;
        // Speculation must not synchronously evict another layer on the compute thread.
        if (prefetch_only) return false;
        discard_inactive_prefixes();
        // Layers execute in cyclic order. Evict the layer whose next use is farthest.
        auto victim = target;
        for (std::size_t remaining = layer_last_used.size() - 1; remaining && !fits(); --remaining) {
            victim = victim == 0 ? layer_last_used.size() - 1 : victim - 1;
            if (victim == protected_layer) continue;
            const auto bytes = multiply(pages_at(victim, Location::GPU).size(), options.page_bytes);
            if (!bytes) continue;
            const bool in_host = ensure_host_room(bytes, target, protected_layer);
            migrate_layer_pages(victim, Location::GPU, in_host ? Location::CPU : Location::SSD);
        }
        return fits();
    }

    void submit_layer_to_gpu(std::size_t layer, bool ahead) {
        auto indices = pages_outside_gpu(layer);
        for (const auto page : indices)
            if (cache->metadata(sequence, page, layer).location == Location::CPU)
                host_read_bytes += options.page_bytes;
        if (ahead)
            counts.prefetched_pages += indices.size();
        else
            counts.demand_fault_pages += indices.size();
        pending.emplace(Pending{layer, {}});
        try {
            for (std::size_t first = 0; first < indices.size(); first += 64) {
                if (pending->jobs.size() == config.max_async_migrations) {
                    pending->jobs.front().wait();
                    pending->jobs.pop_front();
                }
                const std::vector<std::size_t> group(indices.begin() + first,
                                                     indices.begin() +
                                                         std::min(indices.size(), first + 64));
                pending->jobs.push_back(
                    cache->prefetch_pages(sequence, layer, Location::GPU, group));
                if (ahead) ++counts.prefetch_windows;
            }
        } catch (...) {
            try {
                wait_for_migrations();
            } catch (...) {
            }
            throw;
        }
        if (!ahead) wait_for_migrations();
    }
};

PagedModelCache::PagedModelCache(const ModelConfig& c, PagedModelConfig options,
                                 std::shared_ptr<DeviceAllocator> allocator)
    : impl_(std::make_unique<Impl>(c, std::move(options), std::move(allocator))) {}

PagedModelCache::~PagedModelCache() = default;

std::size_t PagedModelCache::begin_request(const int32_t* input, std::size_t count) {
    auto& state = *impl_;
    if (!state.history.empty()) throw std::logic_error("Paged request was not reset");
    state.wait_for_migrations();
    state.prefix_clean = false;
    state.sequence = state.options.prefix_cache
                         ? state.cache->match(std::vector<int32_t>(input, input + count - 1))
                         : infer::PagedKV::Sequence{};
    state.history.assign(input, input + state.sequence.matched_tokens);
    state.counts.last_prefix_hit_tokens = state.sequence.matched_tokens;
    state.counts.prefix_hit_tokens += state.sequence.matched_tokens;
    state.published = state.sequence.page_count();
    return state.sequence.matched_tokens;
}

void PagedModelCache::begin_token(int32_t token, int32_t position) {
    begin_batch(&token, 1, position);
}

void PagedModelCache::begin_batch(const int32_t* input, std::size_t count, int32_t first_position) {
    auto& state = *impl_;
    for (std::size_t i = 0; i < count; ++i) {
        const auto position = first_position + static_cast<int32_t>(i);
        if (position < 0 || std::size_t(position) != state.history.size())
            throw std::logic_error("Non-contiguous paged token admission");
        if (std::size_t(position) % state.tokens_per_page == 0) {
            state.wait_for_migrations();
            const auto no_protected_layer = state.layer_last_used.size();
            if (!state.ensure_host_room(multiply(state.model.layer_num, state.options.page_bytes),
                                        no_protected_layer, no_protected_layer))
                throw std::bad_alloc();
            state.cache->append_page(state.sequence, Location::CPU);
            const auto page = state.sequence.page_count() - 1;
            for (int layer = 0; layer < state.model.layer_num; ++layer) {
                auto lease =
                    state.cache->acquire(state.sequence, page, layer, infer::AccessMode::WRITE);
                std::memset(lease.writable_data(), 0, lease.bytes());
                lease.finish();
            }
        }
        state.history.push_back(input[i]);
        ++state.counts.computed_tokens;
    }
    if (state.pipeline) state.pipeline->begin_token(first_position, count);
}

PagedModelCache::Layer PagedModelCache::prepare_layer(std::size_t layer, int32_t position,
                                                      int32_t first_position) {
    auto& state = *impl_;
    state.wait_for_migrations();
    if (!state.ensure_gpu_room(layer, state.layer_last_used.size(), /*prefetch_only=*/false))
        throw std::bad_alloc();
    if (!state.pages_outside_gpu(layer).empty()) state.submit_layer_to_gpu(layer, /*ahead=*/false);
    state.layer_last_used.at(layer) = ++state.use_clock;
    ++state.counts.layer_calls;
    Layer result;
    result.page_count = state.sequence.page_count();
    result.tokens_per_page = state.tokens_per_page;
    result.leases.reserve(result.page_count);
    auto** host = static_cast<const float**>(state.host_table);
    const auto first_write_page =
        std::size_t(first_position < 0 ? position : first_position) / state.tokens_per_page;
    const auto last_write_page = std::size_t(position) / state.tokens_per_page;
    const auto row_offset = std::size_t(position) % state.tokens_per_page * state.kv_dim;
    // Prefill may write several pages. The returned K/V views point into the final one.
    for (std::size_t page = 0; page < result.page_count; ++page) {
        const bool writable = page >= first_write_page && page <= last_write_page;
        result.leases.push_back(state.cache->acquire(
            state.sequence, page, layer,
            writable ? infer::AccessMode::WRITE : infer::AccessMode::READ, state.compute));
        host[page] = static_cast<const float*>(result.leases.back().data());
        if (writable) {
            auto* base = static_cast<float*>(result.leases.back().writable_data());
            result.key = Tensor::borrow(DataType::FLOAT32, {static_cast<int32_t>(state.kv_dim)},
                                        base + row_offset, DeviceType::CUDA);
            result.value = Tensor::borrow(DataType::FLOAT32, {static_cast<int32_t>(state.kv_dim)},
                                          base + state.tokens_per_page * state.kv_dim + row_offset,
                                          DeviceType::CUDA);
        }
    }
    check_cuda(cudaMemcpyAsync(state.device_table.ptr<int8_t>(), state.host_table,
                               result.page_count * sizeof(void*), cudaMemcpyHostToDevice,
                               state.compute),
               "Paged table upload");
    result.pages = reinterpret_cast<const float* const*>(state.device_table.ptr<int8_t>());
    return result;
}

void PagedModelCache::prefetch_next(std::size_t layer, std::size_t protected_layer) {
    auto& state = *impl_;
    if (!state.options.prefetch || layer >= state.layer_last_used.size()) return;
    if (state.pending) throw std::logic_error("Previous layer prefetch was not consumed");
    if (state.pages_outside_gpu(layer).empty()) return;
    if (!state.ensure_gpu_room(layer, protected_layer, /*prefetch_only=*/true)) {
        ++state.counts.prefetch_skipped;
        return;
    }
    state.submit_layer_to_gpu(layer, /*ahead=*/true);
}

void PagedModelCache::finish_step() {
    auto& state = *impl_;
    state.wait_for_migrations();
    const auto complete = state.history.size() / state.tokens_per_page;
    if (state.options.prefix_cache && (complete > state.published || state.prefix_dirty)) {
        if (state.pipeline) state.pipeline->flush_resident();
        state.cache->publish_prefix(state.sequence, state.history, complete);
        state.published = complete;
        state.prefix_dirty = false;
    }
}

bool PagedModelCache::reset(bool discard_prefix) {
    auto& state = *impl_;
    const auto clear_sequence = [&state] {
        state.sequence = infer::PagedKV::Sequence{};
        state.history.clear();
        state.published = 0;
        state.counts.last_prefix_hit_tokens = 0;
    };
    if (state.pipeline) {
        if (!state.reset_started) {
            state.pipeline->begin_retire();
            state.reset_started = true;
        }
        if (!state.pipeline->poll_retired()) return false;
        state.reset_started = false;
        clear_sequence();
        if (discard_prefix) state.cache->evict_prefixes();
        return true;
    }
    std::exception_ptr error;
    try {
        state.wait_for_migrations();
    } catch (...) {
        error = std::current_exception();
    }
    const auto completed = cudaStreamSynchronize(state.compute);
    clear_sequence();
    if (discard_prefix || error) state.cache->evict_prefixes();
    check_cuda(completed, "Reset paged consumers");
    if (error) std::rethrow_exception(error);
    return true;
}

infer::AsyncKVPipeline* PagedModelCache::pipeline() const noexcept { return impl_->pipeline.get(); }
void PagedModelCache::record_async_token() noexcept {
    impl_->counts.layer_calls += impl_->layer_last_used.size();
}
cudaStream_t PagedModelCache::stream() const noexcept {
    return impl_->pipeline ? impl_->pipeline->compute_stream() : impl_->compute;
}

PagedModelStats PagedModelCache::stats() const {
    const auto& state = *impl_;
    auto result = state.counts;
    result.ssd_spilled_pages = state.ssd_spilled_pages.load(std::memory_order_relaxed);
    const auto gpu = state.cache->storage().stats(Location::GPU);
    const auto host = state.cache->storage().stats(Location::CPU);
    const auto disk = state.cache->storage().stats(Location::SSD);
    const auto io = state.cache->storage().io_stats(Location::SSD);
    result.page_bytes = state.options.page_bytes;
    result.tokens_per_page = state.tokens_per_page;
    result.pages = state.sequence.page_count();
    result.gpu_limit_bytes = gpu.limit_bytes;
    result.host_limit_bytes = host.limit_bytes;
    result.ssd_limit_bytes = disk.limit_bytes;
    result.gpu_reserved_bytes = gpu.reserved_bytes;
    result.host_reserved_bytes = host.reserved_bytes;
    result.ssd_reserved_bytes = disk.reserved_bytes;
    result.gpu_peak_bytes = gpu.peak_reserved_bytes;
    result.host_peak_bytes = host.peak_reserved_bytes;
    result.ssd_peak_bytes = disk.peak_reserved_bytes;
    if (state.pipeline) {
        result.gpu_reserved_bytes = result.gpu_peak_bytes = state.pipeline->reserved_device_bytes();
        result.gpu_limit_bytes =
            state.options.gpu_bytes ? state.options.gpu_bytes : result.gpu_reserved_bytes;
        result.staging_bytes = state.pipeline->staging_bytes();
    } else {
        result.staging_bytes = state.cache->storage().staging_stats().reserved_bytes +
                               state.max_pages * sizeof(void*) + 4096;
    }
    result.host_read_bytes = state.host_read_bytes.load();
    result.host_written_bytes = state.host_written_bytes.load();
    result.ssd_read_bytes = io.payload_read_bytes;
    result.ssd_written_bytes = io.payload_written_bytes;
    result.ssd_record_writes = io.record_writes;
    result.uring_submissions = io.file_io.submissions;
    result.ssd_device = io.device;
    return result;
}
} // namespace mini_infer
