#pragma once
#include "infer/io.h"
#include "infer/mem.h"
#include "infer/storage/storage_backend.h"
#include "infer/storage/ssd_device.h"
#include "infer/storage/tiered_storage.h"
#include <cstdio>
#include <mutex>
#include <system_error>

namespace infer_c {
inline int error() noexcept {
    try { throw; }
    catch (const infer::StorageBusy&) { return -EBUSY; }
    catch (const std::bad_alloc&) { return -ENOMEM; }
    catch (const std::out_of_range&) { return -ENOENT; }
    catch (const std::length_error&) { return -ENOBUFS; }
    catch (const std::invalid_argument&) { return -EINVAL; }
    catch (const std::overflow_error&) { return -EOVERFLOW; }
    catch (const std::system_error& e) { return -e.code().value(); }
    catch (const std::logic_error&) { return -EBUSY; }
    catch (...) { return -EIO; }
}
template<class F> int status(F f) noexcept {
    try { return f(); } catch (...) { return error(); }
}
inline void require(int rc) {
    if (rc < 0) throw std::system_error(-rc, std::generic_category());
}

/* Physical host provider, not a logical Block registry. The backend owns a
 * shared reference because a Lease may outlive its outer C runtime handle. */
class HostMemory final : public infer::DeviceAllocator {
    mutable std::mutex mutex;
    bool retiring = false;
public:
    if_pool *pool = nullptr;
    explicit HostMemory(const if_upstream *up) : DeviceAllocator(infer::DeviceType::CPU) {
        auto cfg = if_pool_cfg_default();
        cfg.max_idle = SIZE_MAX;
        require(if_pool_create(up, &cfg, &pool));
    }
    ~HostMemory() override {
        if (pool && if_pool_destroy(pool))
            std::fputs("infer: host provider cleanup failed; allocation retained\n", stderr);
    }
    void *allocate(size_t bytes, cudaStream_t = nullptr) const override {
        std::lock_guard<std::mutex> lock(mutex);
        void *p = nullptr;
        require(if_pool_alloc(pool, bytes, &p));
        return p;
    }
    void release(void *p, cudaStream_t = nullptr) const override {
        std::lock_guard<std::mutex> lock(mutex);
        require(if_pool_free(pool, p));
    }
    int close() {
        std::lock_guard<std::mutex> lock(mutex);
        return close_locked();
    }
    // Late leases keep this provider reachable after the outer runtime closes.
    // Active core allocations defer teardown; an upstream error remains retryable.
    int retire() {
        std::lock_guard<std::mutex> lock(mutex);
        retiring = true;
        int rc = close_locked();
        return rc == -EBUSY ? 0 : rc;
    }
    int finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!retiring) return 0;
        int rc = close_locked();
        return rc == -EBUSY ? 0 : rc;
    }
private:
    int close_locked() {
        if (!pool) return 0;
        int rc = if_pool_destroy(pool);
        if (!rc) pool = nullptr;
        return rc;
    }
};

/* SSD byte I/O injection. The existing SSDStorageBackend owns extent allocation
 * and record verification. This object only borrows the caller's device. */
class Device final : public infer::SSDDevice {
    if_dev *dev;
public:
    explicit Device(if_dev *d) : dev(d) {}
    size_t alignment() const noexcept override { return 64; }
    uint64_t capacity_bytes() const noexcept override { return dev->size; }
    void read(void *buf, size_t bytes, uint64_t off) override {
        if_io q{nullptr,0,buf,bytes,off};
        require(if_dev_read(dev,&q,1));
    }
    void write(const void *buf, size_t bytes, uint64_t off) override {
        if_io q{nullptr,0,const_cast<void *>(buf),bytes,off};
        require(if_dev_write(dev,&q,1));
    }
    void read_many(const std::vector<infer::FileIORequest>& requests) override {
        for (const auto& q : requests) read(q.data,q.bytes,q.offset);
    }
    void write_many(const std::vector<infer::FileIORequest>& requests) override {
        std::vector<if_io> ranges;
        ranges.reserve(requests.size());
        for (const auto& q : requests)
            ranges.push_back({const_cast<void*>(q.prefix),q.prefix_bytes,q.data,q.bytes,q.offset});
        require(if_dev_write(dev,ranges.data(),ranges.size()));
    }
    infer::SSDDeviceStats stats() const override { return {}; }
};
inline std::unique_ptr<infer::StorageBackend> cpu(size_t bytes, const std::shared_ptr<HostMemory>& pool) {
    return std::make_unique<infer::CPUStorageBackend>(bytes,pool);
}
inline std::unique_ptr<infer::StorageBackend> disk(size_t bytes,size_t budget,if_dev *dev,
                                                size_t piece_bytes = 0) {
    if (!dev) return {};
    return std::make_unique<infer::SSDStorageBackend>(bytes,budget,std::make_unique<Device>(dev),
        infer::SSDVerification::ON_READ,piece_bytes);
}
}
