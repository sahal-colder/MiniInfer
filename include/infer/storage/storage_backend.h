#pragma once

#include "infer/core/allocator.h"
#include "infer/storage/file_io.h"
#include "infer/storage/ssd_device.h"
#include "infer/storage/storage_block.h"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infer {

// 后端分配句柄仅定位某次物理分配。迁移会换句柄，逻辑 BlockId 保持不变。
using StorageHandle = std::uint64_t;
enum class HostMemoryKind { PAGEABLE, PINNED };
// 两种模式都在消费时校验身份和 CRC；READ_BACK 另在发布前完整回读。
enum class SSDVerification { READ_BACK, ON_READ };
struct StorageMemoryView {
    void* data = nullptr;
    std::size_t capacity = 0;
    DeviceType device = DeviceType::UNKNOWN;
    int device_id = -1;
};
struct BackendPoolStats {
    std::size_t active_bytes = 0;
    std::size_t cached_bytes = 0;
    std::size_t pending_bytes = 0;
    std::size_t reserved_bytes = 0;
    bool available = true; // 外部 allocator 没有统计接口时为 false。
};
struct BackendIOStats {
    std::uint64_t reserve_ns = 0, read_ns = 0, write_ns = 0, publish_ns = 0;
    FileIOStats file_io;
    std::uint64_t payload_written_bytes = 0, payload_read_bytes = 0, verification_read_bytes = 0;
    SSDDeviceStats device;
    std::uint64_t record_writes = 0, source_crc_bytes = 0;
};
struct StorageRead {
    StorageHandle handle;
    BlockMeta expected;
    void* host;
};
struct StorageWrite {
    StorageHandle handle;
    BlockMeta meta;
    const void* host;
};

// 介质操作边界，不管理 BlockId、Lease 或淘汰策略。所有方法禁止重入 Storage。
// release 失败须保留原分配；wait 成功表示此前提交给 consumer stream 的工作已完成。
class StorageBackend {
  public:
    virtual ~StorageBackend() = default;
    virtual StorageLocation location() const noexcept = 0;
    virtual std::size_t slot_bytes() const noexcept = 0;
    virtual bool host_is_pinned() const noexcept { return false; }
    // 预算按真实记录容量计费；SSD 在 payload 槽之外还包含记录头。
    virtual std::size_t allocation_bytes() const noexcept { return slot_bytes(); }
    virtual std::size_t allocation_bytes(std::size_t) const { return allocation_bytes(); }
    virtual StorageHandle reserve() = 0;
    virtual StorageHandle reserve(std::size_t) { return reserve(); }
    virtual void release(StorageHandle handle) = 0;
    // CPU/GPU views are nonempty and match allocation_bytes(payload size).
    // Fixed backends return slot_bytes; variable CPU slots return their size class.
    // 地址只借用；调用方须保护 handle 的生命周期，且 CPU/Pinned 使用前须确认 DMA 完成。
    virtual StorageMemoryView memory_view(StorageHandle handle) const = 0;
    virtual void read(StorageHandle handle, std::size_t offset, void* host, std::size_t bytes) = 0;
    virtual void write(StorageHandle handle, std::size_t offset, const void* host,
                       std::size_t bytes) = 0;
    virtual void validate_stream(cudaStream_t stream) const = 0;
    virtual void wait(cudaStream_t stream) const = 0;
    // 搬运流独立于 Lease 的 consumer stream；失败后必须先确认 DMA 完成再复用中转槽。
    virtual void wait_transfer() const {
        // 自定义 GPU 后端必须明确提供恢复等待，默认流不能替代它自己的 DMA 流。
        if (location() == StorageLocation::GPU)
            throw std::logic_error("GPU backend must implement transfer completion recovery");
        wait(nullptr);
    }
    // 内存后端无额外记录头；文件后端在发布/读取时校验身份、长度和内容。
    virtual void publish(StorageHandle, const BlockMeta&) {}
    virtual void write_published(StorageHandle handle, const BlockMeta& meta, const void* host) {
        write(handle, 0, host, meta.size);
        publish(handle, meta);
    }
    virtual void validate(StorageHandle, const BlockMeta&) const {}
    // 完整块迁移：读取的同时核对逻辑身份。默认实现兼容已有自定义后端，
    // SSD 覆盖此方法，将 validate + read 合并为一遍完整读取。
    virtual void read_validated(StorageHandle handle, const BlockMeta& expected, void* host) {
        validate(handle, expected);
        read(handle, 0, host, expected.size);
    }
    virtual BackendPoolStats pool_stats() const = 0;
    virtual BackendIOStats io_stats() const { return {}; }
    virtual void read_validated_many(const std::vector<StorageRead>& requests) {
        for (const auto& request : requests)
            read_validated(request.handle, request.expected, request.host);
    }
};

class CPUStorageBackend final : public StorageBackend {
  public:
    using StorageBackend::allocation_bytes;
    static std::size_t allocation_size(std::size_t bytes);
    // 默认同尺寸普通 CPU 池；PINNED 显式复用已有锁页池。注入 allocator 必须匹配 kind。
    // 注入 allocator 时，其内部额外预留量不属于 Storage 预算。
    explicit CPUStorageBackend(std::size_t slot_bytes,
                               std::shared_ptr<DeviceAllocator> allocator = {},
                               HostMemoryKind kind = HostMemoryKind::PAGEABLE,
                               bool variable_slots = false);
    ~CPUStorageBackend() override;
    StorageLocation location() const noexcept override { return StorageLocation::CPU; }
    std::size_t slot_bytes() const noexcept override;
    bool host_is_pinned() const noexcept override;
    StorageHandle reserve() override;
    StorageHandle reserve(std::size_t bytes) override;
    std::size_t allocation_bytes(std::size_t bytes) const override;
    void release(StorageHandle handle) override;
    StorageMemoryView memory_view(StorageHandle handle) const override;
    void read(StorageHandle handle, std::size_t offset, void* host, std::size_t bytes) override;
    void write(StorageHandle handle, std::size_t offset, const void* host,
               std::size_t bytes) override;
    void validate_stream(cudaStream_t stream) const override;
    void wait(cudaStream_t stream) const override;
    BackendPoolStats pool_stats() const override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class GPUStorageBackend final : public StorageBackend {
  public:
    GPUStorageBackend(std::size_t slot_bytes, std::size_t max_reserved_bytes, int device_id);
    ~GPUStorageBackend() override;
    StorageLocation location() const noexcept override { return StorageLocation::GPU; }
    std::size_t slot_bytes() const noexcept override;
    StorageHandle reserve() override;
    void release(StorageHandle handle) override;
    StorageMemoryView memory_view(StorageHandle handle) const override;
    // host 必须是 pinned 内存，且范围有效；同步返回后才可覆盖/释放。
    // 出现 CUDA 完成错误时，调用方须保留 host，直到设备恢复并确认完成。
    void read(StorageHandle handle, std::size_t offset, void* host, std::size_t bytes) override;
    void write(StorageHandle handle, std::size_t offset, const void* host,
               std::size_t bytes) override;
    // GPU Lease 采用显式 consumer stream；调用者负责其他 stream 到该流的依赖。
    void validate_stream(cudaStream_t stream) const override;
    void wait(cudaStream_t stream) const override;
    void wait_transfer() const override;
    BackendPoolStats pool_stats() const override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 进程内块记录及 CRC：默认使用初始化后 unlink 的私有文件，也可拥有一个 SSDDevice。
// 两种介质共用发布/校验规则，不提供重启恢复或持久化保证。
class SSDStorageBackend final : public StorageBackend {
  public:
    static constexpr std::size_t record_header_bytes = 64;
    static std::size_t allocation_size(std::size_t slot_bytes);
    static std::size_t allocation_size(std::size_t slot_bytes, std::size_t alignment);
    SSDStorageBackend(std::size_t slot_bytes, std::size_t max_reserved_bytes,
                      const std::string& directory = "/tmp", FileIOConfig io = {},
                      SSDVerification verification = SSDVerification::READ_BACK);
    SSDStorageBackend(std::size_t slot_bytes, std::size_t max_reserved_bytes,
                      std::unique_ptr<SSDDevice> device,
                      SSDVerification verification = SSDVerification::READ_BACK,
                      std::size_t read_piece_bytes = 0);
    ~SSDStorageBackend() override;
    StorageLocation location() const noexcept override { return StorageLocation::SSD; }
    std::size_t slot_bytes() const noexcept override;
    std::size_t allocation_bytes() const noexcept override;
    StorageHandle reserve() override;
    void release(StorageHandle handle) override;
    StorageMemoryView memory_view(StorageHandle handle) const override; // data 始终为空。
    void read(StorageHandle handle, std::size_t offset, void* host, std::size_t bytes) override;
    // 从 offset=0 开始顺序写入，后续只能追加；publish 前不可读。
    void write(StorageHandle handle, std::size_t offset, const void* host,
               std::size_t bytes) override;
    void publish(StorageHandle handle, const BlockMeta& meta) override;
    void write_published(StorageHandle handle, const BlockMeta& meta, const void* host) override;
    void write_published_many(const std::vector<StorageWrite>& requests);
    void validate(StorageHandle handle, const BlockMeta& meta) const override;
    void read_validated(StorageHandle handle, const BlockMeta& expected, void* host) override;
    void validate_stream(cudaStream_t stream) const override;
    void wait(cudaStream_t stream) const override;
    BackendPoolStats pool_stats() const override;
    BackendIOStats io_stats() const override;
    void read_validated_many(const std::vector<StorageRead>& requests) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace storage_detail {
void validate_slot_bytes(std::size_t bytes);
StorageHandle next_storage_handle();
void validate_io(std::size_t capacity, std::size_t offset, const void* host, std::size_t bytes);
std::uint32_t crc32c(const void* data, std::size_t bytes);
} // namespace storage_detail
} // namespace infer
