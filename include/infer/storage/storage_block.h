#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace infer {

// 逻辑存储块描述一份数据；CPU/CUDA 池的内存块描述承载数据的空间。
// ID 在一个 TieredStorage 实例内单调递增、删除后不复用，0 表示无效。
// 跨实例/进程传递时还需标识所属 Storage，不能只使用这个整数。
using BlockId = std::uint64_t;
inline constexpr BlockId invalid_block_id = 0;

enum class StorageLocation { GPU, CPU, SSD, REMOTE };
enum class BlockState { RESIDENT, MOVING, EVICTED, ERROR };
enum class AccessMode { READ, WRITE };

// 只读快照，无裸地址。lease_count 只保存在这里，不再另存一份计数。
struct BlockMeta {
    BlockId id = invalid_block_id;
    StorageLocation location = StorageLocation::CPU;
    std::size_t size = 0;
    // 最后提交的内容版本。新建时为 0；finish 成功写入后递增，读取不变。
    // 原地写入被放弃时保留最后版本号，但内容失效，必须完整重写后才能读。
    std::uint64_t version = 0;
    std::size_t last_access = 0; // 单调访问序号，metadata 查询不刷新。
    std::size_t access_count = 0;
    std::size_t lease_count = 0;
    int priority = 0;
    BlockState state = BlockState::RESIDENT;
};

enum class DeviceType;
class TieredStorage;
class PagedKV;
namespace storage_detail {
struct StorageState;
struct LogicalBlockRecord;
} // namespace storage_detail

// 逻辑存储块的 move-only 访问凭证：多个 READ 可并存，WRITE 独占。
// data() 只提供只读地址，写入须用 writable_data() 并显式 finish() 提交。
// GPU 使用必须归入 acquire 时指定的 consumer stream；流须活到完成/错误清理之后。
// 仅保护本 Lease 内的地址访问，不能把裸指针缓存到 Lease 结束之后。
class BlockLease {
  public:
    BlockLease() noexcept = default;
    ~BlockLease();
    BlockLease(const BlockLease&) = delete;
    BlockLease& operator=(const BlockLease&) = delete;
    BlockLease(BlockLease&& other) noexcept;
    BlockLease& operator=(BlockLease&& other) noexcept;

    BlockId id() const noexcept;
    const void* data() const noexcept;
    void* writable_data() const; // 非 WRITE 或空 Lease 时抛 logic_error。
    std::size_t size() const noexcept;
    DeviceType device() const noexcept;
    int device_id() const noexcept; // CPU/空 Lease 返回 -1。
    explicit operator bool() const noexcept { return record_ != nullptr; }

    // 同步等待 GPU 完成；WRITE 提交新版本，然后释放访问凭证。
    // 等待失败时仍持有 Lease，可重试 finish；空 Lease 调用无操作。
    void finish();
    // 结束访问但放弃 WRITE。GPU 同样等待，失败则隔离资源并保留待清理容量。
    void reset() noexcept;

  private:
    friend class TieredStorage;
    friend class PagedKV;
    static void finish_group(const std::vector<BlockLease*>&);
    BlockLease(std::shared_ptr<storage_detail::StorageState> state,
               storage_detail::LogicalBlockRecord* record, AccessMode mode, void* data, int device,
               cudaStream_t stream, std::size_t use_index) noexcept;
    void detach(bool commit, bool completed) noexcept;

    // 保活整个管理状态，让 Lease 可以晚于 TieredStorage 外壳析构。
    std::shared_ptr<storage_detail::StorageState> state_;
    storage_detail::LogicalBlockRecord* record_ = nullptr;
    AccessMode mode_ = AccessMode::READ;
    void* data_ = nullptr;
    int device_id_ = -1;
    cudaStream_t stream_ = nullptr;
    std::size_t use_index_ = 0;
};
} // namespace infer
