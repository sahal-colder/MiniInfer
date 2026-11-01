// 声明 Buffer 的存储所有权、移动语义及释放规则。
#pragma once

#include "infer/core/allocator.h"

namespace infer {

// Pinned host Buffer 的使用约束：
// 读取 D2H 结果、覆盖 H2D 源数据、释放或移动赋值覆盖旧存储前，
// 调用者须等待相应传输完成；析构不会自动等待，也不会延迟到 stream 中释放。
// CUDA Buffer 的使用约束：
// 1. stream 必须活到 Buffer 析构之后；先析构 Buffer，再同步并销毁 stream。
// 2. ASYNC 分配、使用、释放在同一 stream 上依靠流顺序保证安全。
// 3. 跨 stream 时，使用方须等待分配/写入完成，释放方须等待所有使用完成。
//    调用者用 event/wait 或显式同步建立依赖，Buffer 不自动管理这些依赖。
class Buffer {
  public:
    Buffer(std::size_t size, std::shared_ptr<DeviceAllocator> allocator,
           cudaStream_t stream = nullptr);
    Buffer(std::size_t size, void* ptr, DeviceType device_type);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void* ptr() const { return ptr_; }
    std::size_t size() const { return size_; }
    DeviceType device_type() const { return device_type_; }
    bool is_external() const { return is_external_; }

  private:
    void release_owned_buffer() noexcept;
    void move_from(Buffer&& other) noexcept;

    std::size_t size_ = 0;
    void* ptr_ = nullptr;
    bool is_external_ = false;
    DeviceType device_type_ = DeviceType::UNKNOWN;
    std::shared_ptr<DeviceAllocator> allocator_;
    cudaStream_t stream_ = nullptr;
};

} // namespace infer
