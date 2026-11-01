#pragma once

#include "mini_infer/base/mini_infer_memory.h"
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mini_infer {

enum class DataType { UNKNOWN = 0, FLOAT32, INT32, INT8 };
std::size_t data_type_size(DataType dtype);

// Contiguous row-major storage. Copies and views share ownership; clone copies
// data. A borrowed Tensor never extends the external allocation's lifetime.
class Tensor {
  public:
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(DataType dtype, std::vector<int32_t> dims,
           std::shared_ptr<DeviceAllocator> allocator = nullptr);
    static Tensor borrow(DataType dtype, std::vector<int32_t> dims, void* ptr, DeviceType device);
    Tensor view(std::size_t offset, std::vector<int32_t> dims) const;
    void allocate(std::shared_ptr<DeviceAllocator> allocator);
    std::size_t size() const { return size_; }
    std::size_t byte_size() const;
    DataType data_type() const { return dtype_; }
    DeviceType device_type() const;
    bool empty() const { return !buffer_; }
    const std::vector<int32_t>& dims() const { return dims_; }
    int32_t dim(std::size_t i) const { return dims_.at(i); }
    std::vector<std::size_t> strides() const;
    void reshape(std::vector<int32_t> dims);

    template <typename T> T* ptr() {
        check_type<T>();
        require_storage();
        return static_cast<T*>(data());
    }
    template <typename T> const T* ptr() const {
        check_type<T>();
        require_storage();
        return static_cast<const T*>(data());
    }
    template <typename T> T& index(std::size_t offset) {
        check_index(offset);
        return ptr<T>()[offset];
    }
    template <typename T> const T& index(std::size_t offset) const {
        check_index(offset);
        return ptr<T>()[offset];
    }

    // A non-null CUDA stream leaves completion and buffer lifetimes to the caller.
    void copy_from(const Tensor& source, cudaStream_t stream = nullptr);
    Tensor clone(std::shared_ptr<DeviceAllocator> allocator) const;
    Tensor to_cuda(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream = nullptr) const;
    Tensor to_cpu(cudaStream_t stream = nullptr) const;
    bool overlaps(const Tensor& other) const;

  private:
    template <typename T> void check_type() const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Tensor supports float, int32_t, and int8_t only");
        const auto expected = std::is_same_v<T, float>     ? DataType::FLOAT32
                              : std::is_same_v<T, int32_t> ? DataType::INT32
                                                           : DataType::INT8;
        if (dtype_ != expected)
            throw std::invalid_argument("Tensor dtype mismatch");
    }
    void* data() const { return static_cast<std::byte*>(buffer_->ptr()) + byte_offset_; }
    void require_storage() const;
    void check_index(std::size_t offset) const;
    static std::size_t checked_size(const std::vector<int32_t>& dims);
    DataType dtype_ = DataType::UNKNOWN;
    std::vector<int32_t> dims_;
    std::size_t size_ = 0;
    std::size_t byte_offset_ = 0;
    std::shared_ptr<Buffer> buffer_;
};

} // namespace mini_infer
