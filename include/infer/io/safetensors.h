#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace infer::io {
inline constexpr std::size_t page_bytes = 4096;
std::size_t align_up(std::size_t value, std::size_t alignment = page_bytes);
struct DirectAlignment { std::size_t bytes=page_bytes; bool queried=false; };
DirectAlignment direct_alignment(int fd);
struct TensorInfo {
    std::string_view name, dtype; // Views into the owning header, never payload copies.
    std::vector<std::size_t> shape;
    std::uint64_t file_offset = 0;
    std::size_t bytes = 0;
};
struct AlignedRange {
    std::uint64_t offset;
    std::size_t bytes, payload_offset;
};
// Physical read window, not a modification of standard Safetensors offsets.
AlignedRange aligned_range(std::uint64_t offset, std::size_t bytes,
                           std::size_t alignment = page_bytes);
class Safetensors {
  public:
    explicit Safetensors(const std::string& path, std::size_t max_header_bytes = 100000000);
    ~Safetensors();
    Safetensors(const Safetensors&) = delete;
    Safetensors& operator=(const Safetensors&) = delete;
    int fd() const noexcept;
    std::uint64_t file_bytes() const noexcept;
    const std::vector<TensorInfo>& tensors() const noexcept;
    const TensorInfo& tensor(std::string_view name) const;
    std::string_view metadata(std::string_view name) const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace infer::io
