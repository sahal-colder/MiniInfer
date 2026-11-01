#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace mini_infer {

// ============================================================
// 第 10 课：用 mmap 映射模型文件
// ============================================================
//
// 这一课把模型文件分成两层来理解：
//
//   文件格式层：ModelConfig + 后续的权重字节
//   资源管理层：MappedModelFile 拥有 mmap 返回的地址区间
//
// 本头文件只放公开的数据格式和生命周期接口；系统调用细节放在
// mini_infer_mmap.cpp，便于先读接口，再逐步追踪 open/fstat/mmap/munmap。
// ============================================================

// 第 10 课的文件头，对应 Python 的 struct.pack('<7i', ...)。
// 七个 int32_t 占 28 字节，顺序不能改变；vocab_size 可用负号编码
// “分类器权重不共享”，因此这里不能简单地要求它为正数。
struct ModelConfig {
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t layer_num = 0;
    int32_t head_num = 0;
    int32_t kv_head_num = 0;
    int32_t vocab_size = 0;
    int32_t seq_len = 0;
};

static_assert(sizeof(ModelConfig) == 28,
              "The lesson 10 model header must be 28 bytes");
static_assert(std::is_trivially_copyable<ModelConfig>::value,
              "The lesson 10 model header must support bytewise copy");
static_assert(sizeof(float) == 4,
              "The lesson 10 weight format requires 32-bit float");

// 把模型文件只读地映射到进程地址空间。
//
// 资源归属：对象拥有 mmap 区域；构造成功后文件描述符立即关闭，
// 对象析构时 munmap。不能复制，否则两个对象会重复解除同一映射。
//
// 这里仅解释课件中的“配置头 + 第一段 dim*hidden_dim 个 FP32 权重”。
// 后续真实模型的其他权重仍可通过 data()/size_bytes() 按格式解析。
class MappedModelFile {
public:
    explicit MappedModelFile(const std::string& path);
    ~MappedModelFile();

    MappedModelFile(const MappedModelFile&) = delete;
    MappedModelFile& operator=(const MappedModelFile&) = delete;

    const ModelConfig& config() const noexcept { return config_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }
    std::size_t weight_count() const noexcept { return weight_count_; }

    // 原始字节视图仅在本对象存活且模型文件未被外部改写时有效。
    // mmap 是文件的按需映射，不会提前把整个文件拷贝进用户缓冲区。
    const std::byte* data() const noexcept {
        return static_cast<const std::byte*>(mapping_);
    }

    // 第一个权重位于 data() + sizeof(ModelConfig)。
    // 用 memcpy 取一个 float，避免把原始字节直接强转为 float* 时
    // 涉及的对齐、对象生命周期和类型别名问题。
    float weight_at(std::size_t index) const;

private:
    void* mapping_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t weight_count_ = 0;
    ModelConfig config_{};
};

} // namespace mini_infer
