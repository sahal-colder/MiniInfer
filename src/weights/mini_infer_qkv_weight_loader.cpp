#include "mini_infer/weights/mini_infer_qkv_weight_loader.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_infer {
namespace {

// 文件头不是可信的内存大小。乘法/加法溢出会让“大权重”绕过短文件检查，
// 所以先做上界检查，再运算。size_t 也不能防止自身溢出。
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("QKV weight layout multiplication overflow");
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::overflow_error("QKV weight layout addition overflow");
    }
    return a + b;
}

}  // namespace

QKVWeightLayout make_qkv_weight_layout(const ModelConfig& config) {
    if (config.dim <= 0 || config.hidden_dim <= 0 || config.layer_num <= 0 ||
        config.seq_len <= 0 || config.vocab_size == 0 || config.head_num <= 0 ||
        config.kv_head_num <= 0 || config.dim % config.head_num != 0 ||
        config.head_num % config.kv_head_num != 0) {
        throw std::invalid_argument("QKV needs valid model and MHA/GQA dimensions");
    }
    QKVWeightLayout layout;
    layout.dim = config.dim;
    layout.kv_dim = (config.dim / config.head_num) * config.kv_head_num;
    layout.layer_num = config.layer_num;
    const auto dim = static_cast<std::size_t>(layout.dim);
    const auto kv_dim = static_cast<std::size_t>(layout.kv_dim);
    const auto layers = static_cast<std::size_t>(layout.layer_num);

    // vocab_size 的符号表示分类器是否共享，embedding 行数取绝对值。
    // 先扩大到 int64_t，避免直接 abs(INT32_MIN) 的有符号溢出。
    const int64_t signed_vocab = config.vocab_size;
    const auto vocab = static_cast<std::size_t>(
        signed_vocab < 0 ? -signed_vocab : signed_vocab);
    const auto embedding_count = checked_mul(vocab, dim);
    const auto norm_count = checked_mul(layers, dim);
    const auto all_query_count = checked_mul(layers, checked_mul(dim, dim));
    const auto all_kv_count = checked_mul(layers, checked_mul(kv_dim, dim));

    // 先跳过 embedding 和所有 norm，再依次跨过所有 Q、所有 K、所有 V。
    // MHA 时 kv_dim==dim，退化成课件的 N*dim*dim；GQA 不能照抄此大小。
    layout.query_begin = checked_add(embedding_count, norm_count);
    layout.key_begin = checked_add(layout.query_begin, all_query_count);
    layout.value_begin = checked_add(layout.key_begin, all_kv_count);
    layout.end = checked_add(layout.value_begin, all_kv_count);
    // 同时保证“头部 + 元素数*sizeof(float)”可以用 size_t 表示。
    (void)checked_add(sizeof(ModelConfig), checked_mul(layout.end, sizeof(float)));
    return layout;
}

QKVWeightLoader::QKVWeightLoader(std::shared_ptr<const MappedModelFile> model)
    : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("QKV loader needs a mapped model");
    }
    layout_ = make_qkv_weight_layout(model_->config());
    const std::size_t payload_bytes = model_->size_bytes() - sizeof(ModelConfig);
    // 原 mmap 的 weight_count() 只描述第 10 课 dim*hidden_dim 的小例子。
    // 本课应检查实际 payload 长度，不能拿那个旧计数解析完整 QKV 前缀。
    if (payload_bytes % sizeof(float) != 0 ||
        layout_.end > payload_bytes / sizeof(float)) {
        throw std::runtime_error("FP32 model does not contain the complete QKV prefix");
    }
}

Tensor QKVWeightLoader::read_matrix(
    std::size_t offset, int32_t rows,
    const std::shared_ptr<DeviceAllocator>& allocator) const {
    // 构造函数已验证整个 QKV 前缀；load_layer 又检查了层号，因而这里的
    // offset 和矩阵字节数都在映射内。memcpy 避免将 mmap 字节强转 float*。
    auto host_allocator = allocator->device_type() == DeviceType::CPU
        ? allocator : std::make_shared<CPUDeviceAllocator>();
    Tensor host(DataType::FLOAT32, {rows, layout_.dim}, host_allocator);
    const auto byte_offset = sizeof(ModelConfig) + offset * sizeof(float);
    std::memcpy(host.ptr<float>(), model_->data() + byte_offset, host.byte_size());
    if (allocator->device_type() == DeviceType::CPU) {
        return host;
    }
    // 不传异步 stream，确保 CPU 临时 Tensor 析构前 H2D 复制已完成。
    // GPU 权重拥有独立 Buffer；CUDA 指针不是给 CPU 地址改个设备标签。
    return host.to_cuda(allocator);
}

QKVWeights QKVWeightLoader::load_layer(
    int32_t layer_idx, std::shared_ptr<DeviceAllocator> allocator) const {
    if (layer_idx < 0 || layer_idx >= layout_.layer_num) {
        throw std::out_of_range("QKV weight layer index out of range");
    }
    if (!allocator || (allocator->device_type() != DeviceType::CPU &&
                       allocator->device_type() != DeviceType::CUDA)) {
        throw std::invalid_argument("QKV loader needs a CPU or CUDA allocator");
    }
    const auto layer = static_cast<std::size_t>(layer_idx);
    const auto dim = static_cast<std::size_t>(layout_.dim);
    const auto kv_dim = static_cast<std::size_t>(layout_.kv_dim);
    // 每层步长分别为 dim*dim、kv_dim*dim、kv_dim*dim，全部按元素计数。
    // 大小上界已在 make_qkv_weight_layout 对所有层的乘积中检查。
    QKVWeights weights;
    weights.query = read_matrix(layout_.query_begin + layer * dim * dim,
                                layout_.dim, allocator);
    weights.key = read_matrix(layout_.key_begin + layer * kv_dim * dim,
                              layout_.kv_dim, allocator);
    weights.value = read_matrix(layout_.value_begin + layer * kv_dim * dim,
                                layout_.kv_dim, allocator);
    // 若中途读取/分配失败，已创建的局部 Tensor 通过 RAII 自动释放。
    return weights;
}

}  // namespace mini_infer
