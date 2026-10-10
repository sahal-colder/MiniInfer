#include "mini_infer/weights/mini_infer_mlp_weight_loader.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_infer {
namespace {
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("MLP weight layout multiplication overflow");
    }
    return a * b;
}
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::overflow_error("MLP weight layout addition overflow");
    }
    return a + b;
}
}  // namespace

MLPWeightLayout make_mlp_weight_layout(const ModelConfig& config) {
    // 已有 helper 检查头参数，并按 GQA 的 kv_dim 跨过 Q/K/V。
    const QKVWeightLayout qkv = make_qkv_weight_layout(config);
    MLPWeightLayout layout;
    layout.dim = config.dim;
    layout.hidden_dim = config.hidden_dim;
    layout.layer_num = config.layer_num;
    const auto dim = static_cast<std::size_t>(config.dim);
    const auto hidden = static_cast<std::size_t>(config.hidden_dim);
    const auto layers = static_cast<std::size_t>(config.layer_num);
    const auto all_wo = checked_mul(layers, checked_mul(dim, dim));
    const auto all_norm = checked_mul(layers, dim);
    const auto all_projection = checked_mul(layers, checked_mul(dim, hidden));
    layout.norm_begin = checked_add(qkv.end, all_wo);
    layout.w1_begin = checked_add(layout.norm_begin, all_norm);
    layout.w2_begin = checked_add(layout.w1_begin, all_projection);
    layout.w3_begin = checked_add(layout.w2_begin, all_projection);
    layout.end = checked_add(layout.w3_begin, all_projection);
    // 不仅元素偏移，乘 sizeof(float) 后的字节偏移也必须不溢出。
    (void)checked_add(sizeof(ModelConfig), checked_mul(layout.end, sizeof(float)));
    return layout;
}

MLPWeightLoader::MLPWeightLoader(std::shared_ptr<const MappedModelFile> model)
    : model_(std::move(model)) {
    if (!model_) throw std::invalid_argument("MLP loader needs a mapped model");
    layout_ = make_mlp_weight_layout(model_->config());
    const auto payload_bytes = model_->size_bytes() - sizeof(ModelConfig);
    if (payload_bytes % sizeof(float) != 0 ||
        layout_.end > payload_bytes / sizeof(float)) {
        throw std::runtime_error("FP32 model does not contain the complete MLP prefix");
    }
}

Tensor MLPWeightLoader::read_tensor(
    std::size_t offset, std::vector<int32_t> shape,
    const std::shared_ptr<DeviceAllocator>& allocator) const {
    auto host_allocator = allocator->device_type() == DeviceType::CPU
        ? allocator : std::make_shared<CPUDeviceAllocator>();
    Tensor host(DataType::FLOAT32, std::move(shape), host_allocator);
    // load_layer 的层号、布局、完整文件大小均已检查。
    const auto byte_offset = sizeof(ModelConfig) + offset * sizeof(float);
    std::memcpy(host.ptr<float>(), model_->data() + byte_offset, host.byte_size());
    return allocator->device_type() == DeviceType::CPU ? host : host.to_cuda(allocator);
}

MLPWeights MLPWeightLoader::load_layer(
    int32_t layer_idx, std::shared_ptr<DeviceAllocator> allocator) const {
    if (layer_idx < 0 || layer_idx >= layout_.layer_num) {
        throw std::out_of_range("MLP weight layer index out of range");
    }
    if (!allocator || (allocator->device_type() != DeviceType::CPU &&
                       allocator->device_type() != DeviceType::CUDA)) {
        throw std::invalid_argument("MLP loader needs a CPU or CUDA allocator");
    }
    const auto layer = static_cast<std::size_t>(layer_idx);
    const auto dim = static_cast<std::size_t>(layout_.dim);
    const auto hidden = static_cast<std::size_t>(layout_.hidden_dim);
    // 三个投影的元素数相同，但 W2 的行列次序与 W1/W3 相反。
    const auto projection_offset = layer * dim * hidden;
    MLPWeights weights;
    weights.norm = read_tensor(layout_.norm_begin + layer * dim, {layout_.dim}, allocator);
    weights.w1 = read_tensor(layout_.w1_begin + projection_offset,
                             {layout_.hidden_dim, layout_.dim}, allocator);
    weights.w2 = read_tensor(layout_.w2_begin + projection_offset,
                             {layout_.dim, layout_.hidden_dim}, allocator);
    weights.w3 = read_tensor(layout_.w3_begin + projection_offset,
                             {layout_.hidden_dim, layout_.dim}, allocator);
    return weights;
}
}  // namespace mini_infer
