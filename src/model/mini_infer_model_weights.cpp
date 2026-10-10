#include "mini_infer/model/mini_infer_model_weights.h"

#include "mini_infer/weights/mini_infer_mlp_weight_loader.h"
#include "mini_infer/weights/mini_infer_qkv_weight_loader.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mini_infer {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("model layout addition overflow");
    return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("model layout multiplication overflow");
    return a * b;
}
}

ModelWeights load_model_weights(const std::string& path,
                                std::shared_ptr<DeviceAllocator> allocator) {
    if (!allocator || (allocator->device_type() != DeviceType::CPU &&
                       allocator->device_type() != DeviceType::CUDA))
        throw std::invalid_argument("model loader needs a CPU or CUDA allocator");
    const uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1)
        throw std::runtime_error("legacy checkpoint requires a little-endian host");

    auto file = std::make_shared<MappedModelFile>(path);
    const auto& c = file->config();
    const auto qkv_layout = make_qkv_weight_layout(c);
    const auto mlp_layout = make_mlp_weight_layout(c);
    if (c.vocab_size == std::numeric_limits<int32_t>::min() ||
        (c.dim / c.head_num) % 2 != 0)
        throw std::invalid_argument("model needs representable vocab and even RoPE head size");
    const int32_t vocab = c.vocab_size < 0 ? -c.vocab_size : c.vocab_size;
    const auto dim = static_cast<std::size_t>(c.dim);
    const auto embedding_count = multiply(vocab, dim);
    const auto rope_begin = add(mlp_layout.end, dim);
    const auto rope_count = multiply(c.seq_len, c.dim / c.head_num);
    const auto classifier_begin = add(rope_begin, rope_count);
    const auto count = c.vocab_size > 0 ? classifier_begin
                                      : add(classifier_begin, embedding_count);
    const auto expected_bytes = add(sizeof(ModelConfig), multiply(count, sizeof(float)));
    if (file->size_bytes() != expected_bytes)
        throw std::runtime_error("expected complete legacy FP32 model (including RoPE tables): "
                                 + std::to_string(expected_bytes) + " bytes, got "
                                 + std::to_string(file->size_bytes()));

    // 只让当前一段权重暂存在 CPU；CUDA 模型不会额外保留完整 CPU 权重副本。
    const auto cpu = allocator->device_type() == DeviceType::CPU
        ? allocator : std::make_shared<CPUDeviceAllocator>();
    auto read = [&](std::size_t offset, std::vector<int32_t> shape) {
        Tensor tensor(DataType::FLOAT32, std::move(shape));
        if (offset > count || tensor.size() > count - offset)
            throw std::out_of_range("model tensor exceeds mapped payload");
        tensor.allocate(cpu);
        std::memcpy(tensor.ptr<float>(), file->data() + sizeof(ModelConfig)
                    + offset * sizeof(float), tensor.byte_size());
        return allocator->device_type() == DeviceType::CPU ? tensor : tensor.to_cuda(allocator);
    };
    QKVWeightLoader qkv_loader(file);
    MLPWeightLoader mlp_loader(file);
    ModelWeights weights;
    weights.config = c;
    weights.config.vocab_size = vocab;
    weights.embedding = read(0, {vocab, c.dim});
    weights.layers.reserve(c.layer_num);
    for (int32_t layer = 0; layer < c.layer_num; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        TransformerWeights block;
        block.attention_norm = read(embedding_count + index * dim, {c.dim});
        block.qkv = qkv_loader.load_layer(layer, allocator);
        block.output = read(qkv_layout.end + index * dim * dim, {c.dim, c.dim});
        block.mlp = mlp_loader.load_layer(layer, allocator);
        weights.layers.push_back(std::move(block));
    }
    weights.final_norm = read(mlp_layout.end, {c.dim});
    weights.classifier = c.vocab_size > 0 ? weights.embedding
        : read(classifier_begin, {vocab, c.dim});
    return weights;
}
}  // namespace mini_infer
