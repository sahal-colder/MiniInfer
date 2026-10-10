#include "mini_infer/weights/mini_infer_weight_loader.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mini_infer {

FP32WeightLoader::FP32WeightLoader(std::shared_ptr<const MappedModelFile> model)
    : model_(std::move(model)) {
    // 第一步：取得一个有所有权的映射，而不是保存裸指针。
    // shared_ptr 让读取器在整个载入阶段都能保证 data() 仍然有效。
    if (!model_) {
        throw std::invalid_argument("FP32 weight loader needs a mapped model");
    }

    // MappedModelFile 已校验 28 字节头部和第一段 dim*hidden_dim。
    // 本课进一步把头部之后的全部数据当作连续 FP32 权重。
    // 除以 sizeof(float) 之前先检查余数，避免丢弃末尾半个元素。
    const std::size_t payload_bytes = model_->size_bytes() - sizeof(ModelConfig);
    if (payload_bytes % sizeof(float) != 0) {
        throw std::runtime_error("FP32 model ends inside a weight element");
    }
    total_count_ = payload_bytes / sizeof(float);
}

Tensor FP32WeightLoader::read_tensor(std::vector<int32_t> dims) {
    // 第二步：先只建立 shape/dtype 元数据，不立即分配内存。
    // Tensor 构造会检查正维度、元素数乘法溢出、元素数到字节数溢出。
    // 例：shape=[16,128]，size()=2048，byte_size()=8192。
    Tensor weight(DataType::FLOAT32, std::move(dims));
    if (weight.size() > remaining()) {
        throw std::out_of_range("mapped file has no complete weight tensor left");
    }

    // 第三步：由 CPU allocator 分配属于 Tensor 自己的 Buffer。
    // 只复制当前这一段权重，不把整个模型文件先读进另一个大数组。
    weight.allocate(std::make_shared<CPUDeviceAllocator>());
    const std::size_t byte_offset = sizeof(ModelConfig) + position_ * sizeof(float);
    std::memcpy(weight.ptr<float>(), model_->data() + byte_offset,
                weight.byte_size());

    // 第四步：全部成功后，游标按“元素数”推进。
    // 上面的 remaining 校验也保证 position_ + size() 不会越界/溢出。
    // 不能写 position_ += byte_size()，否则下一段会跳过四倍的距离。
    position_ += weight.size();
    return weight;
}

void FP32WeightLoader::load_weight(ParamLayer& layer, std::size_t slot,
                                 std::vector<int32_t> dims) {
    // 先检查目的地：本读取器返回 CPU Tensor，不能把 CPU 指针误标为
    // CUDA 指针。迁移需要真正执行 Tensor::to_cuda 所调用的 cudaMemcpy。
    if (layer.device_type() != DeviceType::CPU) {
        throw std::invalid_argument("load_weight binds CPU layers only");
    }
    (void)layer.get_weight(slot); // at() 检查槽位；失败时尚未读取/分配。

    const std::size_t old_position = position_;
    try {
        Tensor weight = read_tensor(std::move(dims));
        // 原有 set_weight 保存 Tensor 元数据并共享 shared_ptr<Buffer>。
        // 局部 weight 析构不会释放仍由 layer 引用的权重内存。
        layer.set_weight(slot, weight);
    } catch (...) {
        // 即使读取成功而绑定失败，下次也应从同一段权重开始。
        // 局部 Tensor 的 RAII 会释放尚未交给算子的自有内存。
        position_ = old_position;
        throw;
    }
}

} // namespace mini_infer
