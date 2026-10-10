#pragma once

// 第 16 课的第二个问题：三个投影矩阵如何从模型文件取出。
// 先读 attention_qkv.h/.cpp 看懂“怎么用”，再读这里看“从哪里来”。

#include "mini_infer/ops/attention/mini_infer_attention_qkv.h"
#include "mini_infer/weights/mini_infer_mmap.h"

namespace mini_infer {

// 偏移均以 FP32 元素为单位，相对于 28 字节 ModelConfig 之后的 payload。
// 课件 export.py 的顺序是按权重种类分组，而不是逐层交错保存 Q/K/V：
//   embedding             [abs(vocab_size), dim]
//   全部 attention_norm   [layer_num, dim]
//   全部 Wq               [layer_num, dim, dim]
//   全部 Wk               [layer_num, kv_dim, dim]
//   全部 Wv               [layer_num, kv_dim, dim]
//   后续 Wo/FFN/...       本课不读取
//
// 例：dim=4, vocab_size=-3, layer_num=2, kv_dim=2：
//   query_begin = 3*4 + 2*4 = 20
//   key_begin   = 20 + 2*4*4 = 52
//   value_begin = 52 + 2*2*4 = 68
//   end         = 68 + 2*2*4 = 84
// 第 1 层 Wk 的元素偏移是 52 + 1*2*4 = 60；文件字节偏移是 28+60*4。
struct QKVWeightLayout {
    int32_t dim = 0;
    int32_t kv_dim = 0;
    int32_t layer_num = 0;
    std::size_t query_begin = 0;
    std::size_t key_begin = 0;
    std::size_t value_begin = 0;
    std::size_t end = 0;  // Wv 最后一个元素之后，不等于完整模型的结尾。
};

// 检查头参数，并用受溢出保护的 size_t 算术计算布局；不分配模型内存。
QKVWeightLayout make_qkv_weight_layout(const ModelConfig& config);

class QKVWeightLoader {
public:
    // 本读取器对应上述 FP32 教学导出布局，不是 GGUF/量化模型解析器。
    // 复用旧 MappedModelFile 的只读映射；不改变前面课程的文件头和 API。
    explicit QKVWeightLoader(std::shared_ptr<const MappedModelFile> model);

    const QKVWeightLayout& layout() const noexcept { return layout_; }

    // 定位某层三个矩阵，复制到有所有权的 Tensor，再按需要迁移到 CUDA。
    // allocator 不能为空；传 CPU/CUDA allocator 决定输出设备。
    // CUDA 权重迁移在本函数返回前完成，之后才用于异步 forward。
    // 返回权重与映射解耦：读取器/mmap 析构后，算子权重仍有效。
    // 只分配当前层 Q/K/V，不分配或复制被跳过的 embedding 和 norm。
    QKVWeights load_layer(int32_t layer_idx,
                          std::shared_ptr<DeviceAllocator> allocator) const;

private:
    Tensor read_matrix(std::size_t offset, int32_t rows,
                       const std::shared_ptr<DeviceAllocator>& allocator) const;

    std::shared_ptr<const MappedModelFile> model_;
    QKVWeightLayout layout_;
};

}  // namespace mini_infer
