#pragma once

// 第 17 课补充：复用第 16 课的 QKV 前缀定位 FFN 权重。
// 这部分是“从哪里读”；MLP 本身只接收 Tensor，不依赖 mmap。
#include "mini_infer/ops/mlp/mini_infer_mlp.h"
#include "mini_infer/weights/mini_infer_qkv_weight_loader.h"

namespace mini_infer {

// 课件配套 FP32 导出布局，偏移以 ModelConfig 之后的 float 元素计数：
// embedding -> 全部 attention_norm -> 全部 Wq -> 全部 Wk -> 全部 Wv
// -> 全部 Wo [layers,dim,dim] -> 全部 FFN_norm [layers,dim]
// -> 全部 W1 [layers,hidden,dim] -> 全部 W2 [layers,dim,hidden]
// -> 全部 W3 [layers,hidden,dim] -> 后续 final_norm/RoPE/classifier。
// 不能误读成“第 0 层 W1,W2,W3，再第 1 层 W1,W2,W3”。
struct MLPWeightLayout {
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t layer_num = 0;
    std::size_t norm_begin = 0;
    std::size_t w1_begin = 0;
    std::size_t w2_begin = 0;
    std::size_t w3_begin = 0;
    std::size_t end = 0;  // W3 段结尾，不是完整模型结尾。
};

MLPWeightLayout make_mlp_weight_layout(const ModelConfig& config);

class MLPWeightLoader {
public:
    explicit MLPWeightLoader(std::shared_ptr<const MappedModelFile> model);
    const MLPWeightLayout& layout() const noexcept { return layout_; }

    // 只读取指定层四份权重，复制进独立 Buffer；不借用只读 mmap。
    // allocator 决定 CPU/CUDA。上传同步完成，局部 CPU Tensor 才可析构。
    // 读取器/映射销毁后返回值仍有效，不支持 GGUF 或量化导出格式。
    MLPWeights load_layer(int32_t layer_idx,
                          std::shared_ptr<DeviceAllocator> allocator) const;

private:
    Tensor read_tensor(std::size_t offset, std::vector<int32_t> shape,
                        const std::shared_ptr<DeviceAllocator>& allocator) const;
    std::shared_ptr<const MappedModelFile> model_;
    MLPWeightLayout layout_;
};
}  // namespace mini_infer
