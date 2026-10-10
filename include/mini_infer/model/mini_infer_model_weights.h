#pragma once

#include "mini_infer/ops/attention/mini_infer_attention_qkv.h"
#include "mini_infer/ops/mlp/mini_infer_mlp.h"
#include "mini_infer/weights/mini_infer_mmap.h"

namespace mini_infer {

struct TransformerWeights {
    Tensor attention_norm;  // [dim]
    QKVWeights qkv;         // Q=[dim,dim]，K/V=[kv_dim,dim]
    Tensor output;          // Wo=[dim,dim]
    MLPWeights mlp;         // FFN norm、W1、W2、W3
};

struct ModelWeights {
    ModelConfig config;     // 对外 vocab_size 已变为正数。
    Tensor embedding;       // [vocab_size,dim]
    std::vector<TransformerWeights> layers;
    Tensor final_norm;      // [dim]
    Tensor classifier;      // [vocab_size,dim]，共享时与 embedding 共用 Buffer。
};

// 完整读取 llama2.c legacy FP32 / export version 0 格式：
// 28 字节头 -> embedding -> 各层 att norm -> Q -> K -> V -> Wo ->
// FFN norm -> W1 -> W2 -> W3 -> final norm -> 旧 RoPE 两表 -> 可选 classifier。
// 正 vocab 表示共享 classifier，负 vocab 表示文件末尾另有 classifier。
// 两张旧 RoPE 表必须存在，但运行时按 theta=10000 重新计算旋转角度。
// 本格式不含可变 epsilon/theta，固定为 Llama 2 的 1e-5 / 10000。
// 不直接接受 GGUF、safetensors、量化文件或新版带 magic 的 checkpoint。
// 全文件大小通过后才分配权重；复制到自有 Tensor，函数返回即可解除 mmap。
ModelWeights load_model_weights(const std::string& path,
                                std::shared_ptr<DeviceAllocator> allocator);
}  // namespace mini_infer
