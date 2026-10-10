#pragma once

// 第 16 课阅读顺序：
// 1. 本文件：Q/K/V 的形状，以及它们分别存放在哪里。
// 2. mini_infer_attention_qkv.cpp：slice -> 绑定三层 -> 三次 forward。
// 3. mini_infer_qkv_weight_loader.h/.cpp：从 mmap 中找到每层的 Wq/Wk/Wv。
// 依赖第 11～13 课 MatmulLayer 和第 15 课 KVCache，不改写已有算子。

#include "mini_infer/cache/mini_infer_kv_cache.h"
#include "mini_infer/ops/matmul/mini_infer_matmul.h"

namespace mini_infer {

// 一层 Transformer 的三个 FP32 权重，均为行优先 [out_features, in_features]。
// MHA：Wq/Wk/Wv 都是 [dim, dim]。
// GQA：Wq=[dim,dim]，Wk/Wv=[kv_dim,dim]，kv_dim=kv_head_num*head_dim。
// Tensor 普通复制共享 Buffer；请勿在计算过程中修改这些权重。
struct QKVWeights {
    Tensor query;
    Tensor key;
    Tensor value;
};

// ============================================================
// 第 16 课：把 RMSNorm 输出投影成当前 token 的 Q、K、V
// ============================================================
// 主线：token id -> embedding -> RMSNorm -> 本课 QKV -> RoPE -> attention。
// 本类接收已经完成 RMSNorm 的 x=[dim]，不重复做 embedding/RMSNorm。
//   q[o] = sum_i Wq[o,i] * x[i]，q=[dim]
//   k[o] = sum_i Wk[o,i] * x[i]，k=[kv_dim]
//   v[o] = sum_i Wv[o,i] * x[i]，v=[kv_dim]
// 课件把输入画成行向量，此时同一公式应写为 x * transpose(W)。
//
// Q：写入调用者可复用的独立 Tensor，不需要缓存过去的 Q。
// K/V：直接写入 cache 的 [layer_idx, token_pos, :] 两个切片。
// 第 15 课已分配好整块空间，因此这里没有 concatenate，也不调用 append
// 再复制一次。旧 token 的 K/V 保留，只计算当前 token 的三个向量。
// 缓存消耗额外持久内存，换取历史 K/V 无须重新投影；不是消除缓存内存。
//
// 课件图里的 K^T 每列对应一个 token；实际缓存按 [token,kv_dim] 存储，
// 每个 token 是一行。数学上转置不要求这里真的转置/搬移缓存。
// 随后的每个头计算 softmax(q*K_history^T/sqrt(head_dim))*V_history；
// 本课只准备输入，不在此实现多头 attention、softmax 或 RoPE 算法。
class AttentionQKV {
public:
    // 权重已经在 CPU 或 CUDA 上；三个权重必须同设备、FP32、形状匹配。
    // 三个 MatmulLayer 通过 set_weight 共享并保活自有 Buffer，不借用 mmap。
    // 若手动传入 Tensor::borrow，外部内存仍由调用者负责保活。
    explicit AttentionQKV(const QKVWeights& weights);

    // 只计算一个 token，输入和 query 严格为 [dim]。
    // token_pos 从 0 开始，必须等于本层 valid_length，禁止覆盖历史或跳写。
    // 课件 pos_tensor 是 CPU INT32 [1]，可先读取 pos_tensor.index<int32_t>(0)
    // 再传进来；不要在主机上 index CUDA Tensor。
    // 返回当前 K/V 的可写 view，便于下一步对 Q/K 应用 RoPE。
    //
    // 重要：此函数不 commit！K 此时尚未经过位置旋转。
    // 正确的模型接入顺序（rope 是后续课程的算子，不是本课新增 API）：
    //   auto [key, value] = qkv.forward(x, query, cache, layer_idx, pos, stream);
    //   // 在同一 stream 上调用 RoPE(query, key, pos, sin_cache, cos_cache)。
    //   cache.commit(layer_idx, pos);
    //   auto [keys, values] = cache.history(layer_idx);
    //   // 再计算当前 query 与 keys/values 的 attention。
    // 只验证原始投影的小例子可以直接 commit，但不等价于完整 Llama。
    //
    // CUDA forward 异步提交；包括默认 stream 在内，都不能把返回当作
    // GPU 已完成。输入、权重、query、缓存须存活至完成；CPU 读数据或
    // 跨 stream 消费前先同步/建立 event 依赖。一个实例不可并发 forward。
    // 参数错误在计算前抛出；设备运行错误不会回滚已写的 Q/未提交槽位。
    std::pair<Tensor, Tensor> forward(
        const Tensor& rmsnorm_output, Tensor& query, KVCache& cache,
        int32_t layer_idx, int32_t token_pos, cudaStream_t stream = nullptr);

private:
    int32_t dim_;
    int32_t kv_dim_;
    MatmulLayer query_layer_;
    MatmulLayer key_layer_;
    MatmulLayer value_layer_;
};

}  // namespace mini_infer
