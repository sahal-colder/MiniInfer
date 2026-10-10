#pragma once

// 第 18 课主线：本文件形状 -> .cpp 三步标量实现 -> .cu 并行实现。
#include "mini_infer/cache/mini_infer_kv_cache.h"

namespace mini_infer {

// 单序列、单 query token 的 scaled dot-product 多头注意力。
//   head_dim = dim/head_num; kv_mul = head_num/kv_head_num
//   kv_head = head/kv_mul
//   score[h,t] = dot(query[h,:], keys[t,kv_head,:])/sqrt(head_dim)
//   probability[h,:] = softmax(score[h,:])
//   output[h,i] = sum_t probability[h,t]*values[t,kv_head,i]
// MHA：head_num == kv_head_num；GQA/MQA：多个 Q 头共享一个 K/V 头。
//
// query/output=[dim]，keys/values=[length,kv_dim]，kv_dim=kv_head_num*head_dim。
// scores=[head_num,capacity]，capacity>=length，由调用者提前分配并复用。
// 结束时 scores 的前 length 列保存概率，其余列不改动，也不参与归一化。
// keys/values 必须只包含本 query 可见的历史（含当前 token），不能包含未来。
// 所有 Tensor 必须为同设备 FP32；两个输出不得重叠任何输入或彼此。
// CUDA 的三个阶段提交到同一 stream；返回时 GPU 不一定完成。
// 该组件不做 QKV 投影、RoPE、Wo 输出投影或残差，可分别连接已有算子。
using AttentionKernel = void (*)(const Tensor&, const Tensor&, const Tensor&,
                                 Tensor&, Tensor&, int32_t, int32_t, cudaStream_t);
AttentionKernel get_attention_kernel(DeviceType device);
void attention_kernel_cpu(const Tensor& query, const Tensor& keys, const Tensor& values,
                          Tensor& scores, Tensor& output, int32_t head_num,
                          int32_t kv_head_num, cudaStream_t stream = nullptr);
void attention_kernel_cuda(const Tensor& query, const Tensor& keys, const Tensor& values,
                           Tensor& scores, Tensor& output, int32_t head_num,
                           int32_t kv_head_num, cudaStream_t stream = nullptr);

class MultiHeadAttention {
public:
    MultiHeadAttention(int32_t head_num, int32_t kv_head_num, int32_t head_dim);

    // token_pos 从 0 开始，读取本层已 commit 的 [0,token_pos]，共 pos+1 个。
    // 课件 layer_offset 由 cache.history(layer_idx) 的 view 自动完成。
    // 即使缓存里还有较晚的 token，这里也截断视图，保证因果可见性。
    // Q/K 如需 RoPE，由调用者先处理 Q/当前 K，再 commit，再调用本函数。
    // forward 不分配/追加/提交缓存，也不修改 query/历史 K/V。
    // 同一 cache/工作区的 CUDA 生产与消费须在同一 stream 或有 event 依赖。
    void forward(const Tensor& query, const KVCache& cache, int32_t layer_idx,
                 int32_t token_pos, Tensor& scores, Tensor& output,
                 cudaStream_t stream = nullptr) const;
private:
    int32_t head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
};

namespace detail {
void check_attention_tensors(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, int32_t head_num, int32_t kv_head_num,
                             DeviceType device);
}  // namespace detail
}  // namespace mini_infer
