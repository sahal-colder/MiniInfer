#pragma once

#include "mini_infer/core/mini_infer_tensor.h"
#include <utility>

namespace mini_infer {

struct ModelConfig;

// ============================================================
// 连续、FP32、单序列的 KV Cache
// ============================================================
// KVCache 只负责两块连续 K/V 存储、每层有效长度和 Tensor view。
// attention、RoPE、采样和请求调度由调用者负责。
//
// 所有权：构造函数用 allocator 创建 key_cache_ 和 value_cache_；缓存
// Tensor 持有底层 Buffer。slice/history 返回的 view 共享该 Buffer，
// 因此缓存对象析构后，仍存活的 view 继续保持数据有效。
// 每层只缓存 K 和 V，Q 是当前 token 的查询，不保存在这里。
// 两块独立 Tensor 的形状均为 [layer_num, max_seq_len, kv_dim]。
// 对于 GQA：head_dim = dim / head_num，kv_dim = kv_head_num * head_dim；
// 所以 kv_dim 不一定等于模型的 dim。K/V 合计空间：
//   2 * layer_num * max_seq_len * kv_dim * sizeof(float)。
//
// 地址偏移（元素单位）：
//   offset = (layer_idx * max_seq_len + token_pos) * kv_dim。
// 分配发生在构造时，decode 不会为了“拼接”历史再复制整块缓存。
// slice/history 返回 Tensor::view，共享并保活自有的底层存储。
//
// 此类按层记录有效长度，支持多 token prefill 和单 token decode。
// 不支持并发修改同一个对象。CUDA 写入/清零是异步的：同一序列的
// 操作应在同一 stream 排队；跨 stream 使用前建立 event/同步依赖。
// 输入及存储必须存活到 GPU 完成；reset/析构也必须遵守此规则。
class KVCache {
public:
    KVCache(int32_t layer_num, int32_t max_seq_len, int32_t kv_dim,
            std::shared_ptr<DeviceAllocator> allocator);

    // 根据模型配置，检查 head 参数并推导 GQA 的 kv_dim。
    KVCache(const ModelConfig& config, std::shared_ptr<DeviceAllocator> allocator);

    // 防止“共享 K/V 存储，却复制出另一套有效长度”的不一致状态。
    // 缓存本身固定在原对象中；返回的 Tensor 视图可以正常复制/移动。
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    int32_t layer_num() const noexcept { return layer_num_; }
    int32_t max_seq_len() const noexcept { return max_seq_len_; }
    int32_t kv_dim() const noexcept { return kv_dim_; }
    DeviceType device_type() const noexcept { return key_cache_.device_type(); }
    std::size_t byte_size() const noexcept { return 2 * key_cache_.byte_size(); }
    int32_t valid_length(int32_t layer_idx) const;

    // 返回两条 [kv_dim]
    // 可写视图，可直接作为 Wk/Wv 投影算子的输出，避免临时 K/V 的拷贝。
    // slice 不增加有效长度；未写入的切片数据没有初始化，不能读取。
    // 若有 RoPE，应在完成 K 的旋转后再提交为有效缓存。
    std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer_idx, int32_t token_pos) const;

    // 直接写 slice 后调用 commit；token_pos 必须等于当前有效长度，
    // token_count 默认为 1。它只更新主机元数据，不验证 GPU 数据，
    // 也不等待 GPU。消费历史的算子仍需与生产 K/V 的 stream 有依赖。
    void commit(int32_t layer_idx, int32_t token_pos, int32_t token_count = 1);

    // 已有 K/V 时的便利入口：接受 [kv_dim] 或 [token_count, kv_dim]，
    // 追加到本层有效长度之后。两个输入须同形状、同设备且为 FP32。
    // 输入必须在缓存之外；拒绝重叠，避免并行拷贝覆盖未读取的数据。
    // 所有参数先验证，提交 CPU/CUDA 写入成功后才推进有效长度。
    void append(int32_t layer_idx, const Tensor& key, const Tensor& value,
                cudaStream_t stream = nullptr);

    // 返回本层 [valid_length, kv_dim] 的连续历史视图，包括当前 token。
    // 长度为 0 时返回两个默认空 Tensor（已有 Tensor 不支持 0 维）。
    // 按只读历史使用；视图本身可写，且不会自动追踪后续长度或 reset。
    std::pair<Tensor, Tensor> history(int32_t layer_idx) const;

    // 默认只把各层长度置 0，O(layer_num)，下次写入复用现有内存。
    // zero_storage=true 才清零全部 K/V；已有视图共享这些数据。
    // 调用前完成旧请求的使用；CUDA 清零完成前不能从其他 stream 读取。
    void reset(bool zero_storage = false, cudaStream_t stream = nullptr);

private:
    void check_layer(int32_t layer_idx) const;
    void check_append_range(int32_t layer_idx, int32_t token_pos, int32_t count) const;
    std::size_t offset(int32_t layer_idx, int32_t token_pos) const;

    int32_t layer_num_;
    int32_t max_seq_len_;
    int32_t kv_dim_;
    Tensor key_cache_;
    Tensor value_cache_;
    std::vector<int32_t> valid_lengths_;
};

// 两个后端有一致的签名。低层函数只复制当前 K/V，不更新有效长度。
// source/destination 都要求 [D] 或 [T,D] 且四者形状一致。
using KVCacheWriteKernel = void (*)(const Tensor&, const Tensor&, Tensor&, Tensor&,
                                    cudaStream_t);
KVCacheWriteKernel get_kv_cache_write_kernel(DeviceType device);
void kv_cache_write_cpu(const Tensor& key, const Tensor& value,
                        Tensor& key_slot, Tensor& value_slot, cudaStream_t stream);
void kv_cache_write_cuda(const Tensor& key, const Tensor& value,
                         Tensor& key_slot, Tensor& value_slot, cudaStream_t stream);

namespace detail {
// 共享元数据检查，不解引用 GPU 指针；直接调用后端也受保护。
void check_kv_cache_write(const Tensor& key, const Tensor& value,
                          const Tensor& key_slot, const Tensor& value_slot,
                          DeviceType device);
void zero_kv_cache_cuda(Tensor& key_cache, Tensor& value_cache, cudaStream_t stream);
}  // namespace detail

}  // namespace mini_infer
