#pragma once

// 第 15 课阅读顺序：
// 1. mini_infer_kv_cache.cpp：两块 Tensor、元素偏移、slice，先理解课件主线。
// 2. mini_infer_kv_cache_append.cpp：append/commit，只让连续写入成为有效历史。
// 3. mini_infer_kv_cache_history.cpp：history/reset，区分有效长度和物理容量。
// 4. mini_infer_kv_cache_kernel.h/.cpp/.cu：只复制当前一段 K/V。
// 5. mini_infer_kv_cache_config.cpp：可选的 ModelConfig/GQA 构造入口。
// 已有 Tensor::view 直接调用，不需要重新学习或重写 Tensor。
// 最小调用链：slice -> 两次 Sgemv 写 K/V -> commit -> history -> reset。

#include "mini_infer/tensor/mini_infer_tensor.h"
#include "mini_infer/ops/mini_infer_kv_cache_kernel.h"  // 保留已有低层写入接口的可见性。

#include <utility>

namespace mini_infer {

struct ModelConfig;

// ============================================================
// 第 15 课：连续、FP32、单序列的 KV Cache
// ============================================================
// KVCache 只负责两块连续 K/V 存储、每层有效长度和 Tensor view。
// 它不负责 attention、RoPE、采样或请求调度。这样可以先把“缓存数据
// 放在哪里、谁拥有它、何时算作有效”讲清楚，再在后续课程连接注意力。
//
// 所有权：构造函数用 allocator 创建初始 key_cache_ 和 value_cache_；缓存
// Tensor 持有底层 Buffer。slice/history 返回的 view 共享该 Buffer，
// 因此缓存对象析构后，仍存活的 view 继续保持数据有效。
// 每层只缓存 K 和 V，Q 是当前 token 的查询，不保存在这里。
// 两块独立 Tensor 的形状均为 [layer_num, allocated_tokens, kv_dim]。
// 对于 GQA：head_dim = dim / head_num，kv_dim = kv_head_num * head_dim；
// 所以 kv_dim 不一定等于模型的 dim。K/V 合计空间：
//   2 * layer_num * max_seq_len * kv_dim * sizeof(float)。
//
// 地址偏移（元素单位）：
//   offset = (layer_idx * allocated_tokens + token_pos) * kv_dim。
// 初始最多分配 64-token 容量；reserve 按倍数增长，仅复制已提交前缀。
// prefill 可一次 reserve 全段，decode 只在跨越容量边界时重新分配。
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

    // 复用第 10 课配置，检查 head 参数并推导 GQA 的 kv_dim。
    KVCache(const ModelConfig& config, std::shared_ptr<DeviceAllocator> allocator);

    // 防止“共享 K/V 存储，却复制出另一套有效长度”的不一致状态。
    // 缓存本身固定在原对象中；返回的 Tensor 视图可以正常复制/移动。
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    int32_t layer_num() const noexcept { return layer_num_; }
    int32_t max_seq_len() const noexcept { return max_seq_len_; }
    int32_t kv_dim() const noexcept { return kv_dim_; }
    DeviceType device_type() const noexcept { return key_cache_.device_type(); }
    // Logical maximum, retained for checkpoint layout/budget compatibility.
    std::size_t byte_size() const noexcept { return capacity_bytes_; }
    std::size_t allocated_bytes() const noexcept { return 2 * key_cache_.byte_size(); }
    // Growth changes physical capacity, not logical history. Existing views
    // retain their old backing allocation. Producers must belong to stream.
    void reserve(int32_t tokens, cudaStream_t stream = nullptr) const;
    int32_t valid_length(int32_t layer_idx) const;

    // 对应课件的 slice_kv_cache(layer_idx, token_pos)。返回两条 [kv_dim]
    // 可写视图，可直接作为 Wk/Wv 投影算子的输出，避免临时 K/V 的拷贝。
    // slice 不增加有效长度；未写入的切片数据没有初始化，不能读取。
    // 若有 RoPE，应在完成 K 的旋转后再提交为有效缓存。
    std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer_idx, int32_t token_pos,
                                             cudaStream_t stream = nullptr) const;

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

    // 默认只把各层长度置 0，O(layer_num)，保留高水位容量供后续复用。
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
    std::size_t capacity_bytes_ = 0;
    mutable int32_t allocated_tokens_ = 0;
    std::shared_ptr<DeviceAllocator> allocator_;
    mutable Tensor key_cache_;
    mutable Tensor value_cache_;
    std::vector<int32_t> valid_lengths_;
};

}  // namespace mini_infer
