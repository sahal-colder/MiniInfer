#include "mini_infer/ops/attention/mini_infer_attention_qkv.h"

#include <stdexcept>

namespace mini_infer {
namespace {

// 构造之前一次检查三个矩阵。dim() 之前先检查 rank，避免空 Tensor 访问。
int32_t checked_dim(const QKVWeights& weights) {
    const auto device = weights.query.device_type();
    if (device != DeviceType::CPU && device != DeviceType::CUDA) {
        throw std::invalid_argument("QKV weights need a CPU or CUDA device");
    }
    for (const Tensor* weight : {&weights.query, &weights.key, &weights.value}) {
        if (weight->empty() || weight->dims().size() != 2 ||
            weight->data_type() != DataType::FLOAT32 ||
            weight->device_type() != device) {
            throw std::invalid_argument("QKV weights must be FP32 matrices on one device");
        }
    }
    const int32_t dim = weights.query.dim(1);
    const int32_t kv_dim = weights.key.dim(0);
    if (weights.query.dim(0) != dim || weights.key.dim(1) != dim ||
        weights.value.dims() != weights.key.dims() || dim % kv_dim != 0) {
        throw std::invalid_argument("QKV expects Wq=[dim,dim], Wk/Wv=[kv_dim,dim]");
    }
    return dim;
}

}  // namespace

AttentionQKV::AttentionQKV(const QKVWeights& weights)
    : dim_(checked_dim(weights)), kv_dim_(weights.key.dim(0)),
      query_layer_(weights.query.device_type(), dim_, dim_),
      key_layer_(weights.query.device_type(), kv_dim_, dim_),
      value_layer_(weights.query.device_type(), kv_dim_, dim_) {
    // 课件中每层分别放进 wq_layers_/wk_layers_/wv_layers_ 三个数组；
    // 这里把同一层的三个算子放在一起。多层模型可保存多个 AttentionQKV。
    query_layer_.set_weight(0, weights.query);
    key_layer_.set_weight(0, weights.key);
    value_layer_.set_weight(0, weights.value);
}

std::pair<Tensor, Tensor> AttentionQKV::forward(
    const Tensor& rmsnorm_output, Tensor& query, KVCache& cache,
    int32_t layer_idx, int32_t token_pos, cudaStream_t stream) {
    // 第一步：先检查连续位置和设备/宽度，再取得当前的两个输出切片。
    // valid_length 会检查 layer_idx；slice 会检查 token_pos 是否超容量。
    if (token_pos != cache.valid_length(layer_idx)) {
        throw std::invalid_argument("QKV must write the next token, without gaps or overwrites");
    }
    if (cache.kv_dim() != kv_dim_ || cache.device_type() != query_layer_.device_type() ||
        rmsnorm_output.dims() != std::vector<int32_t>{dim_} ||
        query.dims() != std::vector<int32_t>{dim_}) {
        throw std::invalid_argument("QKV expects x/query=[dim] and a matching KVCache");
    }
    auto [key, value] = cache.slice_kv_cache(layer_idx, token_pos);

    // 第二步：绑定三个 Matmul 的输入和输出。
    // set_output 只是复制 Tensor 的元数据和 shared_ptr，不复制数据。
    // 因此 key_layer_ 的输出 Buffer 就是缓存当前 token 的那一小段。
    query_layer_.set_input(0, rmsnorm_output);
    query_layer_.set_output(0, query);
    key_layer_.set_input(0, rmsnorm_output);
    key_layer_.set_output(0, key);
    value_layer_.set_input(0, rmsnorm_output);
    value_layer_.set_output(0, value);

    // 三次检查必须全部完成后再执行第一次数值计算，避免 V 参数错误时
    // Q/K 已写入。单个 Matmul 只知道自己的权重；下面还检查跨层别名。
    query_layer_.check();
    key_layer_.check();
    value_layer_.check();
    for (const Tensor* output : {&query, &key, &value}) {
        for (const MatmulLayer* layer : {&query_layer_, &key_layer_, &value_layer_}) {
            if (output->overlaps(layer->get_weight(0))) {
                throw std::invalid_argument("QKV outputs must not overwrite any projection weight");
            }
        }
    }
    if (query.overlaps(key) || query.overlaps(value)) {
        throw std::invalid_argument("Q must not overlap the current K/V slots");
    }
    // 防止调用者把某层已缓存的 K/V 当作 Q 工作区，破坏历史。
    // 未提交的其他槽位仍由调用者管理，正常使用时应单独分配 query。
    for (int32_t i = 0; i < cache.layer_num(); ++i) {
        auto [keys, values] = cache.history(i);
        if (query.overlaps(keys) || query.overlaps(values)) {
            throw std::invalid_argument("Q must not overwrite committed KV history");
        }
    }

    // 第三步：这三行就是课件 attention_qkv 的核心。
    // MatmulLayer::forward -> compute -> 第 12/13 课 CPU/CUDA GEMV。
    // 同一 stream 保证 Q、K、V 的提交顺序；无需新增重复的 .cu 内核。
    query_layer_.forward(stream);  // Wq * rmsnorm_output -> 独立 query
    key_layer_.forward(stream);    // Wk * rmsnorm_output -> key_cache 当前行
    value_layer_.forward(stream);  // Wv * rmsnorm_output -> value_cache 当前行

    // 第四步：把原位 K/V view 交给调用者。先做 RoPE(Q,K)，再 commit；
    // V 不旋转。这里不提前增加 valid_length，也不复制已有历史。
    return {key, value};
}

}  // namespace mini_infer
