// 第 15 课最后再看：由已有 ModelConfig 推导 GQA 宽度；不读取文件。
#include "mini_infer/tensor/mini_infer_kv_cache.h"
#include "mini_infer/loader/mini_infer_mmap.h"
#include <stdexcept>
namespace mini_infer {
namespace {

int32_t kv_dim_from_config(const ModelConfig& config) {
    if (config.dim <= 0 || config.head_num <= 0 || config.kv_head_num <= 0 ||
        config.dim % config.head_num != 0 ||
        config.head_num % config.kv_head_num != 0) {
        throw std::invalid_argument("KVCache requires valid MHA/GQA head dimensions");
    }
    // kv_head_num <= head_num，因此结果 <= dim，不会发生 int32_t 溢出。
    return (config.dim / config.head_num) * config.kv_head_num;
}

}  // namespace

// 委托给上面的构造函数，避免重复分配逻辑。std::move 转移局部 shared_ptr，
// 不移动底层 allocator 对象；两个 Buffer 会分别保留其所有权。
KVCache::KVCache(const ModelConfig& config, std::shared_ptr<DeviceAllocator> allocator)
    : KVCache(config.layer_num, config.seq_len, kv_dim_from_config(config),
               std::move(allocator)) {}

}  // namespace mini_infer
