#pragma once

#include "mini_infer/core/mini_infer_tensor.h"

#include <cstddef>
#include <memory>

namespace mini_infer {

// 第 7 课的权重表示：每个量化值是 int8，每 group_size 个值共享
// 一个 FP32 scale。values 和 scales 分开保存，正好对应课程里的
// [int8 weights][scales] 文件布局；文件序列化留给后续模型加载课程。
struct QuantizedTensor {
    Tensor values;
    Tensor scales;
    std::size_t group_size = 0;

    std::size_t group_count() const {
        return scales.size();
    }

    // 第 13 课：量化值和分组 scale 必须一起迁移，group_size 保持不变。
    // 返回 Tensor 继续用 Buffer/allocator 管理所有权，不借用临时主机数组。
    // 带 stream 时，同样遵守 Tensor 的同步/存活约定。
    QuantizedTensor to_cuda(std::shared_ptr<DeviceAllocator> allocator,
                            cudaStream_t stream = nullptr) const;
    QuantizedTensor to_cpu(cudaStream_t stream = nullptr) const;
};

struct QuantizationError {
    float max_abs = 0.0f;
    float rmse = 0.0f;
};

// 对 CPU FP32 Tensor 做 per-group symmetric quantization：
//   scale[g] = max(abs(x[g])) / 127
//   q[i]     = round(x[i] / scale[g])，并限制在 [-127, 127]
// 全零组使用 scale=1，避免除零；最后一个不足 group_size 的组也会
// 保留一个独立 scale，而不是越界读取或静默丢弃数据。
QuantizedTensor quantize_symmetric(
    const Tensor& source,
    std::size_t group_size,
    std::shared_ptr<DeviceAllocator> value_allocator = nullptr,
    std::shared_ptr<DeviceAllocator> scale_allocator = nullptr
);

// 将量化结果恢复成 source 原来的 FP32 shape。当前版本明确只支持
// CPU Tensor；第 13 课量化 GEMV 则在 GPU 内核中逐元素反量化并立即乘加。
Tensor dequantize_symmetric(
    const QuantizedTensor& quantized,
    std::shared_ptr<DeviceAllocator> allocator = nullptr
);

QuantizationError measure_quantization_error(
    const Tensor& source,
    const Tensor& restored
);

}  // namespace mini_infer
