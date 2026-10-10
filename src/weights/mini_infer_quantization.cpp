#include "mini_infer/weights/mini_infer_quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace mini_infer {
namespace {

std::shared_ptr<DeviceAllocator> make_cpu_allocator(
    std::shared_ptr<DeviceAllocator> allocator
) {
    if (allocator) return allocator;
    return std::make_shared<CPUDeviceAllocator>();
}

void require_cpu_fp32(const Tensor& tensor, const char* name) {
    if (tensor.empty() || tensor.device_type() != DeviceType::CPU ||
        tensor.data_type() != DataType::FLOAT32) {
        throw std::invalid_argument(
            std::string(name) + " must be a CPU FP32 Tensor"
        );
    }
}

}  // namespace

QuantizedTensor QuantizedTensor::to_cuda(std::shared_ptr<DeviceAllocator> allocator,
                                        cudaStream_t stream) const {
    return {values.to_cuda(allocator, stream), scales.to_cuda(allocator, stream),
            group_size};
}

QuantizedTensor QuantizedTensor::to_cpu(cudaStream_t stream) const {
    return {values.to_cpu(stream), scales.to_cpu(stream), group_size};
}

QuantizedTensor quantize_symmetric(
    const Tensor& source,
    std::size_t group_size,
    std::shared_ptr<DeviceAllocator> value_allocator,
    std::shared_ptr<DeviceAllocator> scale_allocator
) {
    require_cpu_fp32(source, "quantization source");
    if (group_size == 0) {
        throw std::invalid_argument("quantization group_size must be positive");
    }

    // 不使用 n+group_size-1，避免合法的大 group_size 使加法溢出。
    const std::size_t groups = source.size() / group_size +
                               (source.size() % group_size != 0);
    if (groups > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error("quantization scale count exceeds Tensor dimensions");
    }
    auto values_allocator = make_cpu_allocator(std::move(value_allocator));
    auto scales_allocator = make_cpu_allocator(std::move(scale_allocator));
    if (values_allocator->device_type() != DeviceType::CPU ||
        scales_allocator->device_type() != DeviceType::CPU) {
        throw std::invalid_argument(
            "quantization currently requires CPU allocators"
        );
    }

    QuantizedTensor result{
        Tensor(DataType::INT8, source.dims(), values_allocator),
        Tensor(DataType::FLOAT32, {static_cast<int32_t>(groups)}, scales_allocator),
        group_size
    };

    const float* input = source.ptr<float>();
    int8_t* output = result.values.ptr<int8_t>();
    float* scales = result.scales.ptr<float>();

    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t begin = group * group_size;
        const std::size_t end = begin + std::min(group_size, source.size() - begin);
        float max_abs = 0.0f;
        for (std::size_t i = begin; i < end; ++i) {
            if (!std::isfinite(input[i])) {
                throw std::invalid_argument(
                    "quantization source contains NaN or infinity"
                );
            }
            max_abs = std::max(max_abs, std::fabs(input[i]));
        }

        // scale=1 for a zero group gives q=0 and makes dequantization safe.
        const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
        scales[group] = scale;
        for (std::size_t i = begin; i < end; ++i) {
            const long rounded = std::lround(input[i] / scale);
            const long clamped = std::max<long>(-127, std::min<long>(127, rounded));
            output[i] = static_cast<int8_t>(clamped);
        }
    }

    return result;
}

Tensor dequantize_symmetric(
    const QuantizedTensor& quantized,
    std::shared_ptr<DeviceAllocator> allocator
) {
    if (quantized.group_size == 0 || quantized.values.empty() ||
        quantized.scales.empty() ||
        quantized.values.device_type() != DeviceType::CPU ||
        quantized.scales.device_type() != DeviceType::CPU ||
        quantized.values.data_type() != DataType::INT8 ||
        quantized.scales.data_type() != DataType::FLOAT32) {
        throw std::invalid_argument("invalid CPU INT8 quantized tensor");
    }

    const std::size_t expected_groups = quantized.values.size() / quantized.group_size +
                                       (quantized.values.size() % quantized.group_size != 0);
    if (quantized.scales.size() != expected_groups) {
        throw std::invalid_argument("quantized scale count does not match values");
    }

    auto output_allocator = make_cpu_allocator(std::move(allocator));
    if (output_allocator->device_type() != DeviceType::CPU) {
        throw std::invalid_argument("dequantization currently requires CPU allocator");
    }
    Tensor result(DataType::FLOAT32, quantized.values.dims(), output_allocator);
    const int8_t* values = quantized.values.ptr<int8_t>();
    const float* scales = quantized.scales.ptr<float>();
    float* output = result.ptr<float>();
    for (std::size_t i = 0; i < result.size(); ++i) {
        const std::size_t group = i / quantized.group_size;
        if (!std::isfinite(scales[group]) || scales[group] <= 0.0f) {
            throw std::invalid_argument("quantized scale must be finite and positive");
        }
        output[i] = static_cast<float>(values[i]) * scales[group];
    }
    return result;
}

QuantizationError measure_quantization_error(
    const Tensor& source,
    const Tensor& restored
) {
    require_cpu_fp32(source, "error source");
    require_cpu_fp32(restored, "error result");
    if (source.dims() != restored.dims()) {
        throw std::invalid_argument("quantization error needs equal shapes");
    }

    const float* lhs = source.ptr<float>();
    const float* rhs = restored.ptr<float>();
    long double squared = 0.0L;
    QuantizationError result;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const float error = std::fabs(lhs[i] - rhs[i]);
        result.max_abs = std::max(result.max_abs, error);
        squared += static_cast<long double>(error) * error;
    }
    result.rmse = static_cast<float>(
        std::sqrt(squared / static_cast<long double>(source.size()))
    );
    return result;
}

}  // namespace mini_infer
