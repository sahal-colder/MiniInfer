#include "test_support.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/ops/matmul/mini_infer_matmul.h"
#include "mini_infer/ops/matmul/mini_infer_sgemv.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 浮点加法次序可能不同：用绝对/相对容差比较独立参考，同时拒绝 NaN/Inf。
void compare(const Tensor& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size(), "quantized output length mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double value = actual.index<float>(i);
        if (!std::isfinite(value) ||
            std::abs(value - expected[i]) > 2e-4 + 5e-5 * std::abs(expected[i])) {
            throw std::runtime_error("quantized result mismatch at " + std::to_string(i));
        }
    }
}

// GPU 比较独立成一个步骤；调用者提供上面已经手算/计算好的期望结果。
void verify_cuda_backends(const Tensor& input, const QuantizedTensor& weight,
                          float output_scale, cudaStream_t stream,
                          const std::vector<double>& expected,
                          const std::vector<double>& hand_expected) {
    const int32_t in = weight.values.dim(1);
    const int32_t out = weight.values.dim(0);
    std::vector<int32_t> shape = input.dims();
    shape.back() = out;
    std::shared_ptr<CUDADeviceAllocator> gpu = std::make_shared<CUDADeviceAllocator>();
    Tensor dx = input.to_cuda(gpu, stream);
    QuantizedTensor dw = weight.to_cuda(gpu, stream);
    Tensor dy(DataType::FLOAT32, shape, gpu);
    // 两个实际 CUDA kernel：每块一行、每块最多四行，后者要截断尾行。
    for (const QuantizedSgemvKernel kernel : {
             get_quantized_sgemv_kernel(DeviceType::CUDA),
             quantized_sgemv_kernel_cuda_rows4}) {
        kernel(dx, dw, dy, output_scale, stream);
        check_cuda(cudaStreamSynchronize(stream), "wait for quantized Sgemv");
        Tensor result = dy.to_cpu();
        compare(result, expected);
        if (!hand_expected.empty()) compare(result, hand_expected);
    }
    QuantizedMatmulLayer gpu_layer(DeviceType::CUDA, int32_t(out), int32_t(in), output_scale);
    gpu_layer.set_input(0, dx);
    gpu_layer.set_output(0, dy);
    {
        bool was_rejected = false;
        try {
            gpu_layer.set_quantized_weight(weight);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    gpu_layer.set_quantized_weight(dw);
    gpu_layer.forward(stream);
    check_cuda(cudaStreamSynchronize(stream), "wait for quantized MatmulLayer");
    compare(dy.to_cpu(), expected);

    // values/scales 一起往返迁移，group_size 不因设备切换丢失。
    const QuantizedTensor roundtrip = dw.to_cpu();
    require(roundtrip.group_size == weight.group_size &&
                roundtrip.values.dims() == weight.values.dims() &&
                roundtrip.scales.dims() == weight.scales.dims(),
            "quantized migration lost metadata");
    for (std::size_t i = 0; i < weight.values.size(); ++i)
        require(roundtrip.values.index<int8_t>(i) == weight.values.index<int8_t>(i),
                "quantized migration changed INT8 values");
    for (std::size_t i = 0; i < weight.scales.size(); ++i)
        require(roundtrip.scales.index<float>(i) == weight.scales.index<float>(i),
                "quantized migration changed scales");
}

// 量化误差与归约误差分开：按反量化后的权重计算 double 参考。
void verify_backends(const Tensor& input, const QuantizedTensor& weight,
                     float output_scale, bool use_cuda, cudaStream_t stream,
                     const std::vector<double>& hand_expected = {}) {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    const Tensor restored = dequantize_symmetric(weight, cpu);
    const std::size_t in = std::size_t(weight.values.dim(1));
    const std::size_t out = std::size_t(weight.values.dim(0));
    std::vector<int32_t> shape = input.dims();
    shape.back() = int32_t(out);
    Tensor y(DataType::FLOAT32, shape, cpu);
    Tensor fp32_y(DataType::FLOAT32, shape, cpu);
    std::vector<double> expected(y.size());
    for (std::size_t row = 0; row < input.size() / in; ++row) {
        for (std::size_t o = 0; o < out; ++o) {
            double sum = 0;
            for (std::size_t i = 0; i < in; ++i)
                sum += double(input.index<float>(row * in + i)) *
                       double(restored.index<float>(o * in + i));
            expected[row * out + o] = sum * output_scale;
        }
    }
    get_quantized_sgemv_kernel(DeviceType::CPU)(input, weight, y, output_scale, nullptr);
    get_sgemv_kernel(DeviceType::CPU)(input, restored, fp32_y, output_scale, nullptr);
    compare(y, expected);
    compare(fp32_y, expected);
    if (!hand_expected.empty()) compare(y, hand_expected);

    QuantizedMatmulLayer layer(DeviceType::CPU, int32_t(out), int32_t(in), output_scale);
    layer.set_input(0, input);
    layer.set_output(0, y);
    layer.set_quantized_weight(weight);
    layer.forward();
    compare(y, expected);
    if (!use_cuda) return;

    verify_cuda_backends(input, weight, output_scale, stream, expected, hand_expected);
}

// 固定小矩阵可以手算；另一组精确 INT8 输入验证行列布局。
void verify_course(bool use_cuda) {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    Tensor source(DataType::FLOAT32, {2, 2}, cpu);
    const float values[]{1, 3, 5, -1};
    for (std::size_t i = 0; i < source.size(); ++i) source.index<float>(i) = values[i];
    const QuantizedTensor weight = quantize_symmetric(source, 2);
    const int expected_q[]{42, 127, 127, -25};
    for (std::size_t i = 0; i < source.size(); ++i)
        require(int(weight.values.index<int8_t>(i)) == expected_q[i],
                "course INT8 quantization mismatch");
    std::cout << "[lesson 14] quantization [1,3,5,-1] -> [42,127,127,-25], "
                 "scales=[3/127,5/127]\n";
    Tensor x(DataType::FLOAT32, {2}, cpu);
    x.index<float>(0) = 1; x.index<float>(1) = 1;
    verify_backends(x, weight, 1.0f, use_cuda, nullptr,
                    {169.0 * (3.0 / 127.0), 102.0 * (5.0 / 127.0)});
    Tensor restored = dequantize_symmetric(weight);
    const QuantizationError error = measure_quantization_error(source, restored);
    std::cout << "[lesson 14] original FP32 GEMV=[4,4], quantized GEMV=["
              << 169.0 * (3.0 / 127.0) << ',' << 102.0 * (5.0 / 127.0)
              << "], weight max_abs_error=" << error.max_abs << '\n';

    // 手动构造可精确表示的量化权重，保留课件 3×3 的 0/3/6 例子。
    // group_size=4 使分组跨行且最后只有一个元素，不能用 i/group_size。
    QuantizedTensor exact{Tensor(DataType::INT8, {3, 3}, cpu),
                          Tensor(DataType::FLOAT32, {3}, cpu), 4};
    for (std::size_t i = 0; i < exact.values.size(); ++i)
        exact.values.index<int8_t>(i) = int8_t(i + 1);
    for (std::size_t i = 0; i < exact.scales.size(); ++i) exact.scales.index<float>(i) = 1;
    Tensor course_x(DataType::FLOAT32, {3}, cpu);
    course_x.index<float>(0) = 1; course_x.index<float>(1) = 1; course_x.index<float>(2) = -1;
    verify_backends(course_x, exact, 1, use_cuda, nullptr, {0, 3, 6});
    std::cout << "[lesson 14] exact INT8 course GEMV: 0 3 6\n";
}

// 先独立用 double 计算期望值，再比较各后端；刻意覆盖地址偏移和行尾。
void verify_shape(std::vector<int32_t> shape, int32_t out, std::size_t group_size,
                  float scale, bool use_cuda, cudaStream_t stream, bool zero_group = false) {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    const int32_t in = shape.back();
    Tensor source(DataType::FLOAT32, {out, in}, cpu);
    Tensor x(DataType::FLOAT32, shape, cpu);
    for (std::size_t i = 0; i < source.size(); ++i)
        source.index<float>(i) = float(int((i * 7) % 31) - 15) * 0.13f;
    if (zero_group) {
        for (std::size_t i = 0; i < std::min(group_size, source.size()); ++i)
            source.index<float>(i) = 0;
    }
    for (std::size_t i = 0; i < x.size(); ++i)
        x.index<float>(i) = float(int((i * 3) % 17) - 8) * 0.07f;
    const QuantizedTensor weight = quantize_symmetric(source, group_size);
    if (zero_group) require(weight.scales.index<float>(0) == 1, "zero group scale mismatch");
    verify_backends(x, weight, scale, use_cuda, stream);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        verify_course(use_cuda);
        for (const int32_t in : {1, 3, 5, 31, 127, 128, 129, 257, 513, 4097})
            verify_shape({2, in}, 5, in == 1 ? 1 : 4, -0.5f, false, nullptr);
        verify_shape({2, 2, 5}, 3, 7, 1.25f, false, nullptr, true);
        verify_shape({1, 7}, 5, 1000, 0.5f, false, nullptr);
        verify_shape({1, 7}, 5, std::numeric_limits<std::size_t>::max(), 1, false, nullptr);
        verify_shape({1, 5}, 3, 4, 0, false, nullptr);
        // 无 GPU 时仍保留原有大输出维度和跨组边界检查。
        verify_shape({2, 1}, 131073, 128, 1, false, nullptr);
        if (use_cuda) {
            TestStream stream;
            for (const int32_t in : {1, 3, 5, 31, 127, 128, 129, 257, 513, 4097})
                verify_shape({2, in}, 5, in == 1 ? 1 : 4, -0.5f, true, stream.get());
            verify_shape({2, 2, 5}, 3, 7, 1.25f, true, stream.get(), true);
            verify_shape({1, 7}, 5, 1000, 0.5f, true, stream.get());
            verify_shape({1, 7}, 5, std::numeric_limits<std::size_t>::max(), 1, true, stream.get());
            verify_shape({1, 5}, 3, 4, 0, true, stream.get());
            verify_shape({2, 1}, 131073, 128, 1, true, stream.get());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
