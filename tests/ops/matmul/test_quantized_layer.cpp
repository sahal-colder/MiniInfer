#include "test_support.h"
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/ops/matmul/mini_infer_matmul.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 用已学过的 kernel 函数指针检查非法调用，不引入模板或捕获 lambda。
void expect_invalid_gemv(QuantizedSgemvKernel kernel, const Tensor& input,
                         const QuantizedTensor& weight, Tensor& output, float scale) {
    bool was_rejected = false;
    try {
        kernel(input, weight, output, scale, nullptr);
    } catch (const std::invalid_argument&) {
        was_rejected = true;
    }
    require(was_rejected, "invalid GEMV call was accepted");
}

void verify_invalid_inputs() {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    Tensor source(DataType::FLOAT32, {3, 3}, cpu);
    for (std::size_t i = 0; i < source.size(); ++i) source.index<float>(i) = float(i + 1);
    const QuantizedTensor weight = quantize_symmetric(source, 3);
    Tensor x(DataType::FLOAT32, {3}, cpu);
    for (std::size_t i = 0; i < x.size(); ++i) x.index<float>(i) = 1;
    Tensor y(DataType::FLOAT32, {3}, cpu);
    const QuantizedSgemvKernel kernel = get_quantized_sgemv_kernel(DeviceType::CPU);
    expect_invalid_gemv(kernel, Tensor{}, weight, y, 1);
    Tensor integer_input(DataType::INT32, {3}, cpu);
    expect_invalid_gemv(kernel, integer_input, weight, y, 1);
    Tensor wrong_width(DataType::FLOAT32, {2}, cpu);
    expect_invalid_gemv(kernel, wrong_width, weight, y, 1);
    Tensor wrong_output(DataType::FLOAT32, {1, 3}, cpu);
    expect_invalid_gemv(kernel, x, weight, wrong_output, 1);
    expect_invalid_gemv(kernel, x, weight, x, 1);
    Tensor scale_alias = weight.scales;
    expect_invalid_gemv(kernel, x, weight, scale_alias, 1);
    expect_invalid_gemv(kernel, x, weight, y, std::numeric_limits<float>::infinity());
    expect_invalid_gemv(quantized_sgemv_kernel_cuda, x, weight, y, 1);
    {
        bool was_rejected = false;
        try {
            (void)get_quantized_sgemv_kernel(DeviceType::UNKNOWN);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    QuantizedTensor bad = weight;
    bad.group_size = 0;
    expect_invalid_gemv(kernel, x, bad, y, 1);
    bad = weight;
    bad.scales = Tensor(DataType::FLOAT32, {2}, cpu);
    expect_invalid_gemv(kernel, x, bad, y, 1);
    bad = weight;
    bad.values = source;  // FP32 权重不能传进 INT8 算子。
    expect_invalid_gemv(kernel, x, bad, y, 1);
    bad = weight;
    bad.scales = weight.scales.clone(cpu);  // 深拷贝，避免改坏正常权重。
    for (const float invalid : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()}) {
        bad.scales.index<float>(0) = invalid;
        expect_invalid_gemv(kernel, x, bad, y, 1);
    }
    // 在足够大的真正 storage 内同时借用 INT8 权重和 FP32 输出，
    // 验证不同 dtype 的重叠也能被发现，检查之前不触碰借用的数值。
    Tensor storage(DataType::FLOAT32, {12}, cpu);
    bad = weight;
    bad.values = Tensor::borrow(DataType::INT8, {3, 3}, storage.ptr<float>(), DeviceType::CPU);
    Tensor alias_y = Tensor::borrow(DataType::FLOAT32, {3}, storage.ptr<float>(), DeviceType::CPU);
    expect_invalid_gemv(kernel, x, bad, alias_y, 1);

}

// 参数绑定只验证共享 Buffer 生命周期和失败时保留原参数。
void verify_layer_lifetime() {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    Tensor source(DataType::FLOAT32, {3, 3}, cpu);
    for (std::size_t i = 0; i < source.size(); ++i) source.index<float>(i) = float(i + 1);
    const QuantizedTensor weight = quantize_symmetric(source, 3);
    Tensor x(DataType::FLOAT32, {3}, cpu);
    for (std::size_t i = 0; i < x.size(); ++i) x.index<float>(i) = 1;
    Tensor y(DataType::FLOAT32, {3}, cpu);
    const QuantizedSgemvKernel kernel = get_quantized_sgemv_kernel(DeviceType::CPU);
    QuantizedMatmulLayer layer(DeviceType::CPU, 3, 3);
    layer.set_input(0, x);
    layer.set_output(0, y);
    {
        bool was_rejected = false;
        try {
            layer.forward();
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }  // 没有绑定权重不能运行。
    {
        const QuantizedTensor local = quantize_symmetric(source, 3);
        layer.set_quantized_weight(local);
    }  // local 的 Tensor 已析构，但层共享两个 Buffer，权重仍有效。
    layer.forward();
    Tensor reference(DataType::FLOAT32, {3}, cpu);
    kernel(x, weight, reference, 1, nullptr);
    for (std::size_t i = 0; i < y.size(); ++i)
        require(y.index<float>(i) == reference.index<float>(i), "layer lost weight ownership");
    QuantizedTensor bad = weight;
    bad.values = Tensor(DataType::INT8, {1, 9}, cpu);
    {
        bool was_rejected = false;
        try {
            layer.set_quantized_weight(bad);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    // 绑定失败保留旧权重，不能让层进入半更新状态。
    layer.forward();
    for (std::size_t i = 0; i < y.size(); ++i)
        require(y.index<float>(i) == reference.index<float>(i), "failed binding replaced weights");
}


}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        (void)use_cuda;
        verify_invalid_inputs();
        verify_layer_lifetime();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
