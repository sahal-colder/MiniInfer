#include "test_support.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/ops/matmul/mini_infer_sgemv.h"
#include "mini_infer/ops/matmul/mini_infer_matmul.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 浮点加法次序可能不同：用绝对/相对容差比较独立参考，同时拒绝 NaN/Inf。
void compare(const Tensor& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size(), "Sgemv result length mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double value = actual.index<float>(i);
        // CPU BLAS、标量累加、CUDA 树形归约的浮点加法顺序不同。
        // 使用绝对 + 相对容差，并显式拒绝 NaN/Inf，不能只做 > 比较。
        if (!std::isfinite(value) ||
            std::abs(value - expected[i]) > 1e-4 + 5e-5 * std::abs(expected[i])) {
            throw std::runtime_error("Sgemv numerical mismatch at " + std::to_string(i));
        }
    }
}

// 给出可手算的输入和结果，再确认已有 MatmulLayer 的调用没有改变。
void verify_course_example(bool use_cuda) {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    Tensor x(DataType::FLOAT32, {3}, cpu);
    Tensor w(DataType::FLOAT32, {3, 3}, cpu);
    Tensor y(DataType::FLOAT32, {3}, cpu);
    x.index<float>(0) = 1;
    x.index<float>(1) = 1;
    x.index<float>(2) = -1;
    for (std::size_t i = 0; i < w.size(); ++i) w.index<float>(i) = float(i + 1);

    // 课件的独立手算结果，能直接发现权重误转置。
    get_sgemv_kernel(DeviceType::CPU)(x, w, y, 1.0f, nullptr);
    compare(y, {0, 3, 6});
    std::cout << "[lesson 13] CPU course result: " << y.index<float>(0) << ' '
              << y.index<float>(1) << ' ' << y.index<float>(2) << '\n';

    // 原有 MatmulLayer 也经过新的独立 Sgemv 实现。
    MatmulLayer layer(DeviceType::CPU, 3, 3);
    layer.set_input(0, x);
    layer.set_weight(0, w);
    layer.set_output(0, y);
    layer.forward();
    compare(y, {0, 3, 6});

    // 非方阵 W=[2,3]、x=[7,8,9]：课件中行/列主序转换的例子。
    Tensor rectangular(DataType::FLOAT32, {2, 3}, cpu);
    Tensor rectangular_y(DataType::FLOAT32, {2}, cpu);
    for (std::size_t i = 0; i < rectangular.size(); ++i)
        rectangular.index<float>(i) = float(i + 1);
    for (std::size_t i = 0; i < x.size(); ++i) x.index<float>(i) = float(i + 7);
    get_sgemv_kernel(DeviceType::CPU)(x, rectangular, rectangular_y, 0.5f, nullptr);
    compare(rectangular_y, {25, 61});  // 原始内积 50、122，再乘 0.5。

    if (use_cuda) {
        // 恢复 3×3 课件输入，并把真正的数据复制到 GPU。
        x.index<float>(0) = 1; x.index<float>(1) = 1; x.index<float>(2) = -1;
        std::shared_ptr<CUDADeviceAllocator> gpu = std::make_shared<CUDADeviceAllocator>();
        Tensor dx = x.to_cuda(gpu);
        Tensor dw = w.to_cuda(gpu);
        Tensor dy(DataType::FLOAT32, {3}, gpu);
        get_sgemv_kernel(DeviceType::CUDA)(dx, dw, dy, 1.0f, nullptr);
        check_cuda(cudaStreamSynchronize(nullptr), "wait for course Sgemv");
        Tensor result = dy.to_cpu();
        compare(result, {0, 3, 6});
        std::cout << "[lesson 13] CUDA course result: " << result.index<float>(0)
                  << ' ' << result.index<float>(1) << ' ' << result.index<float>(2) << '\n';
        MatmulLayer gpu_layer(DeviceType::CUDA, 3, 3);
        gpu_layer.set_input(0, dx);
        gpu_layer.set_weight(0, dw);
        gpu_layer.set_output(0, dy);
        gpu_layer.forward();
        check_cuda(cudaStreamSynchronize(nullptr), "wait for CUDA MatmulLayer");
        compare(dy.to_cpu(), {0, 3, 6});
    }
}

// 先独立用 double 计算期望值，再比较各后端；刻意覆盖地址偏移和行尾。
void verify_shape(const std::vector<int32_t>& shape, int32_t out, float scale,
                  bool shifted, bool use_cuda, cudaStream_t stream) {
    const int32_t in = shape.back();
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    Tensor x_meta(DataType::FLOAT32, shape);
    Tensor w_meta(DataType::FLOAT32, {out, in});
    const int32_t offset = shifted ? 1 : 0;
    Tensor x_storage(DataType::FLOAT32, {int32_t(x_meta.size()) + offset}, cpu);
    Tensor w_storage(DataType::FLOAT32, {int32_t(w_meta.size()) + offset}, cpu);
    Tensor x = Tensor::borrow(DataType::FLOAT32, shape,
                             x_storage.ptr<float>() + offset, DeviceType::CPU);
    Tensor w = Tensor::borrow(DataType::FLOAT32, {out, in},
                             w_storage.ptr<float>() + offset, DeviceType::CPU);
    std::vector<int32_t> output_shape = shape;
    output_shape.back() = out;
    Tensor y(DataType::FLOAT32, output_shape, cpu);
    Tensor scalar_y(DataType::FLOAT32, output_shape, cpu);
    for (std::size_t i = 0; i < x.size(); ++i)
        x.index<float>(i) = float(int(i % 31) - 15) * 0.07f;
    for (std::size_t i = 0; i < w.size(); ++i)
        w.index<float>(i) = float(int((i * 7) % 23) - 11) * 0.03f;

    // 用 double 独立求期望值，避免 CPU/GPU 同样写错时互相证明正确。
    std::vector<double> expected(y.size());
    for (std::size_t row = 0; row < x.size() / std::size_t(in); ++row) {
        for (int32_t o = 0; o < out; ++o) {
            double sum = 0;
            for (int32_t i = 0; i < in; ++i)
                sum += double(x.index<float>(row * in + i)) *
                       double(w.index<float>(std::size_t(o) * in + i));
            expected[row * out + o] = sum * scale;
        }
    }
    sgemv_kernel_cpu_reference(x, w, scalar_y, scale, nullptr);
    get_sgemv_kernel(DeviceType::CPU)(x, w, y, scale, nullptr);
    compare(scalar_y, expected);
    compare(y, expected);
    if (!use_cuda) return;

    std::shared_ptr<CUDADeviceAllocator> gpu = std::make_shared<CUDADeviceAllocator>();
    Tensor dx_storage = x_storage.to_cuda(gpu);
    Tensor dw_storage = w_storage.to_cuda(gpu);
    // 先上传整个 storage 再偏移，保证 GPU 地址也真正错开 4 B。
    Tensor dx = Tensor::borrow(DataType::FLOAT32, shape,
                              dx_storage.ptr<float>() + offset, DeviceType::CUDA);
    Tensor dw = Tensor::borrow(DataType::FLOAT32, {out, in},
                              dw_storage.ptr<float>() + offset, DeviceType::CUDA);
    Tensor dy(DataType::FLOAT32, output_shape, gpu);
    for (const SgemvKernel kernel : {get_sgemv_kernel(DeviceType::CUDA),
                                     sgemv_kernel_cuda_manual, sgemv_kernel_cuda_rows4}) {
        kernel(dx, dw, dy, scale, stream);
        // stream 是 nonblocking 时，默认 stream 的拷贝不会替它等待。
        // 显式等传入 stream 完成，再读结果/释放借用 storage。
        check_cuda(cudaStreamSynchronize(stream), "wait for boundary Sgemv");
        compare(dy.to_cpu(), expected);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        verify_course_example(use_cuda);
        for (const int32_t in : {1, 3, 4, 5, 31, 32, 33, 127, 128, 129, 511, 512, 513, 4096, 4097})
            verify_shape({2, in}, 7, -0.5f, false, false, nullptr);
        verify_shape({1, 8}, 3, 0.5f, true, false, nullptr);
        verify_shape({2, 2, 5}, 3, 1.25f, false, false, nullptr);
        verify_shape({1, 5}, 3, 0.0f, false, false, nullptr);
        // 原 CPU-only 示例也覆盖大输出维度，不能依赖是否安装了 GPU。
        verify_shape({2, 1}, 32768, 1.0f, false, false, nullptr);
        if (use_cuda) {
            TestStream stream;
            for (const int32_t in : {1, 3, 4, 5, 31, 32, 33, 127, 128, 129, 511, 512, 513, 4096, 4097})
                verify_shape({2, in}, 7, -0.5f, false, true, stream.get());
            verify_shape({1, 8}, 3, 0.5f, true, true, stream.get());
            verify_shape({2, 2, 5}, 3, 1.25f, false, true, stream.get());
            verify_shape({1, 5}, 3, 0.0f, false, true, stream.get());
            verify_shape({2, 1}, 32768, 1.0f, false, true, stream.get());
            // rows4 的 grid-stride 必须超过 65535*4 个输出。
            verify_shape({2, 1}, 131073, 1.0f, false, true, stream.get());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
