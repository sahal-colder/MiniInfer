#include "test_support.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/ops/matmul/mini_infer_sgemv.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 用已学过的 kernel 函数指针检查非法调用，不引入模板或捕获 lambda。
void expect_invalid_gemv(SgemvKernel kernel, const Tensor& input,
                         const Tensor& weight, Tensor& output, float scale) {
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
    Tensor x(DataType::FLOAT32, {3}, cpu);
    Tensor w(DataType::FLOAT32, {3, 3}, cpu);
    Tensor y(DataType::FLOAT32, {3}, cpu);
    const SgemvKernel kernel = get_sgemv_kernel(DeviceType::CPU);
    Tensor empty;
    Tensor unallocated(DataType::FLOAT32, {3});
    Tensor integers(DataType::INT32, {3}, cpu);
    Tensor wrong_input(DataType::FLOAT32, {2}, cpu);
    Tensor wrong_output(DataType::FLOAT32, {1, 3}, cpu);
    Tensor rank3_weight(DataType::FLOAT32, {1, 3, 3}, cpu);
    expect_invalid_gemv(kernel, empty, w, y, 1);
    expect_invalid_gemv(kernel, unallocated, w, y, 1);
    expect_invalid_gemv(kernel, integers, w, y, 1);
    expect_invalid_gemv(kernel, x, rank3_weight, y, 1);
    expect_invalid_gemv(kernel, wrong_input, w, y, 1);
    expect_invalid_gemv(kernel, x, w, wrong_output, 1);
    expect_invalid_gemv(kernel, x, w, x, 1);
    Tensor weight_alias = Tensor::borrow(DataType::FLOAT32, {3},
                                        w.ptr<float>() + 1, DeviceType::CPU);
    expect_invalid_gemv(kernel, x, w, weight_alias, 1);
    {
        bool was_rejected = false;
        try {
            (void)get_sgemv_kernel(DeviceType::UNKNOWN);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    // GPU wrapper 必须在发射 kernel 前拒绝 CPU 指针，故无 GPU 也能检查。
    expect_invalid_gemv(sgemv_kernel_cuda, x, w, y, 1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        (void)use_cuda;
        verify_invalid_inputs();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
