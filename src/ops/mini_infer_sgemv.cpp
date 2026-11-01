// 第 13 课第 2 步：只检查参数并选择后端，不在这里展开计算公式。
#include "mini_infer/ops/mini_infer_sgemv.h"
#include <stdexcept>
namespace mini_infer {
namespace detail {
void check_sgemv_tensors(const Tensor& input, const Tensor& weight,
                        const Tensor& output, DeviceType device) {
    // 元数据检查必须先于指针访问。直接调用后端函数也得到同样保护，
    // 不要求调用者必须先经过 MatmulLayer::check。
    for (const Tensor* tensor : {&input, &weight, &output}) {
        if (tensor->empty() || tensor->dims().empty() || tensor->size() == 0) {
            throw std::invalid_argument("Sgemv requires allocated, nonempty tensors");
        }
        if (tensor->data_type() != DataType::FLOAT32 ||
            tensor->device_type() != device) {
            throw std::invalid_argument("Sgemv requires FP32 tensors on its backend");
        }
    }
    if (weight.dims().size() != 2 || input.dims().back() != weight.dim(1)) {
        throw std::invalid_argument("Sgemv expects weight=[out,in], input=[...,in]");
    }
    std::vector<int32_t> expected_output = input.dims();
    expected_output.back() = weight.dim(0);
    if (output.dims() != expected_output) {
        throw std::invalid_argument("Sgemv expects output=[...,out]");
    }
    // CPU BLAS 与 CUDA 并行计算都可能先写一部分输出，再读另一部分输入。
    // 拒绝内存区间重叠，不能只比较两个 Tensor 对象是否相同。
    if (output.overlaps(input) || output.overlaps(weight)) {
        throw std::invalid_argument("Sgemv output must not overlap input or weight");
    }
}

}  // namespace detail

SgemvKernel get_sgemv_kernel(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return sgemv_kernel_cpu;
        case DeviceType::CUDA: return sgemv_kernel_cuda;
        default: throw std::invalid_argument("Unknown Sgemv backend");
    }
}

}  // namespace mini_infer
