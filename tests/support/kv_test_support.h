#pragma once
#include "test_support.h"
namespace mini_infer_test {
void wait(mini_infer::DeviceType device, cudaStream_t stream);
mini_infer::Tensor input(std::vector<int32_t> shape, float first);
mini_infer::Tensor on_device(const mini_infer::Tensor& source, std::shared_ptr<mini_infer::DeviceAllocator> allocator);
void equal(const mini_infer::Tensor& actual, const mini_infer::Tensor& expected);
void all_zero(const mini_infer::Tensor& tensor);
}
