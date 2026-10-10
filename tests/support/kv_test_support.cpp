#include "kv_test_support.h"
#include <stdexcept>
#include <string>
#include <utility>
namespace mini_infer_test {
using namespace mini_infer;
void wait(DeviceType device, cudaStream_t stream) {
    if (device == DeviceType::CUDA)
        check_cuda(cudaStreamSynchronize(stream), "wait for KVCache demo");
}
Tensor input(std::vector<int32_t> shape, float first) {
    Tensor result(DataType::FLOAT32, std::move(shape),
                  std::make_shared<CPUDeviceAllocator>());
    for (std::size_t i = 0; i < result.size(); ++i)
        result.index<float>(i) = first + static_cast<float>(i % 997);
    return result;
}
Tensor on_device(const Tensor& source, std::shared_ptr<DeviceAllocator> allocator) {
    return allocator->device_type() == DeviceType::CPU ? source : source.to_cuda(allocator);
}
void equal(const Tensor& actual, const Tensor& expected) {
    // GPU 使用者先同步 stream，再通过同步拷贝取得 CPU 数据。
    const Tensor host = actual.to_cpu();
    require(host.dims() == expected.dims(), "KVCache shape mismatch");
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (host.index<float>(i) != expected.index<float>(i)) {
            throw std::runtime_error("KVCache value mismatch at " + std::to_string(i) +
                ": actual=" + std::to_string(host.index<float>(i)) +
                ", expected=" + std::to_string(expected.index<float>(i)));
        }
    }
}
void all_zero(const Tensor& tensor) {
    const Tensor host = tensor.to_cpu();
    for (std::size_t i = 0; i < host.size(); ++i)
        require(host.index<float>(i) == 0.0f, "KVCache clear missed an element");
}

}
