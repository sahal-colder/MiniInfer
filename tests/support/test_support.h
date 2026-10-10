#pragma once
#include "mini_infer/core/mini_infer_tensor.h"
namespace mini_infer_test {
// 测试断言始终执行，不受 Release 构建的 NDEBUG 影响。
void require(bool condition, const char* message);
// --cpu-only 不探测 GPU；--cuda 要求 GPU，无设备时以 CTest skip code 77 退出。
bool test_uses_cuda(int argc, char** argv);
// TestStream 只拥有测试用 nonblocking stream；不拥有 Tensor。
// 计算函数仍须在局部 Tensor 析构前显式同步，stream 析构不能替代这一步。
class TestStream {
public:
    TestStream();
    ~TestStream();
    TestStream(const TestStream&) = delete;
    TestStream& operator=(const TestStream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};
}  // namespace mini_infer_test
