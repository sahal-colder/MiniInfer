#include "test_support.h"
#include <iostream>
#include <stdexcept>
#include <string>
namespace mini_infer_test {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool test_uses_cuda(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--cpu-only") return false;
    if (argc != 1 && !(argc == 2 && std::string(argv[1]) == "--cuda"))
        throw std::invalid_argument("usage: test [--cpu-only|--cuda]");
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaSuccess && count > 0) return true;
    if (status != cudaSuccess && status != cudaErrorNoDevice && status != cudaErrorInsufficientDriver)
        throw std::runtime_error(cudaGetErrorString(status));
    std::cout << "CUDA skipped: no available device/driver\n";
    if (argc == 2) std::exit(77);
    return false;
}
TestStream::TestStream() {
    mini_infer::check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create test stream");
}
TestStream::~TestStream() {
    mini_infer::check_cuda(cudaStreamSynchronize(stream_), "wait for test stream");
    mini_infer::check_cuda(cudaStreamDestroy(stream_), "destroy test stream");
}
}  // namespace mini_infer_test
