#pragma once

#include <cuda_runtime_api.h>

namespace mini_infer {
// CUDA 错误通过 std::runtime_error 交给调用者处理。
void check_cuda(cudaError_t error, const char* message);
}  // namespace mini_infer
