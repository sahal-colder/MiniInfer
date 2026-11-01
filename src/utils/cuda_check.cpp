// 把非成功 CUDA 状态转换为原有 runtime_error。
#include "infer/utils/cuda_check.h"

#include <stdexcept>
#include <string>

namespace infer {

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg ? msg : "CUDA operation") + ": " +
                                 cudaGetErrorString(err));
    }
}

} // namespace infer
