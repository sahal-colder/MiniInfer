#include "mini_infer/base/cuda_error.h"
#include <stdexcept>
#include <string>

namespace mini_infer {
void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message ? message : "CUDA operation") + ": " +
                                 cudaGetErrorString(error));
    }
}

}  // namespace mini_infer
