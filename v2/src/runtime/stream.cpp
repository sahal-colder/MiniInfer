/* CUDA adapter boundary: logical runtime code has no CUDA API calls. */
#include "stream.h"
#include <system_error>
#include <cstdio>
#if INFER_HAS_CUDA
#include <cuda_runtime.h>
#endif
namespace infer_c {
Stream::Stream(int id) : device(id) {
#if INFER_HAS_CUDA
    int previous = 0;
    auto rc = cudaGetDevice(&previous);
    if (rc != cudaSuccess) throw std::system_error(EIO,std::generic_category(),"CUDA current device");
    if (previous != id) rc = cudaSetDevice(id);
    cudaStream_t stream = nullptr;
    if (rc == cudaSuccess) rc = cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    if (previous != id) {
        auto restore = cudaSetDevice(previous);
        if (restore != cudaSuccess) {
            if (stream) {
                auto cleanup = cudaStreamDestroy(stream);
                if (cleanup != cudaSuccess) std::fputs("infer: CUDA stream rollback failed\n",stderr);
            }
            throw std::system_error(EIO,std::generic_category(),"CUDA device restoration");
        }
    }
    if (rc != cudaSuccess) throw std::system_error(EIO,std::generic_category(),"CUDA consumer stream");
    value = stream;
#else
    throw std::system_error(ENOTSUP,std::generic_category());
#endif
}
Stream::~Stream() {
#if INFER_HAS_CUDA
    if (!value) return;
    int previous = 0;
    auto rc = cudaGetDevice(&previous);
    if (rc != cudaSuccess) {
        std::fputs("infer: CUDA stream cleanup cannot select device\n",stderr);
        return;
    }
    if (rc == cudaSuccess) rc = cudaSetDevice(device);
    if (rc == cudaSuccess) rc = cudaStreamDestroy(static_cast<cudaStream_t>(value));
    if (previous != device) {
        auto restore = cudaSetDevice(previous);
        if (restore != cudaSuccess) std::fputs("infer: CUDA device restoration failed\n",stderr);
    }
    if (rc != cudaSuccess) std::fputs("infer: CUDA consumer stream cleanup failed\n",stderr);
#endif
}
}
