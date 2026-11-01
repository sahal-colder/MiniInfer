// CUDA types remain opaque in a host-only build.
#pragma once

#include "infer/config.h"

#if INFER_ENABLE_CUDA
#include <cuda_runtime.h>

namespace infer {
void check_cuda(cudaError_t err, const char* msg);
} // namespace infer
#else
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
struct CUevent_st;
typedef struct CUevent_st* cudaEvent_t;
#endif
