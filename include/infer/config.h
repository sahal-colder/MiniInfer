#pragma once

// Edit these defaults, then configure CMake in a fresh build directory.
// A configured/installed library supplies its exact settings here.
#if __has_include("infer/build_config.h")
#include "infer/build_config.h"
#endif

#ifndef INFER_ENABLE_CUDA
#define INFER_ENABLE_CUDA 1
#endif
#ifndef INFER_ENABLE_NATIVE
#define INFER_ENABLE_NATIVE 1
#endif
#ifndef INFER_ENABLE_VLLM
#define INFER_ENABLE_VLLM 0
#endif
#ifndef INFER_MEMORY_MODE
#define INFER_MEMORY_MODE 2
#endif

#if (INFER_ENABLE_NATIVE != 0 && INFER_ENABLE_NATIVE != 1) ||                                      \
    (INFER_ENABLE_VLLM != 0 && INFER_ENABLE_VLLM != 1)
#error "Backend macros must be 0 or 1."
#endif
#if INFER_MEMORY_MODE < 0 || INFER_MEMORY_MODE > 2
#error "INFER_MEMORY_MODE must be 0 (SYNC), 1 (ASYNC), or 2 (POOL)."
#endif
