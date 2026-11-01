#pragma once
#include "infer/config.h"
#include "infer/core/memory.h"
#include "infer/cpu/cpu_memory_pool.h"
#include "infer/cpu/pinned_memory_pool.h"
#include "infer/cuda/cuda_memory_pool.h"
#include "infer/storage/kv_offload_cache.h"
#if INFER_ENABLE_NATIVE
#include "infer/integrations/native.h"
#include "mini_infer/runtime/mini_infer_generation.h"
#endif
