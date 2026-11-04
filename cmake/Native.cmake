option(MINI_INFER_USE_ARMADILLO "Use Armadillo for CPU Sgemv and SwiGLU" ON)
option(MINI_INFER_BUILD_APP "Build the inference CLI" ON)

# 保留两个原有 target 及其链接关系；供 add_subdirectory 和安装包复用。
add_library(mini_infer_memory
    src/base/cuda_error.cpp
    src/base/mini_infer_memory.cpp
    src/base/mini_infer_memory_pool.cpp
    src/tensor/mini_infer_buffer.cpp
)
add_library(mini_infer_compute
    src/loader/mini_infer_mlp_weight_loader.cpp
    src/loader/mini_infer_mmap.cpp
    src/loader/mini_infer_qkv_weight_loader.cpp
    src/loader/mini_infer_weight_loader.cpp
    src/model/mini_infer_attention_qkv.cpp
    src/model/mini_infer_mlp.cpp
    src/model/mini_infer_paged_cache.cpp
    src/model/mini_infer_model.cpp
    src/model/mini_infer_safetensors.cpp
    src/model/mini_infer_model_weights.cpp
    src/model/mini_infer_transformer.cpp
    src/ops/cpu/mini_infer_add_cpu.cpp
    src/ops/cpu/mini_infer_attention_cpu.cpp
    src/ops/cpu/mini_infer_embedding_cpu.cpp
    src/ops/cpu/mini_infer_kernels_cpu.cpp
    src/ops/cpu/mini_infer_quantized_sgemv_cpu.cpp
    src/ops/cpu/mini_infer_sgemv_cpu.cpp
    src/ops/cpu/mini_infer_softmax_cpu.cpp
    src/ops/cpu/mini_infer_swiglu_cpu.cpp
    src/ops/cuda/mini_infer_add.cu
    src/ops/cuda/mini_infer_attention.cu
    src/ops/cuda/mini_infer_embedding.cu
    src/ops/cuda/mini_infer_kernels_cuda.cu
    src/ops/cuda/mini_infer_kv_cache_kernel.cu
    src/ops/cuda/mini_infer_prefill.cu
    src/ops/cuda/mini_infer_matmul_backend.cu
    src/ops/cuda/mini_infer_quantized_sgemv.cu
    src/ops/cuda/mini_infer_rope.cu
    src/ops/cuda/mini_infer_sgemv.cu
    src/ops/cuda/mini_infer_sgemv_cuda_manual.cu
    src/ops/cuda/mini_infer_sgemv_cuda_rows4.cu
    src/ops/cuda/mini_infer_softmax.cu
    src/ops/cuda/mini_infer_swiglu.cu
    src/ops/mini_infer_add.cpp
    src/ops/mini_infer_attention.cpp
    src/ops/mini_infer_backend.cpp
    src/ops/mini_infer_embedding.cpp
    src/ops/mini_infer_kv_cache_kernel.cpp
    src/ops/mini_infer_layer.cpp
    src/ops/mini_infer_matmul.cpp
    src/ops/mini_infer_matmul_backend.cpp
    src/ops/mini_infer_quantized_matmul.cpp
    src/ops/mini_infer_quantized_sgemv.cpp
    src/ops/mini_infer_quantized_sgemv_check.cpp
    src/ops/mini_infer_rope.cpp
    src/ops/mini_infer_sgemv.cpp
    src/ops/mini_infer_softmax.cpp
    src/ops/mini_infer_swiglu.cpp
    src/runtime/mini_infer_generation.cpp
    src/sampler/mini_infer_sampler.cpp
    src/tensor/mini_infer_kv_cache.cpp
    src/tensor/mini_infer_kv_cache_append.cpp
    src/tensor/mini_infer_kv_cache_config.cpp
    src/tensor/mini_infer_kv_cache_history.cpp
    src/tensor/mini_infer_quantization.cpp
    src/tensor/mini_infer_tensor.cpp
    src/tokenizer/mini_infer_tokenizer.cpp
)
# Compatibility composition with checkpoint adapters and CLI support.
add_library(mini_infer
    integrations/native/kv_cache_storage.cpp
    app/mini_infer_cli.cpp
)
add_library(MiniInfer::mini_infer_memory ALIAS mini_infer_memory)
add_library(MiniInfer::mini_infer ALIAS mini_infer)

foreach(target IN ITEMS mini_infer_memory mini_infer_compute mini_infer)
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_EXTENSIONS NO
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endforeach()
target_include_directories(mini_infer_compute PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mini_infer_memory PUBLIC infer_memory)
target_link_libraries(mini_infer_compute PUBLIC mini_infer_memory infer_io infer_storage)
target_link_libraries(mini_infer PUBLIC mini_infer_compute infer_storage)
set_target_properties(mini_infer_compute PROPERTIES CUDA_RUNTIME_LIBRARY Shared EXPORT_NAME native_compute)

# 安装目录移动后，仍可找到同目录的 MiniInfer 共享库。
set_target_properties(mini_infer mini_infer_compute mini_infer_memory PROPERTIES INSTALL_RPATH "$ORIGIN")
file(RELATIVE_PATH MINI_INFER_BIN_TO_LIB
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")

set(MINI_INFER_WITH_ARMADILLO OFF)
if(MINI_INFER_USE_ARMADILLO)
    find_package(Armadillo QUIET)
endif()
if(MINI_INFER_USE_ARMADILLO AND ARMADILLO_FOUND)
    set(MINI_INFER_WITH_ARMADILLO ON)
    # 为静态安装包保留可重定位的依赖 target，避免写入本机绝对库路径。
    add_library(mini_infer_armadillo INTERFACE IMPORTED)
    set_target_properties(mini_infer_armadillo PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ARMADILLO_LIBRARIES}")
    target_sources(mini_infer_compute PRIVATE
        src/ops/cpu/mini_infer_sgemv_armadillo.cpp
        src/ops/cpu/mini_infer_swiglu_armadillo.cpp)
    target_compile_definitions(mini_infer_compute PRIVATE MINI_INFER_HAS_ARMADILLO=1)
    target_link_libraries(mini_infer_compute PRIVATE mini_infer_armadillo)
    message(STATUS "CPU Sgemv/SwiGLU: Armadillo")
else()
    message(STATUS "CPU Sgemv/SwiGLU: scalar reference")
endif()

if(MINI_INFER_BUILD_APP)
    add_executable(mini_infer_main app/main.cpp)
    target_link_libraries(mini_infer_main PRIVATE MiniInfer::mini_infer)
    set_target_properties(mini_infer_main PROPERTIES
        INSTALL_RPATH "$ORIGIN/${MINI_INFER_BIN_TO_LIB}")
    install(TARGETS mini_infer_main RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


add_library(Infer::native ALIAS mini_infer)
add_library(Infer::native_compute ALIAS mini_infer_compute)
add_library(MiniInfer::native_compute ALIAS mini_infer_compute)


option(INFER_BUILD_MODEL_BENCHMARK "Build independent model storage/KV benchmark driver" OFF)
if(INFER_BUILD_MODEL_BENCHMARK)
    add_executable(miniinfer_model_bench app/benchmark_paged.cpp)
    target_link_libraries(miniinfer_model_bench PRIVATE MiniInfer::mini_infer)
endif()
