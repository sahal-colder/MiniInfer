cmake_minimum_required(VERSION 3.18)
project(MiniInfer VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
option(MINI_INFER_USE_ARMADILLO "Use Armadillo for CPU GEMV and SwiGLU" ON)

# Module include directories also preserve the existing bare header names.
set(MINI_INFER_HEADER_DIRS
    mini_infer/backend
    mini_infer/cache
    mini_infer/core
    mini_infer/generation
    mini_infer/model
    mini_infer/ops/add
    mini_infer/ops/attention
    mini_infer/ops/base
    mini_infer/ops/embedding
    mini_infer/ops/matmul
    mini_infer/ops/mlp
    mini_infer/ops/rope
    mini_infer/ops/softmax
    mini_infer/weights
)
add_library(mini_infer_memory
    src/core/mini_infer_memory.cpp
    src/core/mini_infer_memory_pool.cpp
)
add_library(MiniInfer::memory ALIAS mini_infer_memory)
set_target_properties(mini_infer_memory PROPERTIES
    EXPORT_NAME memory POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(mini_infer_memory PUBLIC cxx_std_17)
target_include_directories(mini_infer_memory PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
foreach(directory IN LISTS MINI_INFER_HEADER_DIRS)
    target_include_directories(mini_infer_memory PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${directory}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${directory}>)
endforeach()
target_link_libraries(mini_infer_memory PUBLIC CUDA::cudart Threads::Threads)

add_library(mini_infer
    src/backend/mini_infer_backend.cpp
    src/backend/mini_infer_kernels_cpu.cpp
    src/backend/mini_infer_kernels_cuda.cu
    src/cache/mini_infer_kv_cache.cpp
    src/cache/mini_infer_kv_cache.cu
    src/core/mini_infer_tensor.cpp
    src/generation/mini_infer_generation.cpp
    src/generation/mini_infer_sampler.cpp
    src/generation/mini_infer_tokenizer.cpp
    src/model/mini_infer_model.cpp
    src/model/mini_infer_model_weights.cpp
    src/model/mini_infer_transformer.cpp
    src/ops/add/mini_infer_add.cpp
    src/ops/add/mini_infer_add.cu
    src/ops/attention/mini_infer_attention.cpp
    src/ops/attention/mini_infer_attention.cu
    src/ops/attention/mini_infer_attention_qkv.cpp
    src/ops/base/mini_infer_layer.cpp
    src/ops/embedding/mini_infer_embedding.cpp
    src/ops/embedding/mini_infer_embedding.cu
    src/ops/matmul/mini_infer_matmul.cpp
    src/ops/matmul/mini_infer_quantized_sgemv.cpp
    src/ops/matmul/mini_infer_quantized_sgemv.cu
    src/ops/matmul/mini_infer_sgemv.cpp
    src/ops/matmul/mini_infer_sgemv.cu
    src/ops/mlp/mini_infer_mlp.cpp
    src/ops/mlp/mini_infer_swiglu.cpp
    src/ops/mlp/mini_infer_swiglu.cu
    src/ops/rope/mini_infer_rope.cpp
    src/ops/rope/mini_infer_rope.cu
    src/ops/softmax/mini_infer_softmax.cpp
    src/ops/softmax/mini_infer_softmax.cu
    src/weights/mini_infer_mlp_weight_loader.cpp
    src/weights/mini_infer_mmap.cpp
    src/weights/mini_infer_qkv_weight_loader.cpp
    src/weights/mini_infer_quantization.cpp
    src/weights/mini_infer_weight_loader.cpp
)
add_library(MiniInfer::mini_infer ALIAS mini_infer)
set_target_properties(mini_infer PROPERTIES
    EXPORT_NAME mini_infer POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(mini_infer PUBLIC mini_infer_memory)

set(MINI_INFER_WITH_ARMADILLO OFF)
if(MINI_INFER_USE_ARMADILLO)
    find_package(Armadillo QUIET)
endif()
if(MINI_INFER_USE_ARMADILLO AND ARMADILLO_FOUND)
    set(MINI_INFER_WITH_ARMADILLO ON)
    if(NOT TARGET Armadillo::Armadillo)
        add_library(Armadillo::Armadillo INTERFACE IMPORTED)
        set_target_properties(Armadillo::Armadillo PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${ARMADILLO_LIBRARIES}")
    endif()
    target_compile_definitions(mini_infer PRIVATE MINI_INFER_HAS_ARMADILLO=1)
    target_link_libraries(mini_infer PRIVATE Armadillo::Armadillo)
    message(STATUS "CPU math backend: Armadillo")
else()
    message(STATUS "CPU math backend: scalar")
endif()

add_library(mini_infer_cli_support STATIC apps/cli/mini_infer_cli.cpp)
target_include_directories(mini_infer_cli_support PUBLIC apps/cli)
target_compile_features(mini_infer_cli_support PUBLIC cxx_std_17)
add_executable(mini_infer_main apps/cli/mini_infer_main.cpp)
target_link_libraries(mini_infer_main PRIVATE mini_infer mini_infer_cli_support)

# Resolve our shared libraries relative to the installed binary/package after
# relocation. External dependencies (CUDA/BLAS) remain system dependencies.
file(RELATIVE_PATH MINI_INFER_BIN_TO_LIB "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
set_target_properties(mini_infer mini_infer_memory PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    INSTALL_RPATH "$ORIGIN")
set_target_properties(mini_infer_main PROPERTIES INSTALL_RPATH "$ORIGIN/${MINI_INFER_BIN_TO_LIB}")

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(TARGETS mini_infer mini_infer_memory EXPORT MiniInferTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS mini_infer_main RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT MiniInferTargets NAMESPACE MiniInfer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniInfer)
configure_package_config_file(cmake/MiniInferConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniInfer)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniInfer)
