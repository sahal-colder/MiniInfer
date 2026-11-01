include_guard(GLOBAL)

# Build the same physical pools and logical storage core for C++ and C ABI callers.
# This helper adds no applications, tests, exports or package configuration.
function(infer_add_runtime root)
    if(TARGET infer_storage)
        return()
    endif()
    set(INFER_RUNTIME_ROOT "${root}")
    option(INFER_ENABLE_CUDA "Enable CUDA pools and GPU storage" ON)
    option(INFER_ENABLE_IO_URING "Enable optional liburing file I/O" ON)
    option(INFER_ENABLE_SPDK "Use an externally installed SPDK SDK" OFF)
    if(NOT DEFINED INFER_MEMORY_MODE)
        set(INFER_MEMORY_MODE 2)
    endif()
    set(INFER_CUDA_VALUE 0)
    if(INFER_ENABLE_CUDA)
        set(INFER_CUDA_VALUE 1)
    endif()
    foreach(backend NATIVE VLLM)
        if(NOT DEFINED INFER_${backend}_VALUE)
            set(INFER_${backend}_VALUE 0)
        endif()
    endforeach()
    configure_file("${INFER_RUNTIME_ROOT}/cmake/build_config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/infer/build_config.h" @ONLY)
    include("${INFER_RUNTIME_ROOT}/cmake/Dependencies.cmake")
    add_library(infer_memory STATIC
        ${INFER_RUNTIME_ROOT}/src/core/memory.cpp
        ${INFER_RUNTIME_ROOT}/src/cpu/cpu_block.cpp
        ${INFER_RUNTIME_ROOT}/src/cpu/cpu_memory_pool.cpp
        ${INFER_RUNTIME_ROOT}/src/cpu/cpu_thread_cache.cpp
        ${INFER_RUNTIME_ROOT}/src/cpu/pinned_memory_pool.cpp
    )
    if(INFER_ENABLE_CUDA)
        target_sources(infer_memory PRIVATE
            ${INFER_RUNTIME_ROOT}/src/cuda/cuda_block.cpp
            ${INFER_RUNTIME_ROOT}/src/cuda/cuda_graph_allocator.cpp
            ${INFER_RUNTIME_ROOT}/src/cuda/cuda_memory_pool.cpp
            ${INFER_RUNTIME_ROOT}/src/cuda/cuda_stream.cpp
            ${INFER_RUNTIME_ROOT}/src/utils/cuda_check.cpp)
        target_link_libraries(infer_memory PUBLIC CUDA::cudart)
    else()
        target_sources(infer_memory PRIVATE ${INFER_RUNTIME_ROOT}/src/cuda/cuda_unavailable.cpp)
    endif()
    add_library(MiniInfer::memory ALIAS infer_memory)
    add_library(Infer::memory ALIAS infer_memory)
    set_target_properties(infer_memory PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(infer_memory PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<BUILD_INTERFACE:${INFER_RUNTIME_ROOT}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(infer_memory PUBLIC cxx_std_17)
    target_compile_definitions(infer_memory PUBLIC
        INFER_ENABLE_CUDA=$<BOOL:${INFER_ENABLE_CUDA}>
        INFER_HAS_CUDA=$<BOOL:${INFER_ENABLE_CUDA}>
        INFER_MEMORY_MODE=${INFER_MEMORY_MODE})
    target_link_libraries(infer_memory PUBLIC Threads::Threads)
    
    add_library(infer_storage STATIC
        ${INFER_RUNTIME_ROOT}/src/storage/cpu_storage.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/file_io.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/kv_offload_cache.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/migration_executor.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/spdk_device.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/ssd_storage.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/storage_block.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/tiered_storage.cpp
        ${INFER_RUNTIME_ROOT}/src/storage/transfer.cpp
    )
    if(INFER_ENABLE_CUDA)
        target_sources(infer_storage PRIVATE ${INFER_RUNTIME_ROOT}/src/storage/gpu_storage.cpp)
    else()
        target_sources(infer_storage PRIVATE ${INFER_RUNTIME_ROOT}/src/storage/gpu_unavailable.cpp)
    endif()
    add_library(MiniInfer::storage ALIAS infer_storage)
    add_library(Infer::storage ALIAS infer_storage)
    set_target_properties(infer_storage PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(infer_storage PUBLIC infer_memory PRIVATE crc32c)
    if(TARGET infer_uring)
        target_compile_definitions(infer_storage PRIVATE INFER_HAS_IO_URING=1)
        target_link_libraries(infer_storage PRIVATE infer_uring)
    endif()
    if(INFER_ENABLE_SPDK)
        target_compile_definitions(infer_storage PRIVATE INFER_HAS_SPDK=1)
        target_link_libraries(infer_storage PRIVATE infer_spdk)
    endif()
    if(INFER_ENABLE_CUDA)
        set(CUDAToolkit_VERSION "${CUDAToolkit_VERSION}" PARENT_SCOPE)
        set(CUDAToolkit_LIBRARY_ROOT "${CUDAToolkit_LIBRARY_ROOT}" PARENT_SCOPE)
    endif()
endfunction()
