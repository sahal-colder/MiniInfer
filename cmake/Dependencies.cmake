find_package(Threads REQUIRED)
if(INFER_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT INFER_MEMORY_MODE STREQUAL "0" AND CUDAToolkit_VERSION VERSION_LESS "11.2")
        message(FATAL_ERROR "ASYNC and POOL modes require CUDA Toolkit 11.2 or newer")
    endif()
endif()

set(CRC32C_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CRC32C_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CRC32C_USE_GLOG OFF CACHE BOOL "" FORCE)
set(CRC32C_INSTALL ON CACHE BOOL "" FORCE)
add_subdirectory("${INFER_RUNTIME_ROOT}/third_party/crc32c" "${CMAKE_CURRENT_BINARY_DIR}/crc32c")
set_target_properties(crc32c crc32c_arm64 crc32c_sse42 PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(INFER_ENABLE_IO_URING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(INFER_URING_INCLUDE_DIR liburing.h)
    find_library(INFER_URING_LIBRARY uring)
    if(INFER_URING_INCLUDE_DIR AND INFER_URING_LIBRARY)
        add_library(infer_uring INTERFACE)
        target_include_directories(infer_uring INTERFACE ${INFER_URING_INCLUDE_DIR})
        target_link_libraries(infer_uring INTERFACE ${INFER_URING_LIBRARY})
    endif()
endif()

# SPDK is supplied by the caller's SDK. No fetch, source copy or bundled build.
if(INFER_ENABLE_SPDK)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "SPDK requires Linux")
    endif()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(INFER_SPDK REQUIRED
        spdk_event spdk_event_bdev spdk_event_iobuf spdk_bdev_nvme
        spdk_bdev_malloc spdk_bdev_uring spdk_env_dpdk spdk_syslibs liburing>=2.3)
    add_library(infer_spdk INTERFACE)
    target_include_directories(infer_spdk INTERFACE ${INFER_SPDK_INCLUDE_DIRS})
    target_link_directories(infer_spdk INTERFACE ${INFER_SPDK_LIBRARY_DIRS})
    target_compile_options(infer_spdk INTERFACE ${INFER_SPDK_CFLAGS_OTHER})
    target_link_options(infer_spdk INTERFACE -Wl,--no-undefined)
    # Preserve constructor-based bdev registration and cyclic static dependencies.
    target_link_libraries(infer_spdk INTERFACE
        -Wl,--no-as-needed -Wl,--start-group -Wl,--whole-archive
        ${INFER_SPDK_STATIC_LDFLAGS}
        -Wl,--no-whole-archive -Wl,--end-group -Wl,--as-needed)
endif()
