find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(storage_io_test storage_io_test.cpp)
target_link_libraries(storage_io_test PRIVATE infer_storage GTest::gtest_main)
target_compile_definitions(storage_io_test PRIVATE
    INFER_SPDK_MALLOC_JSON="${CMAKE_CURRENT_SOURCE_DIR}/spdk_malloc.json")
if(TARGET infer_device_io)
    target_link_libraries(storage_io_test PRIVATE infer_device_io)
    target_compile_definitions(storage_io_test PRIVATE TEST_C_DEVICE=1)
endif()
if(TARGET infer_uring)
    target_compile_definitions(storage_io_test PRIVATE TEST_DIRECT_FILE=1)
endif()
gtest_discover_tests(storage_io_test)
add_executable(cpu_allocator_contract_test cpu_allocator_contract_test.cpp)
target_link_libraries(cpu_allocator_contract_test PRIVATE infer_memory GTest::gtest_main)
gtest_discover_tests(cpu_allocator_contract_test PROPERTIES TIMEOUT 30 LABELS allocator)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(offload_completion_test offload_completion_test.cpp)
    target_link_libraries(offload_completion_test PRIVATE infer_storage GTest::gtest_main)
    target_link_options(offload_completion_test PRIVATE -Wl,--wrap=pwrite -Wl,--wrap=pwritev)
    if(INFER_ENABLE_CUDA)
        target_link_options(offload_completion_test PRIVATE
            -Wl,--wrap=cudaEventSynchronize -Wl,--wrap=cudaStreamSynchronize)
    endif()
    gtest_discover_tests(offload_completion_test PROPERTIES TIMEOUT 30)
    if(INFER_ENABLE_SPDK)
        add_executable(spdk_lifecycle_test spdk_lifecycle_test.cpp)
        target_link_libraries(spdk_lifecycle_test PRIVATE infer_storage infer_spdk GTest::gtest_main)
        target_compile_definitions(spdk_lifecycle_test PRIVATE
            INFER_SPDK_MALLOC_JSON="${CMAKE_CURRENT_SOURCE_DIR}/spdk_malloc.json"
            INFER_SPDK_LOWIO_JSON="${CMAKE_CURRENT_SOURCE_DIR}/spdk_lowio.json"
            INFER_SPDK_ZEROIO_JSON="${CMAKE_CURRENT_SOURCE_DIR}/spdk_zeroio.json")
        target_link_options(spdk_lifecycle_test PRIVATE
            -Wl,--wrap=spdk_bdev_write -Wl,--wrap=spdk_bdev_read
            -Wl,--wrap=spdk_bdev_queue_io_wait -Wl,--wrap=spdk_dma_free -Wl,--wrap=spdk_app_stop)
        gtest_discover_tests(spdk_lifecycle_test PROPERTIES TIMEOUT 30)
    endif()
    if(INFER_BUILD_VLLM AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
        add_executable(native_transfer_test native_transfer_test.cpp)
        target_link_libraries(native_transfer_test PRIVATE infer_storage CUDA::cuda_driver GTest::gtest_main)
        target_link_options(native_transfer_test PRIVATE
            -Wl,--wrap=cuMemcpyAsync -Wl,--wrap=cuEventRecord -Wl,--wrap=cuStreamSynchronize
            -Wl,--wrap=cuEventDestroy_v2 -Wl,--wrap=pread -Wl,--wrap=preadv)
        gtest_discover_tests(native_transfer_test PROPERTIES TIMEOUT 30)
    endif()
endif()
