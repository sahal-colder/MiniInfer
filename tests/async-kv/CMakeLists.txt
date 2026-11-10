find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(async_kv_pipeline_test async_kv_pipeline_test.cu)
target_link_libraries(async_kv_pipeline_test PRIVATE infer_storage GTest::gtest_main)
if(INFER_BUILD_NATIVE)
    target_sources(async_kv_pipeline_test PRIVATE async_kv_model_test.cpp)
    target_link_libraries(async_kv_pipeline_test PRIVATE mini_infer_compute)
endif()
gtest_discover_tests(async_kv_pipeline_test PROPERTIES TIMEOUT 30)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(async_kv_failure_test async_kv_failure_test.cpp)
    target_link_libraries(async_kv_failure_test PRIVATE infer_storage GTest::gtest_main)
    target_link_options(async_kv_failure_test PRIVATE -Wl,--wrap=cudaEventCreateWithFlags
        -Wl,--wrap=cudaEventRecord -Wl,--wrap=cudaStreamWaitEvent -Wl,--wrap=cudaMemcpy2DAsync)
    gtest_discover_tests(async_kv_failure_test PROPERTIES TIMEOUT 30)
endif()
