find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(cuda_graph_test cuda_graph_pool_test.cu)
target_link_libraries(cuda_graph_test PRIVATE infer_memory GTest::gtest_main)
if(INFER_BUILD_NATIVE)
    target_sources(cuda_graph_test PRIVATE cuda_graph_model_test.cpp)
    target_link_libraries(cuda_graph_test PRIVATE mini_infer_compute)
endif()
gtest_discover_tests(cuda_graph_test PROPERTIES TIMEOUT 60 LABELS cuda-graph)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(cuda_allocator_contract_test cuda_allocator_contract_test.cpp)
    target_link_libraries(cuda_allocator_contract_test PRIVATE infer_memory GTest::gtest_main)
    target_link_options(cuda_allocator_contract_test PRIVATE
        -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaFree -Wl,--wrap=cudaEventRecord
        -Wl,--wrap=cudaEventCreateWithFlags -Wl,--wrap=cudaEventQuery
        -Wl,--wrap=cudaDeviceSynchronize)
    gtest_discover_tests(cuda_allocator_contract_test PROPERTIES TIMEOUT 30 LABELS allocator)
endif()
