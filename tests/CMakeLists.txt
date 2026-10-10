option(MINI_INFER_TEST_CUDA "Run tests that require a CUDA device" ON)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
enable_language(C)

add_library(mini_infer_test_support STATIC support/test_support.cpp support/kv_test_support.cpp)
target_include_directories(mini_infer_test_support PUBLIC support)
target_link_libraries(mini_infer_test_support PUBLIC mini_infer)

function(mini_infer_test_target name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mini_infer_test_support)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

function(mini_infer_cpu_test name)
    add_test(NAME ${name} COMMAND ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS cpu TIMEOUT 120)
endfunction()

function(mini_infer_cuda_test name)
    if(MINI_INFER_TEST_CUDA)
        add_test(NAME ${name} COMMAND ${ARGN})
        set_tests_properties(${name} PROPERTIES LABELS cuda RUN_SERIAL TRUE TIMEOUT 180)
    endif()
endfunction()

mini_infer_test_target(test_core_errors core/test_errors.cpp)
foreach(name IN ITEMS null_copy null_buffer null_borrow null_zero bad_copy_kind
        bad_pool_config foreign_pool_pointer cuda_error cpu_sync no_stdout destructor tensor_move)
    mini_infer_cpu_test(core.${name} test_core_errors ${name})
endforeach()
set_tests_properties(core.cpu_sync PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
mini_infer_test_target(test_registry backend/test_registry.cpp)
mini_infer_cpu_test(backend.registry test_registry)
add_executable(test_pool_failures core/test_pool_failures.cpp
    ${PROJECT_SOURCE_DIR}/src/core/mini_infer_memory_pool.cpp)
target_include_directories(test_pool_failures PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(test_pool_failures PRIVATE cxx_std_17)
target_link_libraries(test_pool_failures PRIVATE CUDA::cudart Threads::Threads)
mini_infer_cpu_test(core.pool_metadata test_pool_failures metadata)
mini_infer_cpu_test(core.pool_trim test_pool_failures trim)

foreach(name IN ITEMS sgemv sgemv_checks quantized_sgemv quantized_layer)
    mini_infer_test_target(test_${name} ops/matmul/test_${name}.cpp)
    mini_infer_cpu_test(ops.${name}.cpu test_${name} --cpu-only)
endforeach()
mini_infer_cuda_test(ops.sgemv.cuda test_sgemv --cuda)
mini_infer_cuda_test(ops.quantized_sgemv.cuda test_quantized_sgemv --cuda)

foreach(name IN ITEMS kv_cache kv_cache_checks kv_cache_lifetime cached_attention)
    mini_infer_test_target(test_${name} cache/test_${name}.cpp)
    mini_infer_cpu_test(cache.${name}.cpu test_${name} --cpu-only)
    mini_infer_cuda_test(cache.${name}.cuda test_${name} --cuda)
endforeach()

mini_infer_test_target(test_qkv ops/attention/test_qkv.cpp)
mini_infer_cpu_test(ops.qkv.cpu test_qkv)
mini_infer_cuda_test(ops.qkv.cuda test_qkv --cuda)
mini_infer_test_target(test_mlp ops/mlp/test_mlp.cpp)
mini_infer_cpu_test(ops.mlp.cpu test_mlp)
mini_infer_cuda_test(ops.mlp.cuda test_mlp --cuda)
mini_infer_cuda_test(ops.mlp.cuda_large test_mlp --cuda --large)
mini_infer_test_target(test_attention ops/attention/test_attention.cpp)
mini_infer_test_target(test_elementwise ops/test_elementwise.cpp)
foreach(name IN ITEMS attention elementwise)
    mini_infer_cpu_test(ops.${name}.cpu test_${name})
    mini_infer_cuda_test(ops.${name}.cuda test_${name} --cuda)
    mini_infer_cuda_test(ops.${name}.cuda_default test_${name} --cuda --default-stream)
    mini_infer_cuda_test(ops.${name}.cuda_large test_${name} --cuda --large)
endforeach()

set(MINI_INFER_FIXTURES ${PROJECT_SOURCE_DIR}/data/fixtures/framework)
mini_infer_test_target(test_framework model/test_framework.cpp)
mini_infer_cpu_test(model.framework.cpu test_framework ${MINI_INFER_FIXTURES})
mini_infer_cuda_test(model.framework.cuda test_framework ${MINI_INFER_FIXTURES} --cuda)
mini_infer_cpu_test(cli.cpu ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cli/test_cli.py
    $<TARGET_FILE:mini_infer_main> ${MINI_INFER_FIXTURES})
mini_infer_cuda_test(cli.cuda ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cli/test_cli.py
    $<TARGET_FILE:mini_infer_main> ${MINI_INFER_FIXTURES} --cuda)

add_library(llama2c_reference STATIC reference/reference_bridge.c)
target_link_libraries(llama2c_reference PRIVATE m)
mini_infer_test_target(test_pretrained model/test_pretrained.cpp)
target_link_libraries(test_pretrained PRIVATE llama2c_reference)
mini_infer_cpu_test(model.pretrained.cpu test_pretrained
    ${PROJECT_SOURCE_DIR}/data/models/stories260K.bin
    ${PROJECT_SOURCE_DIR}/data/tokenizers/tok512.bin)
mini_infer_cuda_test(model.pretrained.cuda test_pretrained
    ${PROJECT_SOURCE_DIR}/data/models/stories260K.bin
    ${PROJECT_SOURCE_DIR}/data/tokenizers/tok512.bin --cuda)

get_filename_component(MINI_INFER_CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)
add_test(NAME package.install COMMAND ${CMAKE_COMMAND}
    "-DMINI_INFER_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DMINI_INFER_BINARY_DIR=${PROJECT_BINARY_DIR}"
    "-DMINI_INFER_CONFIG=$<CONFIG>"
    "-DMINI_INFER_GENERATOR=${CMAKE_GENERATOR}"
    "-DMINI_INFER_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DMINI_INFER_CUDA_ROOT=${MINI_INFER_CUDA_ROOT}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/package/test_install.cmake)
set_tests_properties(package.install PROPERTIES LABELS packaging TIMEOUT 180)
