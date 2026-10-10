# Install into a disposable build subdirectory, move it, and compile a consumer
# that knows only the exported package. This catches source-tree path leakage.
set(test_dir "${MINI_INFER_BINARY_DIR}/package-test")
file(REMOVE_RECURSE "${test_dir}")

function(run_checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Command failed (${result}): ${ARGV}\n${output}\n${error}")
    endif()
endfunction()

run_checked("${CMAKE_COMMAND}" --install "${MINI_INFER_BINARY_DIR}"
    --config "${MINI_INFER_CONFIG}" --prefix "${test_dir}/stage")
file(RENAME "${test_dir}/stage" "${test_dir}/relocated")
run_checked("${CMAKE_COMMAND}" -S "${MINI_INFER_SOURCE_DIR}/tests/package"
    -B "${test_dir}/consumer" -G "${MINI_INFER_GENERATOR}"
    "-DCMAKE_BUILD_TYPE=${MINI_INFER_CONFIG}"
    "-DCMAKE_CXX_COMPILER=${MINI_INFER_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${test_dir}/relocated"
    "-DCUDAToolkit_ROOT=${MINI_INFER_CUDA_ROOT}"
    "-DMINI_INFER_TEST_MODEL=${MINI_INFER_SOURCE_DIR}/data/models/stories260K.bin")
run_checked("${CMAKE_COMMAND}" --build "${test_dir}/consumer" --config "${MINI_INFER_CONFIG}")
run_checked("${CMAKE_CTEST_COMMAND}" --test-dir "${test_dir}/consumer"
    --build-config "${MINI_INFER_CONFIG}" --output-on-failure)
run_checked("${test_dir}/relocated/bin/mini_infer_main" --help)
run_checked("${test_dir}/relocated/bin/mini_infer_main"
    --model "${MINI_INFER_SOURCE_DIR}/data/models/stories260K.bin"
    --tokenizer "${MINI_INFER_SOURCE_DIR}/data/tokenizers/tok512.bin"
    --prompt "Once upon a time" --steps 2 --device cpu)
message(STATUS "PASS install, relocation, C++ consumer and installed CLI")
