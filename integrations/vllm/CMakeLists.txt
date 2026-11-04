find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent/'include')"
    OUTPUT_VARIABLE INFER_TORCH_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE INFER_TORCH_RESULT)
if(NOT INFER_TORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "INFER_BUILD_VLLM requires PyTorch in Python3_EXECUTABLE")
endif()
add_library(miniinfer_vllm_bridge SHARED storage_bridge.cpp)
add_library(Infer::vllm_storage ALIAS miniinfer_vllm_bridge)
target_link_libraries(miniinfer_vllm_bridge PRIVATE infer_storage CUDA::cuda_driver)
add_library(miniinfer_pool_bridge SHARED pool_bridge.cpp)
add_library(Infer::vllm_pool ALIAS miniinfer_pool_bridge)
target_include_directories(miniinfer_pool_bridge PRIVATE ${INFER_TORCH_INCLUDE})
target_link_libraries(miniinfer_pool_bridge PRIVATE infer_storage)
set_target_properties(miniinfer_vllm_bridge miniinfer_pool_bridge PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
install(TARGETS miniinfer_vllm_bridge miniinfer_pool_bridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/infer/vllm
    FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
install(FILES NOTICE VLLM-LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/infer/vllm)
