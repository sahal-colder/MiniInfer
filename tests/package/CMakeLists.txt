cmake_minimum_required(VERSION 3.18)
project(MiniInferConsumer LANGUAGES CXX)

find_package(MiniInfer 0.1 CONFIG REQUIRED)
add_executable(package_smoke smoke.cpp)
target_link_libraries(package_smoke PRIVATE MiniInfer::mini_infer)
enable_testing()
add_test(NAME installed.consumer COMMAND package_smoke "${MINI_INFER_TEST_MODEL}")
