#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace mini_infer {
// 参数解析独立出来，让 main 只展示推理主线。
struct InferenceOptions {
    std::string model_path;
    std::string tokenizer_path;
    std::string prompt = "Once upon a time";
    std::string device = "cpu";
    int32_t steps = 64;
    float temperature = 0.0f;
    int32_t top_k = 0;
    uint32_t seed = 42;
    bool help = false;
};
InferenceOptions parse_inference_options(int argc, char** argv);
void print_inference_help(std::ostream& output);

}  // namespace mini_infer
