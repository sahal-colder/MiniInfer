#include "mini_infer_cli.h"

#include <charconv>
#include <cmath>
#include <iostream>
#include <ostream>
#include <stdexcept>

namespace mini_infer {
namespace {
template<class T> T integer(const std::string& value) {
    T result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("invalid integer: " + value);
    return result;
}
}
InferenceOptions parse_inference_options(int argc, char** argv) {
    InferenceOptions options;
    if (argc == 1) options.help = true;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { options.help = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--model") options.model_path = value;
        else if (flag == "--tokenizer") options.tokenizer_path = value;
        else if (flag == "--prompt") options.prompt = value;
        else if (flag == "--device") options.device = value;
        else if (flag == "--steps") options.steps = integer<int32_t>(value);
        else if (flag == "--top-k") options.top_k = integer<int32_t>(value);
        else if (flag == "--seed") options.seed = integer<uint32_t>(value);
        else if (flag == "--temperature") {
            std::size_t used = 0;
            options.temperature = std::stof(value, &used);
            if (used != value.size()) throw std::invalid_argument("invalid temperature: " + value);
        } else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (options.help) return options;
    if (options.model_path.empty() || options.tokenizer_path.empty())
        throw std::invalid_argument("both --model and --tokenizer are required; see --help");
    if (options.device != "cpu" && options.device != "cuda")
        throw std::invalid_argument("--device must be cpu or cuda");
    if (options.steps < 0 || options.top_k < 0 || !std::isfinite(options.temperature) ||
        options.temperature < 0)
        throw std::invalid_argument("steps, top-k and finite temperature must be nonnegative");
    return options;
}

void print_inference_help(std::ostream& out) {
    out << "MiniInfer: single-sequence FP32 Llama inference\n"
        << "Usage: mini_infer_main --model MODEL.bin --tokenizer TOKENIZER.bin [options]\n"
        << "  --prompt TEXT        default: Once upon a time\n"
        << "  --device cpu|cuda    default: cpu\n"
        << "  --steps N            maximum NEW tokens, default: 64\n"
        << "  --temperature T      0 = greedy, >0 = sampling, default: 0\n"
        << "  --top-k K            0 = full vocabulary, default: 0\n"
        << "  --seed N             random seed, default: 42\n"
        << "  --help               show this help\n\n"
        << "Format: llama2.c legacy FP32 (export version 0, 28-byte header)\n"
        << "and its matching tokenizer.bin. RoPE theta=10000, RMSNorm eps=1e-5.\n"
        << "GGUF, safetensors, tokenizer.model and quantized checkpoints are not supported.\n"
        << "CPU execution still uses this project's CUDA-toolkit build.\n";
}
}  // namespace mini_infer
