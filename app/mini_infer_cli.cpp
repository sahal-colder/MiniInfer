#include "mini_infer/app/mini_infer_cli.h"

#include <charconv>
#include <cmath>
#include <iostream>
#include <ostream>
#include <stdexcept>

namespace mini_infer {
InferenceConsole::InferenceConsole()
    : output_(std::cout.rdbuf()) {}
InferenceConsole::~InferenceConsole() = default;

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
    auto& paged = options.model_options.paged;
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
        else if (flag == "--storage-engine") options.storage_engine = value;
        else if (flag == "--gpu-cache-bytes") paged.gpu_bytes = integer<std::size_t>(value);
        else if (flag == "--cpu-cache-bytes") paged.host_bytes = integer<std::size_t>(value);
        else if (flag == "--ssd-bytes") paged.ssd_bytes = integer<std::size_t>(value);
        else if (flag == "--ssd-directory") paged.ssd_directory = value;
        else if (flag == "--io-depth") paged.ssd_io.queue_depth = integer<std::size_t>(value);
        else if (flag == "--file-shards") paged.ssd_io.file_shards = integer<std::size_t>(value);
        else if (flag == "--spdk-config") paged.ssd_spdk.json_config = value;
        else if (flag == "--spdk-bdev") paged.ssd_spdk.bdev_name = value;
        else if (flag == "--spdk-reactor-mask") paged.ssd_spdk.reactor_mask = value;
        else if (flag == "--spdk-iova-mode") paged.ssd_spdk.iova_mode = value;
        else if (flag == "--spdk-mem-mib") paged.ssd_spdk.environment_mib = integer<int>(value);
        else if (flag == "--spdk-chunk-kib") {
            const auto kib = integer<std::size_t>(value);
            if (!kib || kib > 8192 || kib % 4) throw std::invalid_argument("SPDK chunk must be 4..8192 KiB, aligned to 4 KiB");
            paged.ssd_spdk.io_buffer_bytes = kib * 1024;
        } else if (flag == "--spdk-no-huge" || flag == "--spdk-no-pci") {
            const auto enabled = integer<unsigned>(value);
            if (enabled > 1) throw std::invalid_argument(flag + " must be 0 or 1");
            (flag == "--spdk-no-huge" ? paged.ssd_spdk.no_huge : paged.ssd_spdk.no_pci) = enabled;
        } else if (flag == "--ssd-verify") {
            if (value != "on_read" && value != "read_back") throw std::invalid_argument("Invalid SSD verification");
            paged.ssd_read_back = value == "read_back";
        }
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
    if (!paged.ssd_io.queue_depth || paged.ssd_io.queue_depth > 64 ||
        (paged.ssd_io.file_shards != 1 && paged.ssd_io.file_shards != 4))
        throw std::invalid_argument("Invalid SSD I/O depth or shard count");
    if (options.storage_engine == "spdk") {
        auto& spdk = paged.ssd_spdk;
        if (options.device != "cuda" || spdk.json_config.empty() || spdk.bdev_name.empty() ||
            !paged.ssd_bytes || paged.ssd_bytes % 4096 || spdk.environment_mib < 64 ||
            spdk.environment_mib > 4096 ||
            (spdk.iova_mode != "auto" && spdk.iova_mode != "va" && spdk.iova_mode != "pa"))
            throw std::invalid_argument("Native SPDK needs CUDA, JSON, bdev, aligned SSD budget and valid environment options");
        spdk.capacity_bytes = paged.ssd_bytes;
        spdk.queue_depth = paged.ssd_io.queue_depth;
        spdk.dma_budget_bytes = std::size_t(spdk.environment_mib) * 1024 * 1024;
        paged.ssd_direct = false;
    } else {
        if (!paged.ssd_spdk.bdev_name.empty() || !paged.ssd_spdk.json_config.empty())
            throw std::invalid_argument("SPDK options require --storage-engine spdk");
        if (options.storage_engine == "uring") paged.ssd_io.mode = infer::FileIOMode::IO_URING;
        else if (options.storage_engine == "posix") paged.ssd_io.mode = infer::FileIOMode::POSIX;
        else if (options.storage_engine == "auto") paged.ssd_io.mode = infer::FileIOMode::AUTO;
        else throw std::invalid_argument("Invalid storage engine");
        paged.ssd_direct = options.storage_engine == "uring";
    }
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
        << "  --storage-engine posix|uring|auto|spdk  KV storage; default: uring\n"
        << "  --gpu-cache-bytes N --cpu-cache-bytes N --ssd-bytes N  KV budgets\n"
        << "  --spdk-config JSON --spdk-bdev NAME    dedicated NVMe configuration\n"
        << "  Use app/infer.py and infer.conf for all storage options.\n"
        << "  --help               show this help\n\n"
        << "Format: llama2.c legacy FP32 (export version 0, 28-byte header)\n"
        << "and its matching tokenizer.bin. RoPE theta=10000, RMSNorm eps=1e-5.\n"
        << "GGUF, tokenizer.model and quantized checkpoints are not supported. Named MiniInfer FP32 safetensors require CUDA.\n"
        << "CPU execution still uses this project's CUDA-toolkit build.\n";
}
}  // namespace mini_infer
