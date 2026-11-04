// 整个框架的阅读起点。本文件只负责装配和调用，算子实现在独立文件里。
// 从 Infer 根目录构建，CUDA 架构按实际设备设置：
// cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
// cmake --build build -j
// 运行：build/mini_infer_main --model MODEL.bin --tokenizer TOKENIZER.bin --device cuda --prompt "Once upon a time"
#include "mini_infer/app/mini_infer_cli.h"
#include "mini_infer/runtime/mini_infer_generation.h"
#include "mini_infer/base/mini_infer_memory_pool.h"

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    using namespace mini_infer;
    try {
        const auto options = parse_inference_options(argc, argv);
        if (options.help) { print_inference_help(std::cout); return 0; }
        InferenceConsole console;
        auto& output = console.output();

        // 1. 选择设备；CUDA 分配遵循 infer/config.h 的 INFER_MEMORY_MODE。
        std::shared_ptr<DeviceAllocator> allocator;
        if (options.device == "cuda") allocator = std::make_shared<CUDADeviceAllocator>();
        else allocator = std::make_shared<CPUDeviceAllocator>();

        // 2. mmap 读取完整模型；Tokenizer 与模型必须来自同一个词表。
        LlamaModel model(options.model_path, allocator, options.model_options);
        Tokenizer tokenizer(options.tokenizer_path, model.config().vocab_size);
        Sampler sampler(options.temperature, options.top_k, options.seed);
        const auto& weight_io = model.weight_io_stats();
        if (!weight_io.backend.empty())
            std::cerr << "weights_io=" << weight_io.backend
                      << " gds_p2p_verified=" << weight_io.p2p_verified
                      << " fallback=" << weight_io.fallback_reason << '\n';
        std::cerr << "MiniInfer | " << options.device << " | dim=" << model.config().dim
                  << " | layers=" << model.config().layer_num
                  << " | context=" << model.config().seq_len << '\n';

        // 3. 编码 -> prefill -> decode。回调只输出新片段，prompt 打印一次。
        output << options.prompt << std::flush;
        const auto start = std::chrono::steady_clock::now();
        const auto result = generate(model, tokenizer, sampler, options.prompt, options.steps,
            [&output](const std::string& piece) {
                // 保留 UTF-8 字节和常用空白，避免把单字节终端控制码打印出来。
                for (unsigned char byte : piece)
                    if ((byte >= 32 && byte != 127) || byte == '\n' || byte == '\r' || byte == '\t')
                        output.put(static_cast<char>(byte));
                output.flush();
            });
        output << '\n';
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        // 时间包括编码、prefill、decode、打印，不把它误标成纯 decode 吞吐。
        std::cerr << "prompt_tokens=" << result.prompt_tokens
                  << " generated_tokens=" << result.generated_tokens.size()
                  << " stop=" << result.stop_reason << " request_seconds=" << seconds << '\n';
        const auto kv = model.paged_stats();
        std::cerr << "kv_backend=" << (model.uses_paged_kv() ? options.storage_engine : "contiguous")
                  << " kv_ssd_read_bytes=" << kv.ssd_read_bytes
                  << " kv_ssd_written_bytes=" << kv.ssd_written_bytes
                  << " ssd_device_ios=" << kv.ssd_device.completed_ios << '\n';
        return 0; // RAII 依次释放模型 Tensor、KVCache、显存池。
    } catch (const std::exception& error) {
        std::cerr << "MiniInfer error: " << error.what() << '\n';
        return 1;
    }
}
