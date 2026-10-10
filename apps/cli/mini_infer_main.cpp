// 命令行应用：解析选项、装配推理对象、输出文本和运行统计。
// 构建和运行命令见项目 README.md。
#include "mini_infer_cli.h"
#include "mini_infer/generation/mini_infer_generation.h"
#include "mini_infer/core/mini_infer_memory.h"

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    using namespace mini_infer;
    try {
        const auto options = parse_inference_options(argc, argv);
        if (options.help) { print_inference_help(std::cout); return 0; }
        auto& output = std::cout;

        // 1. 选择内存设备；CUDA 路径使用缓存分配器。
        std::shared_ptr<DeviceAllocator> allocator;
        if (options.device == "cuda") allocator = std::make_shared<CachingCUDADeviceAllocator>();
        else allocator = std::make_shared<CPUDeviceAllocator>();

        // 2. mmap 读取完整模型；Tokenizer 与模型必须来自同一个词表。
        LlamaModel model(options.model_path, allocator);
        Tokenizer tokenizer(options.tokenizer_path, model.config().vocab_size);
        Sampler sampler(options.temperature, options.top_k, options.seed);
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
        return 0; // RAII 依次释放模型 Tensor、KVCache、显存池。
    } catch (const std::exception& error) {
        std::cerr << "MiniInfer error: " << error.what() << '\n';
        return 1;
    }
}
