#include "bench_common.h"
#include "mini_infer/model/mini_infer_model.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
int32_t argmax(const mini_infer::Tensor& logits) {
    return std::max_element(logits.ptr<float>(), logits.ptr<float>() + logits.size()) - logits.ptr<float>();
}
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}

int main(int argc, char** argv) {
    try {
        const auto args = bench::arguments(argc, argv);
        const auto variant = args.at("variant");
        const auto workload = args.at("workload");
        const auto generated = bench::number(args, "generate", 128);
        if (variant != "sync" && variant != "async" && variant != "graph")
            throw std::invalid_argument("variant must be sync/async/graph");
        if (workload != "A" && workload != "B-host" && workload != "B-nvme")
            throw std::invalid_argument("workload must be A/B-host/B-nvme");
        if (generated < 2) throw std::invalid_argument("At least two output tokens required for TPOT");
        std::ifstream prompt_file(args.at("prompt-file"));
        std::vector<int32_t> prompt;
        int32_t token = 0;
        while (prompt_file >> token) prompt.push_back(token);
        if (prompt.empty() || !prompt_file.eof()) throw std::invalid_argument("Invalid token ID file");
        check(cudaSetDevice(0));
        check(cudaFree(nullptr));
        // An explicit fixed ballast is reproducible across engines; never fill the desktop GPU implicitly.
        void* ballast = nullptr;
        const auto ballast_bytes = bench::number(args, "ballast-mib", 0) * 1024 * 1024;
        if (ballast_bytes) { check(cudaMalloc(&ballast, ballast_bytes)); check(cudaMemset(ballast, 0, ballast_bytes)); }
        auto allocator = std::make_shared<mini_infer::CUDADeviceAllocator>();
        mini_infer::ModelOptions options;
        options.kv_mode = mini_infer::ModelKVMode::PAGED;
        options.paged.tokens_per_page = bench::number(args, "page-tokens", 16);
        options.paged.resident_layers = workload == "A" ? 0 : 2;
        options.paged.gpu_bytes = bench::number(args, "gpu-bytes", 0);
        options.paged.host_bytes = bench::number(args, "host-bytes", 0);
        options.paged.ssd_directory = args.at("directory");
        options.paged.enable_async_kv_pipeline = variant != "sync";
        options.paged.prefetch = variant != "sync";
        options.enable_cuda_graph = variant == "graph";
        const auto io = args.at("io");
        if (io != "uring" && io != "posix" && io != "spdk") throw std::invalid_argument("Invalid I/O backend");
#ifndef BENCH_HAS_URING
        if (io == "uring") { std::cerr << "liburing not compiled\n"; return 77; }
#endif
        options.paged.ssd_direct = io == "uring";
        options.paged.ssd_io.mode = io == "uring" ? infer::FileIOMode::IO_URING : infer::FileIOMode::POSIX;
        if (io == "spdk") {
            if (!infer::spdk_available()) { std::cerr << "SPDK not compiled\n"; return 77; }
            if (args.at("allow-spdk-write") != "YES") throw std::invalid_argument("Explicit SPDK write consent required");
            auto& spdk = options.paged.ssd_spdk;
            spdk.json_config = args.at("spdk-config");
            spdk.bdev_name = args.at("spdk-bdev");
            spdk.offset_bytes = bench::number(args, "spdk-offset", UINT64_MAX);
            spdk.capacity_bytes = bench::number(args, "spdk-capacity", 0);
            spdk.no_pci = bench::number(args, "spdk-no-pci", 0);
            spdk.no_huge = bench::number(args, "spdk-no-huge", 0);
            spdk.environment_mib = bench::number(args, "spdk-memory-mib", 512);
            options.paged.ssd_bytes = spdk.capacity_bytes;
            if (spdk.offset_bytes == UINT64_MAX || !spdk.capacity_bytes)
                throw std::invalid_argument("Explicit SPDK offset/capacity required");
        }
        const auto init_start = bench::Clock::now();
        mini_infer::LlamaModel model(args.at("model"), allocator, options);
        check(cudaDeviceSynchronize());
        const auto init_ms = bench::seconds(init_start) * 1000;
        if (prompt.size() + generated > std::size_t(model.config().seq_len))
            throw std::invalid_argument("Prompt + output exceed model context");
        auto warm_prompt = prompt;
        warm_prompt[0] = (warm_prompt[0] + 1) % model.config().vocab_size;
        for (std::size_t i = 0; i < bench::number(args, "warmup", 1); ++i) {
            model.reset();
            model.prefill(warm_prompt);
            model.forward(1);
        }
        std::ofstream out(args.at("output") + ".json");
        out << std::setprecision(12) << "{\"model_init_ms\":" << init_ms << ",\"requests\":[";
        std::vector<int32_t> miss_tokens;
        for (int phase = 0; phase < 2; ++phase) {
            model.reset();
            const auto before = model.paged_stats();
            const auto graph_before = model.cuda_graph_stats();
            const auto start = bench::Clock::now();
            const auto& logits = model.prefill(prompt);
            std::vector<int32_t> tokens{argmax(logits)};
            const auto ttft = bench::seconds(start) * 1000;
            const auto after_prefill = model.paged_stats();
            std::vector<double> decode;
            for (std::size_t i = 1; i < generated; ++i) {
                const auto begin = bench::Clock::now();
                tokens.push_back(argmax(model.forward(tokens.back())));
                decode.push_back(bench::seconds(begin) * 1000);
            }
            const auto total = bench::seconds(start) * 1000;
            const auto after = model.paged_stats();
            const auto graph_after = model.cuda_graph_stats();
            if (phase) out << ',';
            out << "{\"phase\":" << bench::quote(phase ? "repeat" : "miss")
                << ",\"ttft_ms\":" << ttft << ",\"tpot_ms\":"
                << std::accumulate(decode.begin(), decode.end(), 0.0) / decode.size()
                << ",\"request_ms\":" << total << ",\"prompt_tokens\":" << prompt.size()
                << ",\"output_tokens\":" << tokens.size()
                << ",\"prefix_hit_tokens\":" << after_prefill.last_prefix_hit_tokens
                << ",\"prefill_ssd_read_bytes\":" << after_prefill.ssd_read_bytes - before.ssd_read_bytes
                << ",\"prefill_host_read_bytes\":" << after_prefill.host_read_bytes - before.host_read_bytes
                << ",\"prefill_host_written_bytes\":" << after_prefill.host_written_bytes - before.host_written_bytes
                << ",\"host_read_bytes\":" << after.host_read_bytes - before.host_read_bytes
                << ",\"host_written_bytes\":" << after.host_written_bytes - before.host_written_bytes
                << ",\"ssd_read_bytes\":" << after.ssd_read_bytes - before.ssd_read_bytes
                << ",\"ssd_written_bytes\":" << after.ssd_written_bytes - before.ssd_written_bytes
                << ",\"gpu_evicted_pages\":" << after.gpu_evicted_pages - before.gpu_evicted_pages
                << ",\"demand_fault_pages\":" << after.demand_fault_pages - before.demand_fault_pages
                << ",\"host_peak_bytes\":" << after.host_peak_bytes
                << ",\"pinned_staging_bytes\":" << after.staging_bytes
                << ",\"gpu_kv_peak_bytes\":" << after.gpu_peak_bytes
                << ",\"gpu_limit_bytes\":" << after.gpu_limit_bytes
                << ",\"host_limit_bytes\":" << after.host_limit_bytes
                << ",\"graph_replays\":" << graph_after.replays - graph_before.replays
                << ",\"graphs\":" << graph_after.graphs
                << ",\"graph_workspace_bytes\":" << graph_after.workspace_bytes
                << ",\"ballast_bytes\":" << ballast_bytes << ",\"tokens\":[";
            for (std::size_t i = 0; i < tokens.size(); ++i) { if (i) out << ','; out << tokens[i]; }
            out << "],\"decode_samples_ms\":[";
            for (std::size_t i = 0; i < decode.size(); ++i) { if (i) out << ','; out << decode[i]; }
            out << "]}";
            if (!phase) miss_tokens = tokens;
            if (phase && tokens != miss_tokens) throw std::runtime_error("Repeated prefix changed generated tokens");
        }
        out << "]}\n";
        if (!out) throw std::runtime_error("Cannot write model results");
        if (ballast) check(cudaFree(ballast));
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
