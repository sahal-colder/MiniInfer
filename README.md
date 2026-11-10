# Infer

Infer 提供两种运行方式，共用 GPU、Pinned CPU、SSD 三级 KV 缓存和存储运行时。

| 模式 | 用途 |
| --- | --- |
| `native` | 单序列 Llama FP32 推理，支持批量 prefill、Paged KV 和前缀复用。异步 KV 流水线与 CUDA Graph 可通过 C++ 选项启用。 |
| `vllm` | 通过 `MiniInferTieredConnector` 卸载和恢复 vLLM 的 KV 缓存。 |
| `storage` | 仅构建存储和分配器库，供 C/C++ 调用。 |

## 统一入口

使用 Linux 或 WSL2、Python 3.10+、CMake 3.24+、Ninja 和 C11/C++17 编译器。Native 和 vLLM 构建需要 CUDA Toolkit。选择 `uring` 需安装 liburing 开发包。

修改 [infer.conf](infer.conf)。下面是 Native 的最小配置，模型使用 llama2.c FP32 checkpoint 或含 MiniInfer 模型元数据的 FP32 Safetensors，tokenizer 需与模型匹配。

```text
framework native
model /path/to/model.bin
tokenizer /path/to/tokenizer.bin
```

```bash
python3 app/infer.py --check-config
python3 app/infer.py build
python3 app/infer.py run --prompt "Explain KV cache"
```

命令行参数优先于配置文件，配置中的相对路径以文件所在目录为基准。可用 `--config` 指定另一个文件，用 `--dry-run` 查看构建或运行计划。切换框架后重新执行 `build`，也可用不同的 `build-dir` 隔离构建。

## vLLM 接入

在安装了 GPU 版 PyTorch 和 vLLM 0.31.0 或 0.32.0 的 Python 环境中，将 `framework` 改为 `vllm`，`model` 填 Hugging Face 模型目录或 ID，`tokenizer` 可留空。用同一 Python 执行上述构建和运行命令。

Connector 支持单 GPU、单 rank。分层准入支持一个 `FullAttentionSpec` KV group。若在现有 vLLM 应用中接入，使用 [启动器的 Connector 配置](app/infer.py) 或直接调用 [分配器适配器](integrations/vllm/pool_adapter.py) 的 `install`。GPU allocator 的替换需显式安装。

## 缓存与存储

- `storage-engine` 支持 `posix`、`uring`、`auto` 和 `spdk`。默认是 `posix`。显式选择的后端不可用时会报错。
- `cpu-cache-bytes` 是 CPU 驻留预算，`ssd-bytes` 按物理记录计入头部和对齐开销。Pinned DMA staging 另计。
- SPDK 需要已构建的 SDK、bdev 配置与独占区域。通过 `spdk-sdk`、`spdk-config`、`spdk-bdev` 指定。配置的 `bdev_io_cache_size` 必须大于零。物理 NVMe 需按部署环境配置 VFIO、IOMMU 和 hugepages。
- Native 的异步 KV 与 CUDA Graph 通过 [ModelOptions](include/mini_infer/model/mini_infer_model_options.h) 组合开启。设置 `options.paged.enable_async_kv_pipeline = true` 和 `options.enable_cuda_graph = true`。

### 存储库

```bash
python3 app/infer.py build --framework storage --cuda no --build-dir build-cpu
```

C++ 入口是 [TieredStorage](include/infer/storage/tiered_storage.h)，C ABI 入口是 [infer.h](include/c/infer/infer.h)。安装包导出 `Infer::memory`、`Infer::io`、`Infer::storage` 和启用时的 `Infer::c_api`。使用 SPDK 的消费者也需通过 `INFER_SPDK_ROOT` 或 `PKG_CONFIG_PATH` 提供 SDK 及依赖。

## 验证与跑分

CUDA 回归测试需要 GoogleTest。启用 SPDK SDK 后，测试可通过仓库的 malloc bdev fixture 验证软件 I/O 路径。

```bash
cmake -S . -B build-verify -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
cmake --build build-verify --parallel 6
ctest --test-dir build-verify --output-on-failure
```

性能测试统一从 `benchmarks/run_benchmarks.sh` 启动，覆盖分配器、文件 I/O 和推理。通过 `--help` 查看配置，通过 `PYTHON` 选择包含 PyTorch 和 vLLM 的 Python 环境。

```bash
bash benchmarks/run_benchmarks.sh --quick
bash benchmarks/run_benchmarks.sh --quick --cuda off --suites allocator storage
```

`--quick` 使用合成模型。实权重测试可用 `benchmarks/prepare_model.py --hf <本地兼容 Llama 目录> --output <新目录>` 生成 FP32 权重对，然后通过 `BENCH_MODEL_PAIR` 指向生成的 `model_pair.json`。结果默认写入 `benchmarks/results/`，含实际参数、输出 token、延迟和 I/O 计数。WSL 文件与 SPDK malloc bdev 的结果仅代表各自的软件路径。

开启 `INFER_BUILD_BENCHMARKS` 后，`benchmarks/bench_allocator --backend cpu-pool --tls 0 --output <前缀>` 可关闭线程缓存做同轨迹对照。把 `--tls` 改为 `1` 测量启用缓存时的延迟和中央池访问次数。

```bash
build-verify/benchmarks/bench_migration --target ssd --directory /tmp \
  --batch 16 --bytes 65536 --rounds 100 --output /tmp/migrate-ssd
build-verify/benchmarks/bench_migration --target gpu \
  --batch 16 --bytes 65536 --rounds 100 --output /tmp/migrate-gpu
```

迁移 CSV 记录每批从提交到完成的延迟及有效载荷吞吐，计时外校验回迁内容。历史入口 `miniinfer_model_bench`、`app/benchmark_paged.py` 和 `benchmark_storage_chunks` 也可独立运行。只构建旧模型跑分程序时可用 `INFER_BUILD_MODEL_BENCHMARK`。

第三方许可证保留在 [third_party](third_party) 和 [vLLM 集成目录](integrations/vllm)。
