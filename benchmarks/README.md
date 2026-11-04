# MiniInfer 自动化 Benchmark

已完成的测量与限制见 [2026-11-04 验证报告](validation-2026-11-04.md)。

这套程序调用仓库的真实 C++ 分配器、Paged KV、异步双缓冲与 CUDA Graph，实现独立进程、多轮重复、原始数据留存和 Markdown 汇总。默认使用确定性的 **Mock 权重验证流程**；不会生成伪造的 GPU、SPDK 或推理数据。缺依赖的项目返回 `77` 并记录 `skipped`，运行错误/超时/greedy token 不一致返回非零并记录失败。

## 环境与快速运行

需要 Linux/WSL、Python 3.10+、CMake **3.24+**、C++17 编译器。CUDA 构建另需 CUDA Toolkit、可用 NVIDIA GPU 和兼容的 host compiler；推荐将代码/build 放在 Linux 原生文件系统。liburing 是可选依赖，Ubuntu 可安装 `liburing-dev`。SPDK SDK 由用户事先准备，脚本不下载、绑定或格式化设备。

Python 总控、报表和 Mock 模型生成只使用标准库。PyTorch allocator/vLLM 对比需要所选 Python 环境已安装 GPU 版 `torch` 和 `vllm`；真实模型转换需要 `torch`、`safetensors`、`numpy`。不自动安装/升级当前环境。vLLM 驱动按本地 `LLMEngine.from_engine_args/add_request/step` API 编写，已针对 0.32.0 接口核对；环境版本保存在结果中。

从仓库根目录执行：

```bash
# 指向同时能导入 torch 和 vllm 的 Python。PATH 中的 cmake 必须 >= 3.24。
export PYTHON=/path/to/python
export CUDA_VISIBLE_DEVICES=0
export BENCH_GPU_ID=0     # nvidia-smi 的物理 GPU 序号/UUID，必须与上面的逻辑 GPU 对应
bash benchmarks/run_benchmarks.sh --quick

# 无 GPU：实测 CPU 分配器和文件 I/O，明确跳过 CUDA 推理
bash benchmarks/run_benchmarks.sh --quick --cuda off --suites allocator storage

# 完整矩阵：输入 2048/4096、输出 128、5 个独立进程重复
bash benchmarks/run_benchmarks.sh

# 单项；--help 查看总控参数
bash benchmarks/run_benchmarks.sh --suites storage
bash benchmarks/run_benchmarks.sh --suites allocator
bash benchmarks/run_benchmarks.sh --suites e2e
```

`--quick` 使用输入 64、输出 8、4 MiB 数据集、100 次分配器操作、1 次进程重复，只用于 smoke check。完整矩阵会反复初始化模型和 vLLM，运行时间可能较长。每个 cell 串行执行，不让两个推理引擎同时竞争 GPU。每轮随机交错后端顺序，固定 seed 可复现。默认不主动填满整块桌面 GPU。

## 同一模型权重

MiniInfer 当前支持的是单序列、FP32 Llama 路径，不能把任意 Qwen/HF 目录直接当作兼容权重。总控默认生成一对小型 Mock 模型：native llama2.c FP32 v0 和 HF Llama safetensors。权重数值相同，并显式转换 Q/K 的 RoPE 排列（native 相邻旋转对、HF 半头排列）。Mock 输出与报告带标记，**不能用作真实模型简历性能数据**。

真实模型请先从本地、兼容的 HF Llama safetensors 转换：

```bash
"$PYTHON" benchmarks/prepare_model.py \
  --hf /models/compatible-llama-fp32-or-bf16 \
  --output /models/benchmark-llama-pair

export BENCH_MODEL_PAIR=/models/benchmark-llama-pair/model_pair.json
bash benchmarks/run_benchmarks.sh --suites e2e
```

转换接受无 bias、SiLU、`rms_norm_eps=1e-5`、`rope_theta=10000`、无 RoPE scaling、标准 head_dim 的 Llama；其他配置明确失败。BF16/FP16 源权重会提升为 FP32，vLLM 同样指定 FP32 和 FP32 KV，不把 BF16 的吞吐与 native FP32 混比。HF 原始文件、config、native 文件均记录 SHA256，每次启动校验。输出目录中的权重不覆盖，重新转换请使用新目录。

输入使用固定 **token ID** 序列，精确满足长度；不含 tokenizer、聊天模板和网络传输成本。greedy 解码，固定输出预算，忽略 EOS，单卡、单请求、无 speculative decoding。每个后端保存生成 token，并与同轮 sync 结果逐项比较；不一致保留性能原始数据但总控失败，不能据此发布同质量加速结论。

如需评估自然语言分布，请更换总控生成 prompt ID 文件的策略，并保存语料与 tokenizer 版本。当前结果是受控长度微基准，不能代表线上混合并发负载。

## 测试矩阵与指标口径

### 分配器

`bench_allocator.cpp` 测试：

- `cpu-raw` / `cpu-pool`：真实 `CPUDeviceAllocator` / `CachingCPUDeviceAllocator`。
- `cuda-raw` / `cuda-pool`：直接 `cudaMalloc/cudaFree` / `infer::CachingCUDADeviceAllocator`。后者正是 native 兼容包装委托的共享实现。
- `pinned-raw` / `pinned-pool`：`cudaMallocHost/cudaFreeHost` / `CachingPinnedHostAllocator`。

每线程维护有限 live slots，SplitMix64 确定性选择 slot；空 slot 分配，否则释放。尺寸在约 256 B～2 MiB 的 14 个阶梯间变化并带对齐扰动。C++ 和 Python 使用同一整数 trace。默认 1/4 线程共享同一池，包含锁竞争；测量前预热，测量后保留活跃块采一次容量快照，再清理。预热不进入延迟分布，reuse/upstream 计数减去预热基线。每次操作的线程、序号、大小、耗时保存到 CSV。

`bench_allocator.py` 额外提供：

- `torch-proxy`：`torch.empty(uint8, device='cuda')`，**PyTorch CUDA caching allocator 的 vLLM 张量路径代理**，不冒称 vLLM 独立 KV allocator。
- `vllm-cumem`：真实 `vllm.device_allocator.cumem.CuMemAllocator.use_memory_pool`，对应可选 sleep-mode 内存池，只做单线程测试；默认与其他后端的 1 线程行一起输出。该 API 的线程上下文不等价于 C++ 共享池，不伪造 4 线程结果。

Python 行包含 Tensor 创建/销毁与 Python dispatch，线程受 GIL 约束；C++ 行计时 native API。两类行**不能直接计算纯分配器加速比**。CUDA 计时是主机 API 延迟：raw free 的同步与 pool free 的事件延后语义各自保留；不在每个 pool 操作后额外加入全设备同步，也没有正在使用分配块的计算 kernel。

指标：

- `allocate/release_mean_us`、`*_p99_us`：逐操作壁钟耗时，P99 使用 nearest-rank，不从批次均值反推。
- `internal_waste_pct = (active_capacity - active_requested) / active_capacity × 100%`：活跃分配的内部碎片快照。
- `reserve_slack_pct = (reserved - active_requested) / reserved × 100%`：包含可复用空闲缓存，**不等于外部碎片**。
- `reuse_hit_pct = measured_reuse_count / measured_allocate_calls × 100%`：native 真实计数；PyTorch 使用 allocation 次数减去 `num_device_alloc` 的代理估算，不适用于所有 allocator 配置。
- 原生 CUDA 驱动未暴露可比的内部池布局、CuMem 自定义池未暴露统一容量/命中统计时记为 `N/A`。没有最大空闲块统计，不能声称测到了 `1-largest_free/total_free` 外部碎片率。raw cudaMalloc 也不被写成“碎片率=0”。

### 推理与流水线

`bench_model.cpp` 直接调用 `LlamaModel`：

- `sync`：Paged KV，同步 demand 路径，关闭预取、异步双缓冲、Graph。
- `async`：开启真实 layer-wise 异步双缓冲，关闭 Graph。
- `graph`：同一异步路径，加真实每层/slot CUDA Graph。记录 `graph_replays`，未执行 replay 则不标为已验证。
- `bench_end_to_end.py`：本地标准 vLLM `LLMEngine`，`max_num_seqs=1`，prefix caching 开启，默认允许 vLLM Graph，关闭 chunked prefill 与 detokenization。可用 `BENCH_VLLM_EAGER=1` 增加 eager 基线。

A：输入 2048/4096，输出 128。先用不同首 token 预热；随后测 `miss` 和相同 prompt 的 `repeat`。记录实际 `prefix_hit_tokens`，未命中不称为缓存命中。sync A 默认 GPU 常驻全层；async/graph A 在预算足够时为每层保留固定 GPU slot，跳过逐层 Host 上传和逐 token KV 回写。完整页前缀发布时才批量回写；预算不足则使用两个流式 slot。Prefill 按剩余 tokens 批量计算，Graph 仅用于单 token decode。记录 KV 保留容量及额外 `pinned_staging_bytes`，Host KV budget 不包含 staging。

B-host：缩小 native GPU KV budget，Host 给足以容纳 KV；async 的两个 slot 从 pinned host 读取。B-nvme：进一步限制 host budget，触发 SSD spill/read。记录 `prefill_ssd_read_bytes`、总读写字节、GPU eviction 与 demand fault。**只有 prefill 内确实读取 SSD，才将 SSD 换入证据标为 `ssd-prefill-read`**；发生 decode 读取不等于已证明 TTFT 隐藏了 NVMe 延迟。记录 `host_read_bytes` / `host_written_bytes` 及 prefill 对应字段；两者均非零时将 B-host 标为 `pinned-host-read-write`。这些计数证明 Pinned Host 路径发生读写，实际传输重叠仍需 Nsight 验证。prefix 可能因容量限制被驱逐，repeat 不保证命中。

标准 vLLM 不具备这套 native 三层 KV 的同一 SSD 接口；B 行标为 `standard-gpu-baseline-only`，不声称 vLLM 完成了相同 NVMe swap。Mock 下 vLLM 显式使用 16 MiB KV 池，避免小模型预留整卡大部分显存；真实权重默认使用标准 memory-utilization 策略，可用 `BENCH_VLLM_KV_BYTES` 显式指定 KV 池字节数。`--model-io posix` 可调试无 liburing 的环境；默认 `uring` 使用实际 O_DIRECT 路径。同步和异步都使用相同 host 参数，但实际 GPU 布局由引擎约束决定；B-nvme 的 sync GPU budget 默认仅容纳一层最大上下文，host 仅保留全层各一页的最小预算，以触发真实 spill/read。不能把受限配置与无限 GPU 配置的比值写成通用性能优势。

指标定义：

- TTFT：本地请求提交到第一个输出 token ID 可见，毫秒。模型加载、CUDA 初始化与预热排除，模型初始化耗时另存。
- TPOT：首 token 之后的 N−1 次解码平均耗时。vLLM 来自增量 `step()` 的可见 token 时间，保存 coalesced token 个数，不把合并输出虚构为精确 kernel ITL。
- VRAM：外部 `nvidia-smi` 周期采样的**整卡**已用显存最高值，覆盖初始化和请求；含桌面/其他进程，是离散采样下界。`gpu_kv_peak_bytes` 仅是 native KV 池峰值，不能作为整个进程峰值。若需要严格的进程/分配峰值，使用专用 GPU 与 Nsight/CUPTI；不要把整卡采样写成精确 allocator 峰值。

显式增加压力可设置 `BENCH_BALLAST_MIB=1024`：B 的两个引擎均在初始化前申请并触碰同样大小 GPU ballast，保留至进程结束；A 不申请。默认 0，优先通过 KV budget 触发换入。不要同时运行其他 GPU 负载。真实 4K FP32 模型如果放不下，记录 OOM，降低模型规模或显式压力，不静默换精度。

### 存储 I/O

`bench_storage.cpp` 使用独占、预分配、退出删除的普通文件，或用户显式指定的 SPDK 独占区域。4 KiB 对齐的地址/长度/偏移，固定随机内容，先写满数据集，随后交替测读/写；读回逐字节验证，准备/校验不计入耗时。

矩阵为 4 KiB、64 KiB、256 KiB、1 MiB；默认 QD=1/8/32，随机块序，支持 sequential。

- `posix-buffered`：同步 `pread/pwrite`，热页缓存基线。
- `posix-direct`：同样 POSIX 调用加 O_DIRECT，提供相同 cache policy 的同步基线。
- `uring`：仓库真实 `infer::io::DirectFile`，O_DIRECT、批量提交、完成后返回；关闭 SQPOLL/IOPOLL 以免隐含额外忙轮询核，保留真实 setup flags 和设备完成数。
- `spdk`：真实 `open_spdk_device`/DMA buffers；没有 SDK/config 时跳过。IOPS 同时保留逻辑 operation 数与 device completion 数，后者可能受拆分影响。

POSIX effective QD 永远为 1，requested QD 表示主机批大小；不能把它与 uring QD=32 的差异全部归因于系统调用。首先比较 QD=1 direct 路径，再分析队列深度扩展。GB/s 使用十进制 10^9，IOPS 为完成逻辑块数/秒。写计时不包含 flush/FUA，不代表断电持久化吞吐；初始化 POSIX 文件会 fsync。O_DIRECT 绕过 guest page cache，仍不能保证 WSL host cache、SSD cache 或物理 NAND 冷态。默认不 drop 系统缓存。

## 主要环境变量

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `PYTHON` | `python3` | 所有 Python 子进程使用相同解释器 |
| `BENCH_CUDA` | `auto` | `on/off/auto`，auto 探测 nvcc 与 nvidia-smi |
| `BENCH_CUDA_ARCH` | `native` | CMake CUDA 架构；交叉编译可设 `89` / `120` |
| `BENCH_BUILD_DIR` | `benchmarks/build/cpu` 或 `cuda` | 独立构建目录，勿指向用户正在使用的 build |
| `BENCH_RESULTS` | 时间戳+PID | 新结果目录；已存在则拒绝覆盖 |
| `BENCH_MODEL_PAIR` | 自动 Mock | `prepare_model.py` 产生的 manifest |
| `BENCH_IO_DIR` | `/tmp` | 被测盘上已存在目录；放在目标 NVMe 挂载点 |
| `BENCH_MODEL_IO` | `uring` | 推理 SSD 路径：`uring/posix/spdk` |
| `BENCH_VLLM_KV_BYTES` | Mock: 16 MiB；真实: 0 | vLLM 显式 KV 池容量；0 使用默认策略 |
| `BENCH_REPETITIONS` | `5` | 独立进程重复次数，quick 默认 1 |
| `BENCH_SEED` | `42` | 调度与工作负载 seed；同轮后端 seed 相同 |
| `BENCH_THREADS` / `BENCH_SLOTS` | `1,4` / `16` | 分配器并发线程/live slots |
| `BENCH_ALLOCATOR_STEPS` / `BENCH_ALLOCATOR_WARMUP` | `10000` / `1000` | 每线程测量/预热操作次数 |
| `BENCH_BLOCKS` | `4096,65536,262144,1048576` | I/O 块尺寸 |
| `BENCH_DEPTHS` | `1,8,32` | 请求批深度，至多 128 |
| `BENCH_STORAGE_BYTES` | `268435456` | 数据集容量，须为所有块尺寸的整数倍 |
| `BENCH_STORAGE_ROUNDS` / `BENCH_STORAGE_PATTERN` | `2` / `random` | 进程内轮数与 `random/sequential` |
| `BENCH_PROMPTS` / `BENCH_GENERATE` | `2048,4096` / `128` | 输入/输出 token 数 |
| `BENCH_MODEL_WARMUP` | `1` | 不同首 token 的请求预热次数 |
| `BENCH_WORKLOADS` | `A,B-host,B-nvme` | 推理 workload 子集 |
| `BENCH_PAGE_TOKENS` | `16` | KV 每页 token 数，须符合 64 KiB 页面容量 |
| `BENCH_GPU_BYTES` | `0` | 覆盖 native GPU KV budget；0 使用各变体默认布局 |
| `BENCH_BALLAST_MIB` | `0` | B 初始化前显式 GPU 压力，两个引擎相同 |
| `BENCH_VLLM_MEMORY` / `BENCH_VLLM_EAGER` | `0.8` / `0` | vLLM 显存利用上限/强制 eager |
| `BENCH_GPU_ID` / `BENCH_GPU_SAMPLE_SECONDS` | `CUDA_VISIBLE_DEVICES` 或 `0` / `0.1` | 单卡物理采样标识与采样间隔 |
| `BENCH_JOBS` | 最多 8 | 构建并行数 |
| `BENCH_TIMEOUT` / `BENCH_BUILD_TIMEOUT` | `1800` / `1200` | 每 cell / 每次构建超时秒数 |

完整性能测试应至少保留默认 5 个独立进程重复，增大到 10～20 更适合评估离散程度。逐操作 P99 与进程间 P99 是不同统计量；不要用 5 次 TTFT 的最大值声称稳定的线上 P99。

## SPDK：只写显式拥有的区域

普通文件测试自动创建私有文件；SPDK 直接写指定 bdev 的指定区域。只有确认区域无文件系统/业务数据且由本次测试独占后才填写以下变量。脚本不会自动选择 PCI 设备、unbind、mount、格式化或使用 sudo。

```bash
export SPDK_ROOT=/path/to/built/spdk
export SPDK_JSON=/path/to/explicit-owned-nvme-bdev.json
export SPDK_BDEV=OwnedNvme0n1
export SPDK_OFFSET_BYTES=1073741824
export ALLOW_SPDK_WRITE=YES
export BENCH_STORAGE_BYTES=268435456  # 将覆盖 offset 起的这一范围
export SPDK_MEMORY_MIB=1024         # 要容纳整个数据集 DMA + 输出 + bounce buffers
bash benchmarks/run_benchmarks.sh --suites storage
```

脚本不验证磁盘所有权，也不会替你作这个判断。应先将普通文件放在同一块盘/等价分区上单独测 POSIX/uring，再在独占 SPDK 环境测相同参数，记录绑定方式与物理设备。不能同时让内核文件系统和 SPDK 拥有同一 NVMe namespace。

仅验证 SPDK 软件栈时，可使用仓库 `tests/storage-io/spdk_malloc.json`、`SPDK_BDEV=Malloc0`、`SPDK_OFFSET_BYTES=0`、`SPDK_NO_PCI=1`、`SPDK_NO_HUGE=1`，并把数据集控制在 malloc bdev 容量内。这是 **RAM bdev**，不作为 NVMe 吞吐；保留 environment.json 中的配置来源。

总控也支持推理 SPDK：在上述独占区域变量之外设置 `BENCH_MODEL_IO=spdk` 和 `SPDK_CAPACITY_BYTES=<该区域可写容量>`，再运行 `--suites e2e`。整个容量会被作为 native SSD pool 的上限，必须能容纳目标 KV；记录中的 `spdk-capacity` 与模型预算以实际日志为准。`BENCH_MODEL_IO` 的完整取值为 `uring/posix/spdk`。

需要单独测试 native 推理 SPDK 换入时，`bench_model` 也提供真实接口（先准备可写独占区域）：

```bash
benchmarks/build/cuda/benchmarks/bench_model \
  --model /models/pair/model.bin --variant graph --workload B-nvme \
  --prompt-file /path/to/saved/prompt-2048.txt --generate 128 \
  --host-bytes 33554432 --directory /tmp --io spdk \
  --spdk-config "$SPDK_JSON" --spdk-bdev "$SPDK_BDEV" \
  --spdk-offset "$SPDK_OFFSET_BYTES" --spdk-capacity 268435456 \
  --allow-spdk-write YES --output /tmp/native-spdk-benchmark
```

使用 `SPDK_JSON` 完成一次 CUDA 构建才能获得已链接 SDK 的 `bench_model`。JSON 中是 NVMe、文件还是 malloc bdev 必须在发布结果时注明。

## 结果文件与解读

每次运行创建 `benchmarks/results/<UTC时间戳>-<PID>/`：

```text
summary.md                  Markdown 汇总，含跳过/失败与正确性状态
allocator_summary.csv       每进程分配/释放均值、P99、容量、reuse
process_statistics.csv      每配置独立进程样本数、均值/中位数/标准差/最小最大值
storage.csv                 每进程/轮次/读写 IOPS 和 GB/s
end_to_end.csv              每请求 TTFT、TPOT、实际 prefix/SSD/Graph 证据
status.csv                  包括失败、超时、跳过，不能只挑成功样本
environment.json            GPU/驱动/OS/编译器/挂载点/参数/源码哈希
worktree.patch              当前未提交的 tracked 产品改动
source_snapshot.tar.gz      本次源码快照，包括未跟踪的 C++/CUDA 改动；第三方按 submodule 记录还原
model_pair.json             配对权重、配置及 SHA256
CMakeCache.txt              实际构建参数
configure.log / build.log   构建原始输出
commands.jsonl             每个 cell 的完整 argv 和元数据
raw/*.csv                  分配器逐操作延迟 / I/O 原始轮次
raw/*.json                 推理逐 token 间隔、token IDs、统计计数器
raw/*.gpu.csv              整卡显存/利用率/温度/功率/SM clock 采样
raw/*.log                  每个子进程标准输出/错误
```

README/简历只引用真实权重、正确性通过、workload 证据成立的结果。报告中值仅用于浏览：回到 CSV 按同轮配对计算比例并检查离散度，明确硬件/TGP、精度、模型、长度、并发、cache 状态、存储路径、独立进程样本数。GPU 桌面占用、温度、时钟、系统后台负载与 WSL 缓存都能改变结论。小型 Mock、热页缓存、单轮胜出或不同 GPU budget 不能支撑“全面超过 vLLM”的结论。

## Profiling

性能采集和 profiler 分开运行；profiler 数据不混入延迟汇总。复用 `commands.jsonl` 的一个 native 命令，在外层加 `nsys profile --trace=cuda,osrt -o /tmp/miniinfer-profile`。在时间线上检查 H2D/D2H 与 layer compute 是否重叠、graph replay 与 prefill SSD read 所在区间；用 `perf record`/`perf report` 分析 C++ 调用栈，用 `py-spy` 分析 Python dispatch。先看真实重叠和请求计数器，再解释 TTFT 改善；设置开关本身不是流水线收益证据。

接口参考：[vLLM CuMemAllocator](https://docs.vllm.ai/en/latest/api/vllm/device_allocator/cumem/)、[vLLM metrics](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md)。实际运行始终以 environment.json 中的本地版本和日志为准。
