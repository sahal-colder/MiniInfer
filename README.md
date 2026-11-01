# Infer · MiniInfer Storage Runtime

面向大模型推理的 C++17 / C11 / CUDA 内存与分层 KV Cache 存储工程，同时包含一个可独立运行的原生 Llama 推理框架。

**核心目标：在有限显存下保留和复用 KV 数据，减少旧前缀被回收后的重复计算，并让业务层通过稳定的 Block 接口使用 CPU、Pinned Memory、GPU 和 SSD。**

本仓库包含三个可以组合使用的部分：

| 部分 | 用途 | 入口 |
|---|---|---|
| Storage Runtime | 内存池、逻辑 Block、Lease、CPU/GPU/SSD 迁移、异步 I/O | [`include/infer`](include/infer)、[`src/storage`](src/storage) |
| Native Inference | 自制 Tensor、Llama 模型、CPU/CUDA 算子、Tokenizer、Sampler、生成 CLI | [`include/mini_infer`](include/mini_infer)、[`src/model`](src/model)、[`app`](app) |
| vLLM Integration | 将共享内存池和 CPU/SSD KV 存储接入 vLLM | [`integrations/vllm`](integrations/vllm) |

Native 和 vLLM 接入复用同一套物理资源与存储实现。vLLM 模式的模型计算、请求调度和 GPU KV BlockPool 仍由 vLLM 提供。

> 仓库仅保留 `main` 分支，包含产品源码、构建配置、必要依赖和本文档。测试、Benchmark、性能日志、模型权重、本地备份和编译产物不随本次产品源码发布。仓库中的 `v2` 是 C ABI 的构建入口，不是另一套独立的逻辑存储引擎。

## 导航

- [解决的问题](#解决的问题)
- [能力与当前边界](#能力与当前边界)
- [架构与数据模型](#架构与数据模型)
- [环境要求](#环境要求)
- [快速构建](#快速构建)
- [C++ 使用示例](#c-使用示例)
- [C ABI 与可替换设备](#c-abi-与可替换设备)
- [原生推理](#原生推理)
- [vLLM 接入](#vllm-接入)
- [异步与所有权约定](#异步与所有权约定)
- [构建选项](#构建选项)
- [性能与验证状态](#性能与验证状态)
- [目录与阅读顺序](#目录与阅读顺序)
- [第三方代码与许可证](#第三方代码与许可证)

## 解决的问题

长上下文、多轮对话和并发请求会产生大量 KV Cache。显存容量有限，旧前缀的 KV 被回收后，再次访问可能需要重新计算；同时，把数据搬到主机或 SSD 会引入缓冲所有权、异步完成、版本一致性和资源回收问题。

Infer 将这些职责分开：

- **逻辑数据生命周期**：谁拥有 Block、内容是否有效、版本是多少、当前位于哪里、能否迁移或释放。
- **物理资源生命周期**：CPU 内存和显存怎样分配、复用，CUDA Stream/Event/Graph 何时允许回收资源。
- **数据搬运**：异步文件 I/O、CPU/GPU 拷贝、Pinned 中转、完成等待和失败回滚。
- **框架策略**：哪些前缀值得保留、何时预取、如何选择驻留块，由对应集成和缓存策略表达。

业务调用不需要计算 SSD 文件偏移，也不需要参与内存池分桶。底层实现可通过已有合同注入；数据有效性和实际完成不能因更换后端而改变。

## 能力与当前边界

| 模块 | 已有能力 | 边界 |
|---|---|---|
| CPU Pool | 对齐分配、容量桶、线程本地缓存、跨线程归还、预算、OOM 回收重试 | 与逻辑 KV Block 分开管理 |
| Pinned Pool | 锁页内存分配与复用、DMA staging | 覆盖或释放必须晚于相关 DMA 完成 |
| CUDA Pool | Stream 关联、Event、pending/ready、拆分合并、OOM 恢复 | 跨 Stream 使用需要明确依赖 |
| CUDA Graph | Capture / workspace 生命周期支持 | Graph 引用的地址不能任意迁移或释放 |
| Storage Core | 动态 BlockId、READ/WRITE Lease、版本、预算、同步/异步迁移 | 每个实例管理自己的资源，销毁外壳不能与新调用竞争 |
| KV Offload | 框架槽绑定、CPU 驻留、SSD 下放、LRU、可选热点保护、预取和副本复用 | 不替代推理框架的请求调度或前缀匹配 |
| SSD | 定长物理记录槽、有效长度、身份/版本/CRC32C 校验、私有文件分片 | 临时 spill，不是重启恢复或掉电持久化格式 |
| I/O | POSIX、可选 dlsym / io_uring、可注入字节设备 | 明确请求不可用后端时会返回错误 |
| SPDK | 可选外部 SDK 后端与设备合同 | 不随仓库内置 SDK；当前硬件环境未完成验证 |
| Native | 单序列 FP32 Llama、MHA/GQA/MQA、CPU/CUDA、采样和 KV checkpoint | 没有连续批处理或分页 Attention |
| vLLM | 实验性单 rank connector、批量 DMA、CPU/SSD 分层 | 针对 vLLM 0.31.0、一个 FullAttention KV group 验证 |

## 架构与数据模型

```mermaid
flowchart TD
    Native[Native Runtime] --> API[C++ API / C ABI]
    Other[其他推理框架] --> API
    VLLM[vLLM 调度与 GPU KV 管理] --> Connector[MiniInfer Connector]
    Connector --> Policy[KVOffloadCache 槽绑定与缓存策略]
    Policy --> Core[TieredStorage / StorageState]
    API --> Core
    Core --> CPU[CPU / Pinned 内存后端]
    Core --> GPU[GPU 内存与完成后端]
    Core --> SSD[SSD 记录与校验后端]
    CPU --> HostPool[物理 CPU / Pinned Pool]
    GPU --> CudaPool[物理 CUDA Pool / Graph Workspace]
    SSD --> IO[FileIO / DeviceIO]
    IO --> Posix[POSIX]
    IO --> Uring[io_uring]
    IO --> Device[注入设备 / 可选 SPDK]
```

### 一份权威逻辑记录

[`LogicalBlockRecord`](src/storage/storage_state.h) 保存逻辑身份、有效大小、内容版本、位置、Lease、当前物理分配，以及正在进行的迁移或待清理资源。

核心目录为 `SlotTable<unique_ptr<LogicalBlockRecord>>`。句柄带 tag 和直接槽索引，复用目录槽不会让旧句柄重新有效；记录地址在目录扩容时保持稳定。

`KVOffloadCache` 使用连续 Entry 数组保存框架槽绑定、LRU 链接、准入/热点/预取状态。Entry 借用核心记录，不另存一套位置、大小和版本。逻辑 BlockId、框架槽号、SSD 物理槽、物理池指针分别承担不同职责，不能相互当作地址使用。

### 迁移是一段明确的资源事务

```text
保护源 + 预留目标预算
        ↓
分配目标资源
        ↓
复制 / I/O / 必要的 Pinned 中转
        ↓
等待实际完成 + 校验目标
        ↓
提交新位置
        ↓
释放或保留旧源；释放成功后归还预算
```

提交前失败保留原有效源。提交后源清理失败会报告 `cleanup_pending`，保留资源记录和预算，允许重试。不能确认 CUDA 完成的资源被隔离，不进入可复用列表。

### SSD 读取路径

文件记录由 64 字节头和载荷组成，校验身份、版本、长度、头 CRC 和载荷 CRC32C。物理槽布局由后端负责，业务不能依赖 `BlockId × stride` 寻址。

并行批读复用已有 I/O lane，并按约 1 MiB 载荷组织读取/CRC 窗口：小记录继续合批，大记录加入后结束当前窗口。窗口完成后及时校验，再处理后续记录；载荷直接进入最终目标缓冲。当前保留独立的头/载荷读请求，没有新增整块拼接缓冲。

## 环境要求

| 构建内容 | 依赖 |
|---|---|
| CPU Storage Runtime | Linux/WSL2、C/C++17 编译器、pthread、CMake；无需 CUDA Toolkit |
| C ABI | C11 + C++17、CMake **3.24+**；CUDA 可关闭 |
| 独立 C 物理模块 | C11、pthread；关闭逻辑 runtime 后无需 C++ |
| CUDA / Pinned / Graph | NVIDIA GPU、兼容驱动、CUDA Toolkit；ASYNC/POOL 模式要求 Toolkit 11.2+ |
| Native 推理 | CUDA Toolkit / nvcc；CPU 运行模式同样需要构建此后端 |
| vLLM 桥接 | CUDA、所选 Python 环境中的 PyTorch、vLLM **0.31.0** |
| io_uring | 可选系统 liburing、允许 io_uring 的 Linux 内核 |
| SPDK | 可选外部 SDK、对应 pkg-config 文件和设备配置 |

根工程最低 CMake 为 3.18；为统一使用 C ABI 和 `native` CUDA 架构检测，建议直接使用 CMake 3.24+。Ninja 可选。原生 Armadillo 加速可选，未找到时使用标量 CPU 实现。

Ubuntu / WSL 的基础工具可以安装为：

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build pkg-config liburing-dev
cmake --version
```

如果系统 CMake 低于 3.24，在自己的开发环境中安装较新版本。CUDA、PyTorch 和 vLLM 应使用与实际 GPU/驱动兼容的组合。

## 快速构建

以下命令均从仓库根目录运行。不同能力使用独立构建目录，避免旧 CMake cache 混用。

```bash
git clone https://github.com/sahal-colder/MiniInfer.git
cd MiniInfer
```

### 1. 只构建 CPU 存储库

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFER_ENABLE_CUDA=OFF \
  -DINFER_BUILD_NATIVE=OFF \
  -DINFER_BUILD_VLLM=OFF
cmake --build build-cpu --parallel 6
```

得到 `infer_memory` 和 `infer_storage` 静态库。CPU-only 构建不会要求 CUDA SDK，不能使用 GPU/Pinned 分配功能。

### 2. 启用 CUDA 存储库

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFER_ENABLE_CUDA=ON \
  -DINFER_BUILD_NATIVE=OFF \
  -DINFER_BUILD_VLLM=OFF
cmake --build build-cuda --parallel 6
```

这一配置只构建共享基础设施，不构建 Native 推理算子。若构建需要 nvcc 的后端，可通过 `CMAKE_CUDA_COMPILER` 和 `CMAKE_CUDA_ARCHITECTURES` 指定 Toolkit 与目标架构。

### 3. 构建 C ABI

```bash
cmake -S v2 -B build-capi -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFER_ENABLE_CUDA=OFF
cmake --build build-capi --parallel 6
```

生成 `libinfer.so.3`，以及启用时的 `libinfer_preload.so`。公开 ABI 为 **3**；ABI 2 的二进制调用者需要重新编译。

只构建 C 物理模块及适配器：

```bash
cmake -S v2 -B build-physical -G Ninja \
  -DINFER_BUILD_RUNTIME=OFF \
  -DINFER_BUILD_PRELOAD=OFF
cmake --build build-physical --parallel 6
```

产品快照不包含测试、示例或 Benchmark 目录，相关构建开关默认关闭，不应在此源码包中开启。

## C++ 使用示例

下面是一个完整的 CPU 写入、异步下放 SSD、校验读回示例。调用方只使用逻辑 BlockId，不操作文件偏移。

```cpp
#include <infer/storage/tiered_storage.h>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>

int main() {
    infer::StorageConfig config;
    config.block_bytes = 4096;
    config.budget.cpu_bytes = 8192;
    config.budget.ssd_bytes = 8 * 1024 * 1024;
    config.ssd_directory = "/tmp";
    infer::TieredStorage storage(config);

    const char payload[] = "hello from MiniInfer";
    const auto id = storage.create(sizeof(payload));
    {
        auto lease = storage.acquire(id, infer::AccessMode::WRITE);
        std::memcpy(lease.writable_data(), payload, sizeof(payload));
        lease.finish();
    }

    auto migration = storage.migrate_async(id, infer::StorageLocation::SSD);
    const auto result = migration.get();
    if (result.error)
        std::rethrow_exception(result.error);
    if (!result.committed || result.cleanup_pending)
        throw std::runtime_error("Migration did not complete cleanly");

    char restored[sizeof(payload)]{};
    storage.read_to_host(id, restored, sizeof(restored));
    if (std::memcmp(payload, restored, sizeof(payload)) != 0)
        throw std::runtime_error("Restored content mismatch");
    std::cout << restored << '\n';
    if (!storage.erase(id))
        throw std::runtime_error("Block is still in use");
}
```

### 安装并供外部项目链接

```bash
cmake --install build-cpu --prefix "$PWD/install-cpu"
```

将上例保存为外部项目的 `main.cpp`，使用：

```cmake
cmake_minimum_required(VERSION 3.18)
project(StorageExample LANGUAGES CXX)
find_package(Infer CONFIG REQUIRED)
add_executable(storage_example main.cpp)
target_link_libraries(storage_example PRIVATE Infer::storage)
```

配置外部项目时传入 `-DCMAKE_PREFIX_PATH=/absolute/path/to/install-cpu`。公开目标包括 `Infer::memory`、`Infer::storage`；启用 Native 时还提供 `Infer::native`。兼容的 `MiniInfer::*` 目标继续保留。

## C ABI 与可替换设备

C 头文件不暴露 CUDA、liburing 或 SPDK 结构体，业务通过 opaque handle 操作资源：

| 接口 | 职责 |
|---|---|
| `if_runtime_allocate / release` | 创建和释放动态逻辑 Block |
| `if_runtime_write / read` | 使用主机缓冲写入和读取 |
| `if_runtime_acquire` | 请求 CPU/GPU 访问；必要时迁移，返回 Lease |
| `if_lease_finish / destroy` | 完成访问、提交 WRITE、销毁访问句柄 |
| `if_runtime_migrate_async` | 提交有界异步迁移 |
| `if_migration_poll / wait` | 查询或等待真正完成的迁移结果 |
| `if_runtime_retry_cleanup` | 重试迁移遗留资源清理 |
| `if_store_*` | 框架固定槽兼容入口，复用同一逻辑核心 |
| `if_aio_*` | 有界异步存储作业 |
| `if_pool_*` | 可独立使用的 C 物理内存池 |

成功通常返回 0，读取返回长度；失败采用 `-errno`。`if_migration_poll` 返回 1 表示未完成。迁移结果中的 `committed`、`cleanup_pending` 和 `error` 必须一起处理。C++ 异常不会越过 C ABI。

`allocate` 后内容尚未初始化，完整 `write` 或成功 WRITE Lease `finish` 后才能读取。READ Lease 可共享，WRITE 独占；持有 Lease 时不能迁移或释放同一块。

初始化时通过 `if_upstream` 注入内存上游，通过 `if_dev_ops` / `if_dev` 注入字节设备。现有 POSIX、RTLD_NEXT dlsym、io_uring 实现可以独立选择；业务 allocate/read/write 不变。C++ 调用者也可以通过现有 allocator / StorageBackend 合同注入资源。

```bash
cmake --install build-capi --prefix "$PWD/install-capi"
```

C 头安装在 `include/miniinfer/infer/`，避免与 C++ 的 `infer/infer.h` 冲突。C 程序包含 `<infer/runtime.h>` 时，编译增加 `-I<install>/include/miniinfer`，链接 `-L<install>/lib -linfer`，并配置运行期库搜索路径。

`INFER_IO_DLSYM`、`INFER_IO_URING` 控制 C DeviceIO 模块；根工程的 `INFER_ENABLE_IO_URING` 控制 C++ 文件后端。两条构建路径使用不同开关，不要把关闭其中一个理解为删除所有 liburing 依赖。

`LD_PRELOAD` 分配垫片为可选 glibc 模块；动态符号解析只发生在适配边界，包含一次初始化和递归 bootstrap。它不是业务使用 Storage Runtime 的前提，也不支持在仍有存活分配时热卸载。

## 原生推理

Native 后端包含 Tensor、连续 KV Cache、权重加载、Attention/QKV、MLP、RoPE、Embedding、Softmax、GEMV、Tokenizer、Sampler 和 CLI。

```bash
cmake -S . -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFER_ENABLE_CUDA=ON \
  -DINFER_BUILD_NATIVE=ON \
  -DINFER_BUILD_VLLM=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-native --parallel 6
./build-native/mini_infer_main --help
```

`native` 架构检测需要 CMake 3.24+ 和可见 GPU。交叉编译或没有可见 GPU 时，按目标硬件改为 Toolkit 支持的明确架构值。

模型权重和 Tokenizer 由使用者提供：

```bash
./build-native/mini_infer_main \
  --model /path/to/model.bin \
  --tokenizer /path/to/tokenizer.bin \
  --device cpu \
  --prompt "Once upon a time" \
  --steps 64
```

切换 `--device cuda` 使用 GPU。支持 greedy、temperature、top-k 采样；全部参数以 `--help` 为准。

**当前 Native 模型格式：** `llama2.c` legacy FP32 checkpoint（28 字节头）与匹配的 `tokenizer.bin`。不直接读取 GGUF、safetensors 或完整量化 checkpoint；独立 INT8 分组 GEMV 算子不等于完整 INT8 模型支持。

一个 `LlamaModel` 维护一条序列，不能并发调用。`prefill(tokens)` 和 `forward(token)` 推进同一 KV 状态；prefill 当前仍逐 token 执行 decoder，没有批量 GEMM prefill 或多请求调度。

### 共享池与 KV checkpoint

`infer::native::AllocatorAdapter` 将共享分配器接入原生 Tensor/模型；`infer::native::KVCacheStorage` 将各层有效 K/V 保存到共享 CPU/SSD 存储。

Checkpoint 按最多 256-token 等量的页组织，尾页仅保存有效字节；恢复先校验暂存数据，再更新目标。它是显式 save/restore，不自动把连续 Attention KV 改为分页布局，也不负责模型 position 或调度状态恢复。恢复暂存与有效载荷总量相关，不能当作固定额外内存大小的流式恢复。

## vLLM 接入

此集成为实验性单 rank 接入，明确检查 vLLM 0.31.0；升级 vLLM 后需要重新验证其内部接口和布局。模型、请求调度、前缀匹配仍由 vLLM 提供。

先准备兼容的 PyTorch / CUDA Python 环境，再安装对应版本并编译桥接库：

```bash
python3 -m pip install 'vllm==0.31.0'
cmake -S . -B build-vllm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFER_ENABLE_CUDA=ON \
  -DINFER_BUILD_NATIVE=OFF \
  -DINFER_BUILD_VLLM=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-vllm --parallel 6

export PYTHONPATH="$PWD/integrations/vllm${PYTHONPATH:+:$PYTHONPATH}"
export MINIINFER_BRIDGE="$PWD/build-vllm/libminiinfer_vllm_bridge.so"
export MINIINFER_POOL_BRIDGE="$PWD/build-vllm/libminiinfer_pool_bridge.so"
```

下面给出 CPU/SSD 分层 connector 的配置形式。`model` 可以是本地 Hugging Face 模型目录；硬件、模型和显存预算需匹配。

```python
from pathlib import Path
from vllm import LLM, SamplingParams

cache = Path("/tmp/miniinfer-kv")
cache.mkdir(parents=True, exist_ok=True)

llm = LLM(
    model="/path/to/huggingface-model",
    enable_prefix_caching=True,
    kv_transfer_config={
        "kv_connector": "MiniInferTieredConnector",
        "kv_connector_module_path": "tiered_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lazy_offload": False,
            "kv_offload_backend": "disk",
            "use_page_cache": True,
            "disk_path": str(cache / "kv.cache"),
            "cpu_bytes_to_use": 32 << 30,
            "disk_capacity_bytes": 32 << 30,
            "miniinfer_cpu_bytes": 256 << 20,
            "disk_buffer_slots": 64,
            "miniinfer_min_reuse": 1,
        },
    },
)
outputs = llm.generate(
    ["Explain why a KV cache helps language-model inference."],
    SamplingParams(temperature=0, max_tokens=32),
)
print(outputs[0].outputs[0].text)
```

在 disk 模式下，`cpu_bytes_to_use` 是上游调度槽命名所使用的容量参数；实际 MiniInfer CPU 驻留预算由 `miniinfer_cpu_bytes` 决定。SSD 记录头和物理容量取整也占空间。`disk_buffer_slots=64` 对应每方向两个 32-block DMA 窗口；两个方向的 Pinned 开销约为 `2 × disk_buffer_slots × payload_bytes`，不是免费的额外缓存。

这段配置只启用分层 connector。若还要让 GPU KV Tensor backing 和 Pinned 分配经过共享物理池，在创建 LLM 前使用 [`pool_runtime.Pools`](integrations/vllm/pool_runtime.py) 与 [`pool_adapter.install`](integrations/vllm/pool_adapter.py)；这会影响进程内分配入口，应按部署方式显式选择。

### 常用后端配置

| 环境变量 | 可选值 / 作用 |
|---|---|
| `MINIINFER_FILE_IO` | `posix` / `uring` / `auto` |
| `MINIINFER_IO_DEPTH` | 1–64，ring 的配置上限 |
| `MINIINFER_FILE_SHARDS` | 1 或 4 个私有文件分片 |
| `MINIINFER_SSD_VERIFY` | `read_back` / `on_read`；均保留消费校验 |
| `MINIINFER_WRITE_CONCURRENCY` | 有界 CPU→SSD 下放并发 |
| `MINIINFER_COMPACT_SLOTS` | `1` 使用更细的物理容量取整 |
| `MINIINFER_CPU_HOT` | `1` 启用可选 CPU 热块保护 |
| `MINIINFER_PROFILE_TRANSFERS` | `1` 开启可选传输观测，存在计时开销 |

显式 `uring` 使用真实 io_uring；`auto` 优先尝试可用 ring，不能理解成根据运行负载自动选择最快实现。当前文件 connector 要求 buffered I/O；不将原生 vLLM 的 O_DIRECT 结果和 buffered 结果混为相同缓存条件。

两个 Pinned 半区分别等待自己的 DMA 完成后才可复用。后台文件任务的终结与 GPU Event 的完成是不同条件；关停时需要排空两者。

## 异步与所有权约定

- **Lease**：资源访问凭证，不能把裸地址缓存到 Lease 结束之后。WRITE 需要显式完成提交，放弃写入会使内容不可读。
- **GPU consumer stream**：C++ acquire 使用明确 stream；C ABI 提供后端拥有的 stream，可通过 `if_lease_stream` 获取。其他 stream 的使用需要建立依赖，不会被自动追踪。
- **异步缓冲**：源、目标和 fd 必须活到同步返回或相应异步完成；submit 成功不等于数据已可消费。
- **关闭**：destroy 不得与同一外壳的新调用并发。已接受迁移会排空；Lease 可以保活核心状态。AIO 借用 store 时，过早销毁 store 返回忙。
- **清理失败**：检查显式 finish/destroy 和迁移结果。需要重试时保留句柄；不能将“新位置已提交但旧源未清理”当作完全未发生迁移。
- **Graph**：捕获的地址必须保持到所有相关 Graph/GraphExec 引用释放。物理池的 Graph 生命周期与逻辑缓存淘汰分开。
- **预算**：逻辑驻留、迁移预留、副本和待清理容量都计费；物理池统计与逻辑预算分别观察，外部注入 allocator 的额外缓存也要单独核算。

## 构建选项

优先使用 CMake 开关；[`include/infer/config.h`](include/infer/config.h) 提供默认值。配置变更后重新配置相应构建目录。安装包会导出本次构建的宏配置，不能只修改消费者宏来切换已经编译的库。

| 根工程选项 | 默认 | 含义 |
|---|---|---|
| `INFER_ENABLE_CUDA` | ON | CUDA / Pinned 及 GPU Storage |
| `INFER_BUILD_NATIVE` | config.h 默认 ON；CPU-only 默认 OFF | Native 推理和 CLI |
| `INFER_BUILD_VLLM` | config.h 默认 OFF | vLLM C++ bridges |
| `INFER_MEMORY_MODE` | 2 | 0 同步、1 CUDA async、2 缓存池 |
| `INFER_ENABLE_IO_URING` | ON，依赖可选 | C++ 文件 I/O |
| `INFER_ENABLE_SPDK` | OFF | 外部 SPDK SDK |
| `MINI_INFER_USE_ARMADILLO` | ON，仅 Native | 可选 CPU 加速 |

| C ABI 构建选项 | 默认 | 含义 |
|---|---|---|
| `INFER_BUILD_RUNTIME` | ON | 共享逻辑核心与 C 业务 ABI |
| `INFER_BUILD_PRELOAD` | ON | glibc LD_PRELOAD 分配垫片 |
| `INFER_ENABLE_CUDA` | OFF | C ABI GPU 资源能力 |
| `INFER_IO_DLSYM` | ON | RTLD_NEXT 文件适配器 |
| `INFER_IO_URING` | ON，依赖可选 | C DeviceIO 的 io_uring 实现 |

SPDK 通过系统 pkg-config 引用，不从仓库下载或编译。启用前设置 SDK 的 `PKG_CONFIG_PATH`，再配置 `-DINFER_ENABLE_SPDK=ON`；适配接口见 [`storage_backend.h`](include/infer/storage/storage_backend.h)。

## 性能与验证状态

以下为开发工作区中 2026-10-31 / 2026-11-01 的本机记录，不是所有硬件上的保证。环境为 WSL2、RTX 5060 Ti、GCC 11.4、CUDA Toolkit 12.8、PyTorch 2.13.0+cu130、vLLM 0.31.0。

1792 KiB 载荷、batch 8、128 块，同一文件读取入口，真正完成并核对数据：

| 文件场景 | 优化前 | 当前实现 | 变化 |
|---|---:|---:|---:|
| 热 buffered 读取 | 11.302 GiB/s | 13.308 GiB/s | +17.8% |
| 私有文件 fdatasync/DONTNEED 后读取 | 4.822 GiB/s | 5.916 GiB/s | +22.7% |

DONTNEED 不能驱逐 Windows/VHD/设备缓存，因此第二行不代表物理冷 NVMe 带宽。相同本机负载的 vLLM filesystem buffered 读取约 12.427 GiB/s，但文件布局、线程和发布合同不同，不能当作相同内核实现的直接倍率。

实际 Qwen3-1.7B / BF16 / CUDA Graph 模型回访的交付版两对结果为 **338→316 ms** 和 **318→321 ms**，不足以声称稳定端到端提速。双方都使用 vLLM 的模型计算；不能把文件层提高 17.8% 描述成推理 tokens/s 同幅提高。

开发工作区最终回归记录：CUDA 核心 26/26、C ABI CUDA 11/11、CPU 核心 ASan/UBSan 10/10；vLLM 全套 114 项和 4 个 subtest 通过，包含真实模型。Native 存储集成、独立 C 模块等也做过独立验证。上述测试、原始样本和基准工具按本次发布范围保留在本地，不在这个产品源码快照内，不能直接在此快照运行相应套件。

尚未完成的验证包括多 GPU、真实冷设备、SPDK 实机、生产在线 P99 和长期运行。TSan 在本机遇到启动映射错误，没有竞态检测通过结论。Native 与 vLLM 的完整模型兼容范围也不同，不能把一个框架的通过结果推广到另一个。

## 目录与阅读顺序

```text
MiniInfer/
├── CMakeLists.txt             # C++ runtime、Native、vLLM 的构建入口
├── cmake/                    # 依赖、安装包、Native 构建
├── app/                      # 原生生成 CLI
├── include/
│   ├── infer/                # 内存池、Storage、Native 适配 API
│   └── mini_infer/           # 原生推理框架 API
├── src/
│   ├── core/ cpu/ cuda/      # 共享物理资源
│   ├── storage/              # Block、Lease、迁移、KV 策略、SSD
│   └── base/ tensor/ ops/
│       model/ loader/
│       tokenizer/ sampler/
│       runtime/              # 原生推理实现
├── integrations/
│   ├── native/               # 原生 KV checkpoint
│   └── vllm/                 # connector、桥接、DMA 与策略
├── v2/
│   ├── CMakeLists.txt         # C ABI / 独立 C 模块入口
│   ├── include/infer/         # C 公开头
│   └── src/                  # C 物理模块、ABI、共享核心桥接
└── third_party/crc32c/       # 必要 CRC32C 源码与许可证
```

建议按下面顺序阅读产品代码：

1. [`tiered_storage.h`](include/infer/storage/tiered_storage.h)：业务能力与错误/完成合同。
2. [`storage_state.h`](src/storage/storage_state.h)：唯一权威逻辑记录、所有权与预算。
3. [`transfer.cpp`](src/storage/transfer.cpp)：准备、提交、完成、回滚和清理。
4. [`kv_offload_cache.cpp`](src/storage/kv_offload_cache.cpp)：框架槽绑定与缓存策略。
5. [`ssd_storage.cpp`](src/storage/ssd_storage.cpp) / [`file_io.cpp`](src/storage/file_io.cpp)：记录布局、校验和定位 I/O。
6. [`runtime.h`](v2/include/infer/runtime.h)：动态 Block 的 C ABI。
7. [`tiered_connector.py`](integrations/vllm/tiered_connector.py)：vLLM CPU/SSD 接入。
8. [`mini_infer_model.h`](include/mini_infer/model/mini_infer_model.h)：原生推理模型入口。

## 第三方代码与许可证

- CRC32C 的源代码、版权说明和 BSD-style 许可证保留在 [`third_party/crc32c`](third_party/crc32c)。
- vLLM connector 复用或改编部分 vLLM 传输逻辑，说明和 Apache-2.0 文本保留于 [`NOTICE`](integrations/vllm/NOTICE) 与 [`VLLM-LICENSE`](integrations/vllm/VLLM-LICENSE)。安装的 vLLM 源文件不会被修改。
- CUDA、PyTorch、vLLM、可选 Armadillo、liburing、SPDK 等依赖仍遵循各自许可证。

本仓库尚未声明覆盖全部自有代码的统一根许可证；不要将第三方许可证自动理解为整个项目的许可证。
