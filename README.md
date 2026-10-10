# MiniInfer

C++17 / CUDA 的单序列 FP32 Llama 推理库与命令行程序。模型加载、算子、KV Cache、分词、采样和生成流程按模块组织；命令行入口调用同一套库接口。

当前支持 CPU / CUDA 推理、MHA / GQA / MQA、共享或独立输出权重、贪心与 temperature / top-k 采样。INT8 分组权重 GEMV 作为独立算子提供，完整模型加载仍使用 FP32。

## 目录

```text
include/mini_infer/   公开 C++ 头文件
src/                 各模块实现
  core/              Buffer、Tensor、CPU/CUDA 分配器、显存池
  backend/           算子注册和后端选择、RMSNorm 内核
  cache/             连续 KV Cache、视图和写入内核
  ops/               Add、Attention/QKV、Embedding、Matmul/GEMV、MLP、RoPE、Softmax
  weights/           mmap、权重读取、分组量化
  model/             模型权重、TransformerBlock、LlamaModel
  generation/        Tokenizer、Sampler、生成循环
apps/cli/            参数解析、main、文本输出
tests/               按模块划分的测试、独立参考实现、安装包测试
data/                示例模型、配套 tokenizer、固定数值测试数据
tools/               重建测试数据的工具
cmake/               安装导出配置
docs/                改造与验收记录
build/               本机生成的编译结果，不是源代码
```

`include/mini_infer/` 与 `src/` 的模块目录相对应。`cli` 是 *command-line interface*，负责接收命令行参数和显示结果；推理逻辑放在库中，其他应用可以直接链接使用。

## 构建

支持 Linux / WSL，权重映射使用 POSIX mmap。需要 CMake ≥ 3.18、C++17 编译器、CUDA Toolkit；即使选择 CPU 运行，当前构建也需要 CUDA Toolkit。测试额外需要 C 编译器和 Python 3；重建测试数据才需要 NumPy。Armadillo 可选，关闭或未安装时使用标量 CPU 实现。

在项目根目录执行：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure --parallel 4
```

`86` 对应本机 RTX 3060 Laptop GPU；为其他 GPU 构建时设置相应的 CUDA 架构，CUDA 安装位置也应按实际环境修改。没有 GPU 时配置 `-DMINI_INFER_TEST_CUDA=OFF`，仍可运行 CPU 测试。只构建应用和库时设置 `-DBUILD_TESTING=OFF`。

关闭 Armadillo：

```bash
cmake -S . -B build-scalar -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMINI_INFER_USE_ARMADILLO=OFF
cmake --build build-scalar --parallel 6
ctest --test-dir build-scalar --output-on-failure --parallel 4
```

## 运行

项目自带的小模型用于功能验收和示例，生成质量不能代表大模型：

```bash
./build/mini_infer_main \
  --model data/models/stories260K.bin \
  --tokenizer data/tokenizers/tok512.bin \
  --prompt "Once upon a time" \
  --steps 32 --device cpu
```

将 `--device cpu` 改为 `--device cuda` 使用 GPU。`--steps` 是最大新增 token 数，`--temperature 0` 为贪心；随机采样可设置 `--temperature 0.8 --top-k 40 --seed 42`。`--help` 显示全部选项。

生成文本写入 stdout，配置、耗时和错误写入 stderr。参数或模型错误返回非零退出码。`request_seconds` 包含编码、prefill、decode 和输出，不能作为纯 decode 吞吐。

模型必须是 **llama2.c legacy FP32 / export version 0**：28 字节、7 个 int32 的小端文件头，随后为完整 FP32 权重及旧格式 RoPE 表；Tokenizer 必须是与模型词表匹配的 `tokenizer.bin`。RoPE theta 固定为 10000，RMSNorm epsilon 为 1e-5。不读取 GGUF、safetensors、`tokenizer.model` 或完整量化 checkpoint。

## 安装和库调用

```bash
cmake --install build --prefix "$PWD/install"
```

安装包含库、公开头文件、`bin/mini_infer_main` 和 CMake package；模型与测试数据由应用自行配置，不写入安装目录。
默认构建静态库，配置时增加 `-DBUILD_SHARED_LIBS=ON` 可构建动态库；项目自身的动态库使用相对加载路径，安装目录可以整体移动。

外部项目的 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyInferenceApp LANGUAGES CXX)
find_package(MiniInfer 0.1 CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE MiniInfer::mini_infer)
```

配置外部项目时设置 `-DCMAKE_PREFIX_PATH=/absolute/path/to/install`。使用模块限定的头文件路径，例如：

```cpp
#include <mini_infer/generation/mini_infer_generation.h>
#include <iostream>

int main() {
    using namespace mini_infer;
    auto allocator = std::make_shared<CPUDeviceAllocator>();
    LlamaModel model("model.bin", allocator);
    Tokenizer tokenizer("tokenizer.bin", model.config().vocab_size);
    Sampler sampler;
    generate(model, tokenizer, sampler, "Once upon a time", 32,
             [](const std::string& text) { std::cout << text; });
}
```

`MiniInfer::memory` 可供仅使用内存分配和 Buffer 的应用链接。完整外部调用例子及安装迁移测试在 `tests/package/`。

## 接口约定

- 一个 `LlamaModel` 对象维护一条序列，不能并发调用。`generate()` 为新请求重置缓存；逐 token 调用 `forward()` 时自行调用 `reset()` 切换请求。当前 prefill 按 token 顺序执行，没有请求调度和动态 batching。
- `forward()` 返回模型内部 logits 工作区，下一次调用会覆盖它；需要保留时使用 `clone()`。Tensor 的普通复制共享存储，`clone()` 才复制数据，`view()` 会保持自有底层存储存活。外部借用指针的生命周期由调用者负责。
- 独立 CUDA 算子异步提交到指定 stream。跨 stream 使用需要 event 或同步，读取结果、释放或复用 Tensor/显存池内存前必须等待完成。高层 `LlamaModel::forward()` 是同步接口。
- 普通 `AddLayer` / `ElementwiseAdd` 禁止输出覆盖输入；MLP 残差底层加法支持完全原位更新，拒绝部分重叠。合并后的 Add 内核共享计算实现，检查策略保持区分。
- 参数、形状、文件和 CUDA 错误通过异常返回，库不主动退出宿主进程。RAII 清理不抛出异常，释放失败记录到 stderr；调用者仍需处理 CUDA 运行故障。

## 验证与数据

`ctest -L cpu`、`ctest -L cuda` 和 `ctest -L packaging` 可分别选择测试；从其他目录执行时加 `--test-dir build`。验收覆盖算子尾部与对齐、别名和生命周期、内存池故障注入、六组独立 NumPy 模型参考、真实模型逐 token 对照、CLI 错误路径，以及安装目录迁移后的外部调用。

`data/fixtures/framework/` 已包含固定测试数据，日常测试无需下载或 NumPy。需要重建时运行 `python3 tools/make_fixtures.py`，随机种子固定为 `20261009`。文件来源和哈希见 `data/SOURCES.json`，测试参考代码的 MIT 许可在 `tests/reference/reference_LICENSE`。参考实现仅参与测试，不链接进产品库或 CLI。

本次合并、修复和本机验证结果见 [docs/VALIDATION.md](docs/VALIDATION.md)。
