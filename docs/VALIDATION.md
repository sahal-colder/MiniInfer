# 改造与验收记录

验收日期：2026-10-09。交付范围为 Linux / WSL 上的单序列 FP32 Llama 推理库和 CLI。

## 结构和合并

生产代码（`include/`、`src/`、`apps/` 中的 `.h/.cpp/.cu`）从 89 个文件收敛为 70 个；测试和测试参考实现单独统计。保留标量数值参考和有不同执行策略的 CUDA 内核，删除重复实现和无引用文件。

| 模块 | 整理结果 |
|---|---|
| 内存 | `memory_pool.h` 并入 `core/mini_infer_memory.h`；纯主机 API 实现由 `.cu` 改为 `.cpp`；显存池实现继续独立 |
| KV Cache | 配置、append、history、CPU 写入合入 `mini_infer_kv_cache.cpp`；低层接口合入同名头文件；CUDA 写入归入同名 `.cu` |
| FP32 GEMV | CPU 标量与 Armadillo 合入 `.cpp`；三种 CUDA 执行策略合入 `.cu`，共用点积和对齐检查 |
| Matmul | 后端选择及短包装函数、量化 Layer 合入 `mini_infer_matmul.h/.cpp` |
| INT8 GEMV / SwiGLU | 检查与各自 CPU 实现集中；可选 Armadillo 由编译宏控制 |
| Add | CPU 循环和 CUDA 内核各保留一份；保留普通 Add 与残差原位更新的不同别名规则 |
| 无效文件 | 删除空 `temp.cpp` 和无引用的 `mini_infer_model_file.h/.cpp`，保留实际 mmap 实现 |

头文件统一采用模块路径。被合并的旧头文件需要这样迁移：

| 旧头文件 | 新路径 |
|---|---|
| `mini_infer_memory_pool.h` | `mini_infer/core/mini_infer_memory.h` |
| `mini_infer_kv_cache_kernel.h` | `mini_infer/cache/mini_infer_kv_cache.h` |
| `mini_infer_matmul_backend.h` | `mini_infer/ops/matmul/mini_infer_matmul.h` |
| `mini_infer_quantized_matmul.h` | `mini_infer/ops/matmul/mini_infer_matmul.h` |
| `mini_infer_model_file.h` | `mini_infer/weights/mini_infer_mmap.h` |

原文件快照位于工作区外：`/home/wty/wty/mini_infer_backups/module_cleanup_20261009_g5caow05`，包括原始哈希清单、合并前快照和合并清单。正式工程没有复制这些历史源文件。

## 修复

- 修复 Tokenizer 的成员名拼写错误，恢复整体编译。
- 将库中的直接退出改为异常，使错误返回调用方；检查空指针、未知拷贝方向、非法设备和显存池参数。
- CPU 到 CPU 的拷贝不再因同步参数触发 CUDA 初始化；支持主机地址重叠的拷贝。
- Buffer 析构和移动赋值在释放失败时不终止宿主进程；清理错误写入 stderr。
- Tensor 显式移动后清空源对象的形状、类型和长度，避免形状已空但长度仍非零；保留共享存储的复制语义。
- 后端注册表使用稳定存储并加锁，修复新增条目导致已返回引用失效的问题。
- 显存申请成功、元数据申请失败时释放显存；回收逐块提交状态，避免中途失败后再次释放同一块内存；异常路径恢复原 CUDA device。
- 移除分配器的正常 stdout 日志和 CLI 对全局 `cout` 的重定向。文本与诊断分别输出到 stdout / stderr。
- 修复合并 Add 后破坏 MLP 原位残差更新的问题；普通 Add 的重叠拒绝行为保持不变。
- 修复动态库安装后找不到项目自身 `.so` 的问题，使用相对 `$ORIGIN` 路径，并统一使用共享 CUDA runtime。

15 项新增缺陷回归先在修复前复现失败，再验证修复；Add 的合并问题由已有 MLP CPU / CUDA 测试检出。显存池故障注入测试单独编译真实池实现并替换 CUDA 分配边界，不把测试替身链接进产品。

## 本机结果

环境：GCC 11.4、CMake 3.22.1、CUDA 12.8.93、NVIDIA GeForce RTX 3060 Laptop GPU（sm_86）；Release 构建。以下仅表示实际执行的验证，不代表其他硬件、模型格式或线上负载均已验收。

| 检查 | 结果 |
|---|---|
| Armadillo 开启 | 49 / 49 CTest 通过：30 CPU、18 CUDA、1 安装测试 |
| Armadillo 关闭 | 49 / 49 CTest 通过 |
| 独立 NumPy FP64 参考 | 6 个 MHA/GQA/MQA × 共享/独立输出权重模型，逐前缀 logits 对照；CPU 18,407 项断言，CUDA 18,409 项 |
| 真实 stories260K 对照 | 与独立 llama2.c 参考比较 80 个位置 × 512 个 logits；CPU/CUDA 贪心 token 相同 |
| logits 最大绝对误差 | CPU `1.69277e-05`，CUDA `1.81198e-05`；同时按测试容差逐元素检查 |
| CUDA memcheck | 完整框架和包含三种 CUDA 变体的 FP32 GEMV 测试均为 `ERROR SUMMARY: 0 errors` |
| 公开头文件 | 28 个头文件分别作为唯一项目 include，均可编译 |
| 固定测试数据 | 在独立目录重新生成，与随项目附带的数据逐字节一致 |
| 静态 / 动态库安装 | 安装目录迁移后，独立 C++ 项目可 find_package、链接、运行 GEMV 和完整模型；安装后的 CLI 可运行 |
| 安装版端到端生成 | 相同 prompt、贪心解码，CPU / CUDA 各生成 64 个新 token，stdout 文本完全一致 |

### CPU Sanitizer 的明确边界

Debug、标量后端下，29 项 CPU 测试在 ASan + UBSan + LeakSanitizer 开启时通过。`core.cuda_error` 验证真实 CUDA 错误转换，会调用 `cudaGetErrorString`；本机 WSL CUDA 驱动路径报告 50,212 字节、5 笔退出时未释放分配。

该报告已由一个**不链接 MiniInfer**、只调用 `cudaGetErrorString(cudaErrorInvalidValue)` 的最小程序复现，堆栈含 `/usr/lib/wsl/lib/libcuda.so.1`。该单项关闭 leak 检查后通过 ASan / UBSan；没有为整个测试集关闭 leak 检查，也不将此单项记作 leak 检查通过。

## 复现命令与日志

Release 构建、模型运行和普通测试命令见根目录 README。动态库验证：

```bash
cmake -S . -B build-shared -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DMINI_INFER_TEST_CUDA=OFF
cmake --build build-shared --target mini_infer_main --parallel 4
ctest --test-dir build-shared -R package.install --output-on-failure
```

CPU 检查：

```bash
cmake -S . -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DMINI_INFER_USE_ARMADILLO=OFF \
  -DMINI_INFER_TEST_CUDA=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined -no-pie'
cmake --build build-asan --parallel 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-asan -L cpu -E '^core.cuda_error$' --output-on-failure
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-asan -R '^core.cuda_error$' --output-on-failure
```

CUDA 检查：

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build/tests/test_framework data/fixtures/framework --cuda
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build/tests/test_sgemv --cuda
```

本次日志保留在 `build/validation/`、`build-scalar/validation/`、`build-asan/validation/`。CTest 的具体断言和输出在对应构建目录的 `Testing/Temporary/LastTest.log`；后续运行会覆盖该文件。`build/validation/source-sha256.json` 记录交付代码哈希。构建目录均为可重新生成的本机产物。

当前没有多请求调度、服务端 API、分页 KV Cache 或其他 checkpoint 格式；这里的验证不能代替这些功能的实现和目标部署环境下的压力测试。
