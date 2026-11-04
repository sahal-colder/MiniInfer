# Fixed-slot CUDA Graph decode

Enable `ModelOptions::enable_cuda_graph` together with
`ModelOptions::paged.enable_async_kv_pipeline`. This is an opt-in native FP32,
single-sequence path. It captures each Transformer block for both fixed KV
slots, rather than capturing storage scheduling or the whole model. The final
normalization, classifier and logits download remain outside the layer graphs.

`GraphCaptureScope` begins/ends both allocator and CUDA capture, and aborts
both on exceptions (including outstanding host leases). Pre-reserved private
scratch is never placed in eager bins or freed by eager `trim()`. Different
capture sizes restart at the same base address. The runner uses identical
allocation layouts across all captures: intermediate tensors, K/V scratch,
hidden state and device position all retain fixed addresses. Host Tensor
handles return their allocator lease immediately; graphs retain the entire
reservation until their executables are destroyed.

A pinned host position is copied asynchronously to its fixed device address
once per token, outside capture. A completion event protects reuse of that
pinned host cell. RoPE, QK, softmax and AV read position (and derive context
length) on the device. A captured scatter kernel commits new K/V into the
position-dependent page row, so GEMV outputs also have fixed addresses.
Launch grids are fixed; no per-token node updates or recaptures are needed.

All graph executions share one compute stream and one private workspace.
Independent requests need independent runners. `AsyncKVPipeline` enqueues
`wait(ready) -> compute callback / cudaGraphLaunch -> record(compute_done)`;
H2D/D2H and NVMe operations run outside capture on their existing streams.
Runner destruction waits for compute, destroys all GraphExec/Graph handles,
forgets their allocator records and frees scratch. `LlamaModel` destroys the
runner before retiring/destroying the pipeline and its KV slots.
Destroying a standalone allocator with registered graphs retains its private
reservation and prints a diagnostic instead of freeing referenced addresses;
the runner's ownership order avoids that orphaned reservation.

## Build and test

Requires CMake >= 3.24, CUDA Toolkit and GTest. Set the CUDA architecture for
your GPU (120 below is Blackwell). Add `-DGTest_DIR=...` if GTest is not in a
standard installation prefix.

```sh
cmake -S . -B /tmp/infer-cuda-graph \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DINFER_BUILD_NATIVE=ON -DINFER_BUILD_VLLM=OFF \
  -DINFER_BUILD_C_API=OFF -DINFER_BUILD_PRELOAD=OFF \
  -DBUILD_TESTING=ON -DMINI_INFER_BUILD_APP=OFF
cmake --build /tmp/infer-cuda-graph -j6
cd /tmp/infer-cuda-graph/tests/cuda-graph
ctest -V --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./cuda_graph_test \
  '--gtest_filter=-*ReportsPairedEndToEndDecodeTPOT'
```

The tests cover private-pool isolation, different-sized captures sharing the
same base, eager trim/cache eviction, exception rollback, foreign capture
requests, dynamic positions across page boundaries, changing token IDs,
context limits, reset, SSD spills and prefix reuse. GPU tests skip explicitly
when no CUDA device is available.

The benchmark reports wall-clock microseconds per decode token, including KV
I/O, host scheduling, final projection and logits D2H. Both modes use the same
async KV implementation and storage configuration. Each receives an untimed
warmup; five pairs alternate execution order, with 4 prefill tokens followed
by 24 timed decode tokens. Capture/instantiation and reset/prefill are excluded.
Latency is reported, with no noisy speedup assertion.

## Initial local measurement (2026-11-04)

WSL2, RTX 5060 Ti, CUDA 12.8, Release, SM 120. Deterministic synthetic FP32
checkpoint: dim=32, hidden_dim=64, 5 layers, 4 query heads, 2 KV heads,
vocab=48, context=40, 4 tokens/page, POSIX SSD callback, prefix cache disabled
for timing. The precision test separately enables prefix reuse and compares
all 48 logits across two requests up to the context limit.

| Pair | Eager TPOT (µs) | Graph TPOT (µs) |
| --- | ---: | ---: |
| 1 | 1808.93 | 1250.44 |
| 2 | 1739.15 | 1301.53 |
| 3 | 1791.69 | 1243.22 |
| 4 | 1824.75 | 1288.62 |
| 5 | 1703.65 | 1254.81 |
| Median | 1791.69 | 1254.81 |

Median ratio: 1.428× (29.96% lower TPOT). Maximum absolute logit error: 0.
Ten graph instances share 4096 bytes of private device workspace. CUDA graph
metadata and the two KV slots are additional allocations. These are synthetic
model regression/latency measurements, not evidence of pretrained-model,
physical-NVMe or vLLM throughput improvements.

Final native CTest regression: 14 passed (including all 7 graph tests),
4 SPDK cases skipped (SPDK disabled). CPU-only build including the C API and
preload library: 4 passed, 1 SPDK case skipped.
Compute Sanitizer could not instrument this WSL device: it reported
`Failed to initialize WDDM debugger interface` and `Device not supported`.
The tests executed successfully underneath the failed tool initialization,
but this is **not** a successful memcheck result. No Windows debugger settings
were changed.

CUDA captures launch arguments by value; see NVIDIA's
[dynamic graph parameter explanation](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/).
For general shared-pool lifetime constraints, see
[PyTorch CUDA graph memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#graph-memory-management).
This runner supplies a fixed allocation layout and serial stream ordering
directly, without depending on PyTorch's allocator.
