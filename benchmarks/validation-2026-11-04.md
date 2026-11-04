# CUDA prefill, KV pipeline, and allocator validation

Measured on 2026-11-04 with an RTX 5060 Ti under WSL2. These are **mock-weight
pipeline measurements, not real-model performance claims**: FP32 Llama, four
layers, dimension 128, four attention heads, two KV heads, vocabulary 256,
context 88, input 64 tokens, and eight generated tokens. Each cell is the median
of three independent processes, with seeds 42, 43, and 44 and one model warmup.

## Changes

- Prefill processes all unmatched prompt rows layer by layer. Projections use a
  16-by-16 tiled FP32 GEMM; RoPE, paged KV writes, and causal attention submit
  batches. Only the final hidden row receives the vocabulary projection and
  logits download. Single-token decode retains GEMV and optional CUDA Graphs.
- The async pipeline can retain a fixed GPU slot per layer when the configured
  budget covers rounded allocations. It skips per-token KV uploads/writebacks;
  prefix publication flushes dirty rows. Reset can still require prefix upload.
  Smaller budgets retain the two-slot Host/SSD streaming path.
- CPU, pinned, and CUDA pools use finer size classes with direct indexing.
  CUDA amortizes upstream allocations through splittable segments, bounded by
  explicit budgets. A failed oversized segment request retries the requested
  capacity. CPU/pinned pools retain their existing small-object thread caches.
- Host read/write counters and separate pinned staging capacity make the
  B-host path observable. Counters establish traffic, not measured overlap.

## End-to-end measurements

TTFT and TPOT are milliseconds; lower is better. TTFT starts with prepared token
IDs and excludes model initialization. TPOT averages the seven subsequent
decode tokens. `repeat` reuses a matching prompt prefix.

| Workload | Engine | Miss TTFT | Miss TPOT | Repeat TTFT | Repeat TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| A: GPU resident | MiniInfer sync | 2.767 | 1.117 | 1.851 | 1.126 |
| A: GPU resident | MiniInfer async | 1.355 | 1.022 | 1.347 | 0.742 |
| A: GPU resident | MiniInfer graph | 1.146 | 0.745 | 1.555 | 0.371 |
| A: GPU resident | vLLM | 11.921 | 1.967 | 9.975 | 2.066 |
| B-host | MiniInfer sync | 2.759 | 3.614 | 3.221 | 3.864 |
| B-host | MiniInfer async | 1.666 | 1.589 | 1.379 | 1.453 |
| B-host | MiniInfer graph | 1.971 | 0.990 | 1.213 | 1.288 |
| B-host | vLLM GPU baseline | 12.722 | 1.890 | 10.447 | 1.815 |
| B-nvme | MiniInfer sync | 11.693 | 11.180 | 8.880 | 10.588 |
| B-nvme | MiniInfer async | 20.610 | 5.403 | 3.710 | 4.291 |
| B-nvme | MiniInfer graph | 20.615 | 4.674 | 3.921 | 3.605 |
| B-nvme | vLLM GPU baseline | 11.053 | 1.958 | 9.871 | 1.969 |

The vLLM B rows do **not** perform equivalent Host/NVMe offload. The mock model,
short requests, and three repetitions do not establish production-model
superiority. The native before/after baseline uses saved pre-change executables
with the same paired model and traces:

| A miss | Previous TTFT | New TTFT | Previous TPOT | New TPOT |
| --- | ---: | ---: | ---: | ---: |
| sync | 53.698 | 2.767 | 0.984 | 1.117 |
| async | 78.717 | 1.355 | 1.302 | 1.022 |
| graph | 47.557 | 1.146 | 0.930 | 0.745 |

B-nvme async miss TPOT increased from 2.813 to 5.403 ms. Reduced prefill latency
does not establish an across-the-board decode improvement.

## Allocation measurements

Short trace: 16 live slots, 16 warmup operations, and 100 measured operations per
thread. Allocation times below are medians of per-process allocation means.

| Pool | Threads | Previous allocation us | New allocation us | Previous internal waste % | New internal waste % |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | 0.860 | 1.831 | 19.037 | 0.993 |
| CPU | 4 | 2.640 | 6.414 | 19.599 | 1.219 |
| CUDA | 1 | 16.063 | 0.526 | 19.037 | 0.993 |
| CUDA | 4 | 25.549 | 9.853 | 18.693 | 1.219 |
| Pinned | 1 | 65.264 | 190.559 | 19.037 | 0.993 |
| Pinned | 4 | 101.262 | 271.408 | 19.592 | 1.219 |

Finer classes and stricter reuse reduce waste but increase cold CPU/pinned pool
misses; pinned allocation has **not** reached a five-microsecond target. The
three single-thread CUDA process means range from 0.386 to 8.000 us, so the
0.526-us median is not a per-process or per-allocation latency bound.

Long CUDA trace: 1,000 warmup operations and 10,000 measured operations, one
thread and 16 slots, separately executed after the full suite:

| Seed | Allocation samples | Mean us | P99 us | Internal waste % | Measured upstream allocations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 5002 | 0.284 | 0.543 | 1.270 | 0 |
| 43 | 5001 | 0.342 | 0.476 | 1.418 | 1 |
| 44 | 5001 | 0.240 | 0.347 | 1.257 | 1 |

Internal waste concerns **active block capacity**. The short single-thread CUDA
trace reserves 18 MiB; its median requested/reserved fraction is only 4.735%.
Cached idle segments are reusable but consume memory. Consequently, a 99.007%
active-block utilization must not be presented as whole-pool utilization.

## Verification and reproduction

- 153 benchmark cells completed successfully; 72 inference records agreed with
  the new sync reference and independently with saved pre-change sync outputs.
- B-host native rows recorded both pinned host reads and writes. B-nvme native
  rows recorded SSD reads during prefill. No Nsight/CUPTI overlap trace was taken.
- CUDA CTest: 15 passed, four SPDK tests skipped because SPDK was not enabled.
- CPU-only CTest: two passed, one SPDK test skipped. Both configurations were
  rerun before publication without failures.
- Tests cover sequential-versus-batched logits, partial-page appends, prefix
  reuse, graph replay, rounded GPU budget fallback, no per-token Host traffic
  on the resident path, and allocator split/reuse/coalescing.

Use Python with compatible CUDA PyTorch/vLLM installed, CMake 3.24+, and CUDA
Toolkit on `PATH`, then run from the repository root:

```bash
PYTHON=/path/to/python bash benchmarks/run_benchmarks.sh \
  --quick --cuda on --repetitions 3
```

See [README.md](README.md) for exact metric definitions and environment controls.
The runner preserves commands, source hashes, a source snapshot, raw samples,
output token IDs, process dispersion, and failures under `benchmarks/results/`.
Generated models, binaries, raw result directories, and local environment files
are excluded from Git; this document preserves the publishable measurement
summary. FP32 batch attention still allocates a batch-by-head-by-context score
workspace, so long prompts remain subject to GPU memory limits.
