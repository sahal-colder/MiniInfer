# Storage simplification validation

Measured on 2026-11-05 against `b9dcd6912a8c6b55d9ad67e24dced0b562ca72bd`.
This change shares KV demotion completion bookkeeping, removes the redundant
SSD descriptor alias, and shares streaming/resident/Graph KV address calculation.
Public headers, configuration, C exports, record formats, and benchmark programs
are unchanged. A CPU pool cleanup candidate was evaluated and reverted.

## Functional checks

- Published tests plus four new characterization cases: 21 passed, four SPDK
  cases skipped. The new cases also pass against the original implementation.
  Each final characterization case passed 20 repetitions.
- CPU-only ASan and UBSan: ten passed, one SPDK case skipped. TSan compiled but
  aborted before tests with WSL's `unexpected memory mapping`; it is unverified.
- Real host bridge and GPU/pinned/CPU/SSD round trips passed before and after,
  including DMA reuse off/on and staging capacities of 2, 6, and 16 slots.
- A separately preserved historical suite plus current/new tests: 105 passed,
  12 failed, four skipped on both sides, with identical per-test status. The
  failures include historical pool strategy/count assertions, optional batch
  promotion/failure expectations, a device failure expectation, and CPU/CUDA
  checkpoint recovery. These were not fixed or relabeled as passing.
- Historical Python integration suite: 86 passed, the same 14 failed, one
  skipped, and four subtests passed on each side. Failures include the adapter's
  vLLM 0.31 requirement under installed vLLM 0.32, budget/CRC expectations,
  DMA coordination/queue cases, and missing historical benchmark utilities.
- Output token IDs, prefix hits, pool footprints, and Graph replay counts match
  across all 45 paired model processes and their 90 request phases per side.

The new syscall-injection tests check multiple accepted failing demotions,
preserved sources, retry, multi-wave migration, data, and capacity. They do not
directly inject submission failure, committed-cleanup failure, or competing
exception priorities. Independent read-only review found no actionable issue
in the three production changes.

## Performance measurements

Environment: WSL2 Linux 6.18.54.1, Core Ultra 7 265KF (16 visible CPUs), 9.7 GiB
RAM, RTX 5060 Ti 16 GiB, CUDA 12.8.93, GCC 11.4, CMake 3.31.6, liburing 2.1,
Release `-O3 -DNDEBUG`, SM 120. CUDA, Native, vLLM, C ABI, and preload targets
built successfully. SPDK SDK, physical NVMe, and cuFile were not validated.

The final matrix alternates frozen before/after binaries in five independent
processes per side for each of 30 workloads: 300/300 successful processes.
Even rounds run before then after; odd rounds reverse that order. The initial
pre-change baseline and all intermediate and follow-up runs are retained locally.

The [complete comparison CSV](validation-2026-11-05-refactor.csv) contains 279
metrics with medians, five-process ranges, and sample counts. Positive latency
changes are slower; positive throughput changes are faster. A metric median is
the median of per-process statistics, including per-process P50/P95/P99 values.

| KV payload / overlap | Before batch-write mean us | After batch-write mean us | Change |
| --- | ---: | ---: | ---: |
| 64 KiB / off | 185.3 | 191.6 | +3.37% |
| 64 KiB / on | 123.2 | 104.1 | -15.47% |
| 4 MiB / off | 5179 | 5377 | +3.81% |
| 4 MiB / on | 4487 | 4401 | -1.91% |

KV workload: 64 logical / 16 CPU slots, eight records per batch, concurrency
two, CRC on read, two warmup rounds and 96 measured batches per process. Write
timing includes required demotions and completed I/O; the subsequent read
mainly exercises CPU hits. Chunk read/write medians changed from 0.833/0.850
to 0.863/0.890 GB/s. Neither observation establishes a stable improvement.

Other workloads use the unchanged benchmark programs:

- Allocators: CPU/pinned/CUDA pools, one/four threads, 20,000 steps, 2,000
  warmup steps, 16 slots, seed 42.
- File I/O: POSIX buffered/io_uring, 64 KiB/4 MiB blocks, depth 1/32, 64 MiB
  datasets, random access, three rounds.
- Native model: fixed mock FP32 Llama (dimension 128, hidden 256, four layers,
  four query/two KV heads, vocabulary 256, context 88), 64 input/16 output
  tokens, two warmups, 16-token pages; sync/async/Graph across A/B-host/B-nvme.
  Host budgets are 2,752,512 bytes for A/B-host and 655,360 for B-nvme; sync
  B-nvme uses a 393,216-byte GPU budget. Each process measures miss/repeat.
- Original chunk I/O, prefetch/DMA round-trip, and storage control benchmarks.
  DMA timing includes CPU→GPU→CPU completion and source recovery, not one-way
  migration. Prefetch timing includes the subsequent read.

TTFT includes prompt work and the first token; pure prefill tokens/s was not
separately measured. Model results use mock weights and do not establish real
model or vLLM serving performance. `B-nvme` is a workload name: files here use
warm WSL ext4 `/tmp`, without `fsync` or physical-disk isolation. These are not
cold physical-SSD or durable-I/O measurements.

Noise is substantial. The final allocator and raw-storage executables are each
byte-identical across sides, yet show double-digit differences in some cells.
AsyncKV disassembly instructions also match. Follow-up five-pair checks of the
three A model variants and 64 KiB non-overlapped KV did not reproduce the larger
regressions consistently: Graph A miss TTFT changed from +37.91% in the final
matrix to +1.86% in confirmation, while sync A repeat TPOT changed from +17.76%
to -12.22%. All original matrix values remain in the published CSV.

No stable regression attributable to these changes was identified in these
runs. This is not a statistical equivalence or noninferiority result, a P99
guarantee, or evidence for other hardware. The CPU cleanup candidate was reverted
after its inconclusive pinned-allocation performance gate.

Full local evidence is retained under `benchmark-results/2026-11-05-refactor/`:
commands, per-process samples, source hashes and snapshots, fixtures, CMake
settings, test logs, sanitizer output, intermediate measurements, and the
complete Chinese report under `docs/refactoring-2026-11-05.md`. Generated models,
binaries, archives, and raw logs remain excluded by the repository's existing
publication rules.
