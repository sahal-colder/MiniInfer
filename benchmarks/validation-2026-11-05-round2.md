# Storage refactor validation — 2026-11-05

Baseline: `ac04788c1325e259ac485ec0ee801c3424842dc7`.

This refactor changes seven production files and two existing test files:

- KVOffloadCache shares batch writeback and migration result handling, uses its existing CPU LRU for residency queries, and no longer accesses StorageState or LogicalBlockRecord internals.
- TieredStorage owns the private cache queries and shares staging preparation, cleanup accounting, and batch read admission.
- SSDStorageBackend shares batch record locking and CRC reads, deriving physical addresses from the existing slot instead of retaining duplicate chunk addresses.
- AsyncKVPipeline replaces four execution flags with one phase, shares the D2H completion event, borrows its owner's callbacks, and avoids an idle worker in resident mode.

No new runtime classes, dependencies, public API, or record format changes were introduced. POSIX, O_DIRECT, io_uring, and SPDK code paths remain; SPDK was not exercised in this environment.

## Tests

- Repository Native/CUDA/io_uring tests: **25 passed, 4 SPDK tests skipped**.
- CPU Debug ASan/UBSan with leak detection: **11 passed, 1 SPDK test skipped**, no sanitizer reports.
- Including the existing external validation fixtures: baseline and candidate both have **71 passed, 4 skipped, and the same 10 failures**. This broader suite is not green.
- Four added behavior tests pass against both implementations: transactional resize failure, batch read admission and lease release, resident download/retirement admission, and LRU placement/read accounting.
- Final model comparison: 40 processes / 80 requests produced identical tokens, prefix hits, I/O byte counts, graph replay counts, and memory budgets. Prefetch/DMA pool and staging budgets and chunk completion counts also match.

## Final paired measurements

Release `-O3`, CUDA 12.8, WSL2/ext4, RTX 5060 Ti, identical CPU affinity (0–3), five alternating process pairs. Values are medians of process-level sample means. These are cumulative baseline-to-final comparisons, not isolated attribution to individual modules.

| Workload | Before | After | Change |
|---|---:|---:|---:|
| 64 KiB batch write | 167.053 µs | 161.737 µs | -3.18% latency |
| 64 KiB batch read | 22.874 µs | 21.963 µs | -3.98% latency |
| 4 MiB batch write | 4713.022 µs | 4759.576 µs | +0.99% latency |
| 4 MiB batch read | 2852.595 µs | 2850.835 µs | -0.06% latency |
| 4 KiB chunk write | 0.939 GB/s | 0.915 GB/s | -2.56% throughput |
| 4 KiB chunk read | 0.903 GB/s | 0.896 GB/s | -0.78% throughput |
| Async B-host miss TPOT | 1.498 ms | 1.542 ms | +2.94% latency |
| Async B-nvme repeat TPOT | 2.806 ms | 2.706 ms | -3.57% latency |

Some medians increased. Prefetch/DMA and control microbenchmarks also varied between processes; these measurements do not establish zero regression across every scenario. Earlier unrestricted-affinity trials and all raw measurements remain in the locally ignored `benchmark-results/2026-11-05-round2/` directory.

The model fixture is the existing mock model. The benchmark's B-nvme label denotes the SSD code path here; temporary-file WSL measurements are not cold physical-NVMe or SPDK hardware evidence. No SPDK SDK/device was available for this validation.
