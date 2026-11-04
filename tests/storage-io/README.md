# Aligned page I/O

Requires CMake 3.24+, GoogleTest and liburing. CUDA and SPDK follow the runtime build options.

```sh
cmake -S . -B build-chunk-io -G Ninja \
  -DINFER_BUILD_NATIVE=OFF -DINFER_BUILD_VLLM=OFF \
  -DBUILD_TESTING=ON -DINFER_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-chunk-io -j8
ctest --test-dir build-chunk-io --output-on-failure
build-chunk-io/benchmark_storage_chunks uring /tmp
```

Enable SPDK with -DINFER_ENABLE_SPDK=ON and -DINFER_SPDK_ROOT=/path/to/sdk.
The fixture uses a bounded malloc bdev with PCI and hugepages disabled.

```sh
build-chunk-io/benchmark_storage_chunks spdk-malloc
```

Each benchmark alternates single-page and 64-page batch modes for six repetitions, with 2,048 pages of 4 KiB each. Both modes use ON_READ CRC32C verification. Randomized group traversal and within-group traversal share the same order. GB/s uses logical payload bytes, page IOPS counts logical pages, and device IOPS counts completed device operations. Write timing ends at I/O completion.

For a dedicated physical bdev, provide an SPDK JSON configuration, its explicit bdev name, an exclusive byte offset and a reactor mask. This writes the configured 32 MiB region.

```sh
build-chunk-io/benchmark_storage_chunks spdk /path/to/config.json BdevName OFFSET_BYTES 0x1
```

Tests cover aligned allocation and cache reuse, Fixed Buffer direct I/O across multiple submission waves, vectored headers, short-read retry to EOF, draining peer CQEs, chunk lookup, stale-replica overwrite, CRC error isolation, GPU page round trips, CUDA-registration fallback, and SPDK SGL payload-copy counters. CTest runs SPDK cases in separate processes because the runtime owns one SPDK application lifecycle per process.
