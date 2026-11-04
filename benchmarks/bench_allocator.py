#!/usr/bin/env python3
"""PyTorch tensor allocator proxy and the actual optional vLLM CuMem memory pool.

Python/Tensor dispatch is included: compare these rows separately from C++ API latency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import threading
import time


def random_value(state):
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    value = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return state, value ^ (value >> 31)


def main(args):
    import torch
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 77
    custom = args.backend == "vllm-cumem"
    if custom and args.threads != 1:
        print("vLLM CuMem contexts require --threads 1 in this harness")
        return 77
    if custom:
        from vllm.device_allocator.cumem import CuMemAllocator, cumem_available
        if not cumem_available:
            print("vLLM CuMem extension unavailable")
            return 77
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.empty_cache()
    gate = threading.Barrier(args.threads + 1)
    rows = [[] for _ in range(args.threads)]
    requested = [0] * args.threads

    def worker(thread):
        try:
            torch.cuda.set_device(0)
            context = CuMemAllocator.get_instance().use_memory_pool(tag="benchmark") if custom else nullcontext()
            with context:
                state = args.seed + thread
                live = [None] * args.slots

                def operation(step, measured):
                    nonlocal state
                    state, value = random_value(state)
                    slot = (value >> 32) % args.slots
                    if live[slot] is not None:
                        size = live[slot].numel()
                        start = time.perf_counter_ns()
                        live[slot] = None
                        elapsed = (time.perf_counter_ns() - start) / 1000
                        if measured:
                            rows[thread].append((thread, step, "release", size, elapsed))
                        return
                    size = (256 << (value % 14)) + ((value >> 8) % 16) * 256
                    start = time.perf_counter_ns()
                    live[slot] = torch.empty(size, dtype=torch.uint8, device="cuda")
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    if measured:
                        rows[thread].append((thread, step, "allocate", size, elapsed))

                for step in range(args.warmup):
                    operation(step, False)
                gate.wait()
                gate.wait()
                for step in range(args.steps):
                    operation(step, True)
                requested[thread] = sum(tensor.numel() for tensor in live if tensor is not None)
                gate.wait()
                gate.wait()
                live.clear()
        except BaseException:
            gate.abort()
            raise

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(worker, thread) for thread in range(args.threads)]
        try:
            gate.wait()
            before = torch.cuda.memory_stats()
            start = time.perf_counter()
            gate.wait()
            gate.wait()
            wall = time.perf_counter() - start
            after = torch.cuda.memory_stats()
            gate.wait()
        except threading.BrokenBarrierError:
            for future in futures:
                future.result()
            raise
        for future in futures:
            future.result()
    allocations = sum(row[2] == "allocate" for batch in rows for row in batch)
    upstream = after.get("num_device_alloc", 0) - before.get("num_device_alloc", 0)
    # Custom MemPool counters are not guaranteed in the default allocator's stats.
    metadata = dict(backend=args.backend, wall_seconds=wall, active_requested_bytes=sum(requested),
                    active_capacity_bytes=None if custom else after["active_bytes.all.current"],
                    reserved_bytes=None if custom else after["reserved_bytes.all.current"],
                    upstream_allocations=None if custom or "num_device_alloc" not in after else upstream,
                    reuse_count=None if custom or "num_device_alloc" not in after else allocations - upstream,
                    torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                    allocator_backend=torch.cuda.memory.get_allocator_backend(),
                    timing_scope="Python Tensor dispatch; Python threads are constrained by the GIL")
    Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2))
    with Path(str(args.output) + ".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["thread", "step", "operation", "bytes", "latency_us"])
        writer.writerows(row for batch in rows for row in batch)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch-proxy", "vllm-cumem"], default="torch-proxy")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.slots, args.steps) < 1 or args.warmup < 0:
        parser.error("threads/slots/steps must be positive, warmup nonnegative")
    try:
        raise SystemExit(main(args))
    except ModuleNotFoundError as error:
        print(f"Dependency unavailable: {error}")
        raise SystemExit(77)
