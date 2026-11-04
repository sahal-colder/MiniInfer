#!/usr/bin/env python3
"""Compile, run isolated/repeated benchmark cells, retain evidence, and render a report."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import traceback

from prepare_model import prepare, sha256

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks"


def env_int(name, default):
    return int(os.environ.get(name, default))


def numbers(name, default):
    return [int(value) for value in os.environ.get(name, default).split(",")]


def capture(command, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    return result.stdout


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, quantile):
    # Nearest-rank empirical quantile. Keep all operation/token samples on disk.
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)]


def monitor_gpu(stop, path, identifier, interval):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["monotonic_s", "used_MiB", "util_percent", "temperature_C", "power_W", "sm_MHz"])
        while not stop.is_set():
            result = subprocess.run([
                "nvidia-smi", "-i", identifier,
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=10)
            if result.returncode:
                return
            writer.writerow([time.monotonic(), *[part.strip() for part in result.stdout.strip().split(",")]])
            stream.flush()
            stop.wait(interval)


def execute(command, log, timeout, monitor=None):
    stop = threading.Event()
    thread = None
    if monitor and shutil.which("nvidia-smi"):
        thread = threading.Thread(target=monitor_gpu, args=(stop, *monitor), daemon=True)
        thread.start()
    started = time.perf_counter()
    with log.open("w") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    stop.set()
    if thread:
        thread.join(timeout=12)
    return code, time.perf_counter() - started


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)])


def report(output, status, allocator, storage, inference, mock):
    write_csv(output / "status.csv", status)
    write_csv(output / "allocator_summary.csv", allocator)
    write_csv(output / "storage.csv", storage)
    write_csv(output / "end_to_end.csv", inference)
    # Aggregate within-process I/O rounds before describing independent process dispersion.
    metrics = defaultdict(lambda: defaultdict(list))
    for row in allocator:
        for metric in ["allocate_mean_us", "allocate_p99_us", "release_mean_us", "release_p99_us"]:
            if row[metric] is not None:
                metrics[("allocator", row["backend"], str(row["threads"]), metric)][row["cell"]].append(row[metric])
    for row in storage:
        for metric in ["GB_s", "IOPS"]:
            metrics[("storage", row["backend"], str(row["block_bytes"]), str(row["requested_depth"]),
                     row["operation"], row["pattern"], metric)][row["cell"]].append(float(row[metric]))
    for row in inference:
        for metric in ["ttft_ms", "tpot_ms", "device_peak_vram_mib"]:
            if row.get(metric) is not None:
                metrics[("e2e", row["engine"], row["workload"], str(row["prompt_length"]),
                         row["phase"], metric)][row["cell"]].append(row[metric])
    statistics_rows = []
    for group, processes in sorted(metrics.items()):
        values = [statistics.median(samples) for samples in processes.values()]
        statistics_rows.append(dict(group="/".join(group), n_processes=len(values), mean=statistics.mean(values),
                                    median=statistics.median(values), minimum=min(values), maximum=max(values),
                                    stddev=statistics.stdev(values) if len(values) > 1 else None))
    write_csv(output / "process_statistics.csv", statistics_rows)
    lines = ["# MiniInfer benchmark results", "",
             "**MOCK WEIGHTS / PIPELINE VALIDATION ONLY**" if mock and inference else
             "**Real weights; inspect correctness and workload evidence before quoting results.**" if inference else
             "No inference measurements in this report.",
             "", "Each row below is the median across independent process repetitions. Full cells and logs remain in raw/.",
             "Failures/skips are retained. No automatic speedup claim is made.", "", "## Allocator", "",
             "C++ rows time native APIs. Python rows include Tensor/Python dispatch and the GIL; do not interpret their ratio as allocator-only speedup.",
             "Fragmentation column is active internal capacity waste. Cached reserve slack is reported separately and is reusable capacity, not proven external fragmentation.", ""]
    groups = defaultdict(list)
    for row in allocator:
        groups[(row["backend"], row["threads"])].append(row)
    med = lambda rows, key: statistics.median(row[key] for row in rows if row.get(key) is not None)
    value = lambda rows, key: f"{med(rows, key):.3f}" if any(row.get(key) is not None for row in rows) else "N/A"
    lines.append(table(["Backend", "Threads", "n", "Alloc mean µs", "Alloc P99 µs", "Free mean µs", "Free P99 µs", "Internal waste %", "Reserve slack %", "Reuse hit %"], [
        [backend, threads, len(rows), *(value(rows, key) for key in ["allocate_mean_us", "allocate_p99_us", "release_mean_us", "release_p99_us", "internal_waste_pct", "reserve_slack_pct", "reuse_hit_pct"])]
        for (backend, threads), rows in sorted(groups.items())]))
    lines += ["", "## Storage", "", "GB/s uses decimal 10^9. Timings include completed I/O, exclude preparation/verification; no timed durability flush. Buffered reads are warm page-cache traffic. Direct I/O still includes device/host caches.", ""]
    groups.clear()
    for row in storage:
        groups[(row["backend"], int(row["block_bytes"]), int(row["requested_depth"]), row["operation"], row["pattern"])].append(row)
    lines.append(table(["Backend", "Block B", "Requested QD", "Effective QD", "Operation", "Pattern", "Samples", "GB/s median", "IOPS median"], [
        [backend, block, depth, rows[0]["effective_depth"], operation, pattern, len(rows),
         f"{statistics.median(float(row['GB_s']) for row in rows):.3f}",
         f"{statistics.median(float(row['IOPS']) for row in rows):.0f}"]
        for (backend, block, depth, operation, pattern), rows in sorted(groups.items())]))
    lines += ["", "## End to end", "", "TTFT starts at local request submission (token IDs already prepared); TPOT excludes the first output token. VRAM is the sampled whole-device high-water mark across initialization + requests, including other desktop processes; it is a lower bound on the true peak, not allocator/process peak.", "",
              "vLLM rows for B are standard GPU baselines, without an equivalent host/NVMe offload path. Prefix hits and prefill SSD reads are explicit evidence columns.", ""]
    groups.clear()
    for row in inference:
        groups[(row["engine"], row["workload"], row["prompt_length"], row["phase"])].append(row)
    lines.append(table(["Engine", "Workload", "Input", "Phase", "n", "TTFT ms", "TPOT ms", "Device VRAM MiB", "Prefix hits", "Prefill SSD B", "Host read B", "Host write B", "Evidence", "Tokens agree"], [
        [engine, workload, prompt, phase, len(rows), value(rows, "ttft_ms"), value(rows, "tpot_ms"),
         value(rows, "device_peak_vram_mib"), value(rows, "prefix_hit_tokens"), value(rows, "prefill_ssd_read_bytes"), value(rows, "host_read_bytes"), value(rows, "host_written_bytes"),
         ", ".join(sorted({row["evidence"] for row in rows})),
         ", ".join(sorted({str(row.get("tokens_match_sync", "unavailable")) for row in rows}))]
        for (engine, workload, prompt, phase), rows in sorted(groups.items())]))
    lines += ["", "## Status", "", table(["Cell", "Status", "Reason / log"], [
        [row["cell"], row["status"], row.get("reason", "").replace("|", "/").replace("\n", " ")[:200]] for row in status]), "",
        "See environment.json, model_pair.json, commands.jsonl and build.log for provenance. P99 with few samples is descriptive only. Compare paired process results and dispersion, not only this median table.", ""]
    (output / "summary.md").write_text("\n".join(lines))


def run(args):
    output = args.results.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    status, allocator, storage, inference = [], [], [], []
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Preserve user-specified values, and record every relevant override below.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    cuda = args.cuda == "on" or (args.cuda == "auto" and shutil.which("nvcc") is not None and shutil.which("nvidia-smi") is not None)
    manifest = dict(mock=True)
    environment = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                       python=sys.version, executable=sys.executable, platform=platform.platform(),
                       cpu=platform.processor(), cuda=cuda, settings=vars(args).copy(),
                       cpu_affinity=sorted(os.sched_getaffinity(0)),
                       environment={key: value for key, value in os.environ.items()
                                    if key.startswith(("BENCH_", "CUDA_", "VLLM_", "PYTORCH_", "OMP_", "SPDK_", "ALLOW_SPDK_"))},
                       git_head=capture(["git", "rev-parse", "HEAD"]),
                       git_submodules=capture(["git", "submodule", "status"]),
                       git_status=capture(["git", "status", "--short"]),
                       git_diff_sha256=hashlib.sha256(capture(["git", "diff", "HEAD"]).encode()).hexdigest())
    for name, command in [("gpu", ["nvidia-smi", "-q"]), ("compiler", ["c++", "--version"]),
                          ("cuda_toolkit", ["nvcc", "--version"]),
                          ("filesystem", ["findmnt", "-T", str(args.io_directory.resolve())]),
                          ("block_devices", ["lsblk", "-o", "NAME,TYPE,SIZE,MODEL,TRAN,MOUNTPOINTS"])]:
        environment[name] = capture(command) if shutil.which(command[0]) else "unavailable"
    sources = [path for directory in ["src", "include", "cmake", "app", "integrations"]
               for path in (ROOT / directory).rglob("*")
               if path.is_file() and path.suffix in [".c", ".cpp", ".cu", ".h", ".cuh", ".cmake", ".in", ".py", ".txt"]]
    sources += [path for path in HERE.iterdir() if path.is_file()] + [ROOT / "CMakeLists.txt", ROOT / ".gitignore"]
    environment["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    (output / "environment.json").write_text(json.dumps(environment, default=str, indent=2))
    # Save the worktree diff because this repository can contain ongoing product optimizations.
    (output / "worktree.patch").write_text(capture(["git", "diff", "HEAD"]))
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    print(f"Results: {output}", flush=True)
    try:
        if args.spdk_config:
            shutil.copyfile(args.spdk_config, output / "spdk_config.json")
        if "e2e" in args.suites:
            pair = args.model_pair or prepare(output / "mock_model", True, max(args.prompts) + args.generate + 16, None)
            manifest = json.loads(Path(pair).read_text())
            if sha256(manifest["native"]) != manifest["native_sha256"]:
                raise ValueError("Native model hash changed since pairing")
            for path, digest in manifest["hf_sha256"].items():
                if sha256(path) != digest:
                    raise ValueError(f"HF model hash changed: {path}")
            (output / "model_pair.json").write_text(json.dumps(manifest, indent=2))
        build = args.build_dir or HERE / "build" / ("cuda" if cuda else "cpu")
        configure = ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     f"-DINFER_ENABLE_CUDA={'ON' if cuda else 'OFF'}", "-DINFER_BUILD_RUNTIME=ON",
                     f"-DINFER_BUILD_NATIVE={'ON' if cuda and 'e2e' in args.suites else 'OFF'}",
                     "-DINFER_BUILD_VLLM=OFF", "-DINFER_BUILD_C_API=OFF", "-DINFER_BUILD_PRELOAD=OFF",
                     "-DINFER_BUILD_BENCHMARKS=ON", "-DBUILD_TESTING=OFF",
                     f"-DINFER_ENABLE_SPDK={'ON' if args.spdk_config else 'OFF'}"]
        if cuda and "e2e" in args.suites:
            configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={os.environ.get('BENCH_CUDA_ARCH', 'native')}")
        if args.spdk_config and os.environ.get("SPDK_ROOT"):
            configure.append(f"-DINFER_SPDK_ROOT={os.environ['SPDK_ROOT']}")
        for index, command in enumerate([configure, ["cmake", "--build", str(build), "--parallel", str(args.jobs),
                                                    "--target", "bench_allocator", "bench_storage",
                                                    *(["bench_model"] if cuda and "e2e" in args.suites else [])]]):
            print("Configuring" if not index else "Compiling benchmark targets", flush=True)
            code, _ = execute(command, output / ("configure.log" if not index else "build.log"), args.build_timeout)
            if code:
                status.append(dict(cell="build", status="failed", reason=f"exit {code}; see configure.log/build.log"))
                return 1
        shutil.copyfile(build / "CMakeCache.txt", output / "CMakeCache.txt")
        programs = build / "benchmarks"
        jobs = []
        def add(suite, name, command, **metadata):
            jobs.append(dict(suite=suite, name=name, command=command, **metadata))
        if "allocator" in args.suites:
            for threads in args.threads:
                backends = ["cpu-raw", "cpu-pool"] + (["cuda-raw", "cuda-pool", "pinned-raw", "pinned-pool", "torch-proxy"] if cuda else [])
                if cuda and threads == 1:
                    backends.append("vllm-cumem")
                for backend in backends:
                    command = ([sys.executable, str(HERE / "bench_allocator.py")] if backend in ["torch-proxy", "vllm-cumem"]
                               else [str(programs / "bench_allocator")])
                    add("allocator", f"allocator-{backend}-t{threads}", command + ["--backend", backend, "--threads", str(threads),
                        "--slots", str(args.slots), "--steps", str(args.steps), "--warmup", str(args.allocator_warmup)], backend=backend, threads=threads)
            if not cuda:
                status.append(dict(cell="allocator-cuda", status="skipped", reason="CPU build; CUDA and vLLM allocators unavailable"))
        if "storage" in args.suites:
            for block in args.blocks:
                for depth in args.depths:
                    for backend in ["posix-buffered", "posix-direct", "uring", *(["spdk"] if args.spdk_config else [])]:
                        command = [str(programs / "bench_storage"), "--backend", backend, "--directory", str(args.io_directory),
                                   "--bytes", str(args.storage_bytes), "--block", str(block), "--depth", str(depth),
                                   "--rounds", str(args.storage_rounds), "--pattern", args.pattern]
                        if backend == "spdk":
                            command += ["--spdk-config", str(args.spdk_config), "--spdk-bdev", args.spdk_bdev,
                                        "--spdk-offset", str(args.spdk_offset), "--allow-spdk-write", args.allow_spdk_write,
                                        "--spdk-no-pci", str(env_int("SPDK_NO_PCI", 0)), "--spdk-no-huge", str(env_int("SPDK_NO_HUGE", 0)),
                                        "--spdk-memory-mib", str(env_int("SPDK_MEMORY_MIB", 512))]
                        add("storage", f"storage-{backend}-b{block}-q{depth}", command)
            if not args.spdk_config:
                status.append(dict(cell="storage-spdk", status="skipped", reason="No explicitly owned SPDK bdev/config/offset supplied"))
        if "e2e" in args.suites and not cuda:
            status.append(dict(cell="e2e", status="skipped", reason="Native inference requires a CUDA build"))
        if "e2e" in args.suites and cuda:
            config = manifest["config"]
            for length in args.prompts:
                if length + args.generate > config["seq_len"]:
                    raise ValueError("Prompt + generation exceed paired model context")
                prompt = output / f"prompt-{length}.txt"
                prompt.write_text(" ".join(str((index * 17 + 3) % abs(config["vocab_size"])) for index in range(length)) + "\n")
                page_count = math.ceil(config["seq_len"] / args.page_tokens)
                page_bytes = 65536
                if args.page_tokens * 2 * 4 * config["kv_head_num"] * (config["dim"] // config["head_num"]) > page_bytes:
                    raise ValueError("BENCH_PAGE_TOKENS exceeds the native 64KiB KV page capacity")
                for workload in args.workloads:
                    host = ((config["layer_num"] + 2) * (page_count + 1) * page_bytes if workload != "B-nvme"
                            else config["layer_num"] * page_bytes)
                    for engine in ["sync", "async", "graph", "vllm"]:
                        ballast = args.ballast_mib if workload != "A" else 0
                        if engine == "vllm":
                            command = [sys.executable, str(HERE / "bench_end_to_end.py"), "--model", manifest["hf"],
                                       "--context", str(config["seq_len"]), "--vocab", str(abs(config["vocab_size"])),
                                       "--gpu-memory-utilization", str(args.vllm_memory),
                                       "--kv-cache-bytes", str(args.vllm_kv_bytes or (16 * 1024 * 1024 if manifest["mock"] else 0))]
                            if args.vllm_eager:
                                command += ["--eager"]
                        if engine != "vllm":
                            command = [str(programs / "bench_model"), "--model", manifest["native"], "--variant", engine,
                                       "--workload", workload, "--page-tokens", str(args.page_tokens),
                                       "--host-bytes", str(host), "--directory", str(args.io_directory), "--io", args.model_io]
                            if args.model_io == "spdk":
                                command += ["--spdk-config", str(args.spdk_config), "--spdk-bdev", args.spdk_bdev,
                                            "--spdk-offset", str(args.spdk_offset), "--spdk-capacity", str(args.spdk_capacity),
                                            "--allow-spdk-write", args.allow_spdk_write,
                                            "--spdk-no-pci", str(env_int("SPDK_NO_PCI", 0)),
                                            "--spdk-no-huge", str(env_int("SPDK_NO_HUGE", 0)),
                                            "--spdk-memory-mib", str(env_int("SPDK_MEMORY_MIB", 512))]
                            # Sync must hold a complete layer's maximum context in GPU memory.
                            gpu = args.gpu_bytes or (page_bytes * page_count if workload == "B-nvme" and engine == "sync" else 0)
                            if gpu:
                                command += ["--gpu-bytes", str(gpu)]
                        add("e2e", f"e2e-{engine}-{workload}-p{length}", command + ["--prompt-file", str(prompt),
                            "--generate", str(args.generate), "--warmup", str(args.model_warmup), "--ballast-mib", str(ballast)],
                            engine=engine, workload=workload, prompt_length=length)
        rng = random.Random(args.seed)
        commands = output / "commands.jsonl"
        for repetition in range(args.repetitions):
            order = list(jobs)
            rng.shuffle(order)
            for job in order:
                cell = f"r{repetition}-{job['name']}"
                print(f"[{repetition + 1}/{args.repetitions}] {job['name']}", flush=True)
                base = output / "raw" / cell
                # Same seed for every paired backend in a repetition; seed changes between processes.
                command = job["command"] + ["--seed", str(args.seed + repetition), "--output", str(base)]
                with commands.open("a") as stream:
                    stream.write(json.dumps(dict(cell=cell, command=command, metadata=job)) + "\n")
                code, wall = execute(command, base.with_suffix(".log"), args.timeout,
                                     (base.with_suffix(".gpu.csv"), args.gpu_id, args.sample_interval) if job["suite"] == "e2e" else None)
                row = dict(cell=cell, repetition=repetition, status="ok" if not code else "skipped" if code == 77 else "failed",
                           exit_code=code, wall_seconds=wall, reason="" if not code else f"exit {code}; raw/{cell}.log")
                status.append(row)
                if code:
                    continue
                if job["suite"] == "allocator":
                    metadata = json.loads(base.with_suffix(".json").read_text())
                    with base.with_suffix(".csv").open() as stream:
                        samples = list(csv.DictReader(stream))
                    summary = dict(cell=cell, repetition=repetition, backend=job["backend"], threads=job["threads"], **{key: value for key, value in metadata.items() if key != "backend"})
                    for operation in ["allocate", "release"]:
                        values = [float(sample["latency_us"]) for sample in samples if sample["operation"] == operation]
                        summary[f"{operation}_mean_us"] = statistics.mean(values) if values else None
                        summary[f"{operation}_p99_us"] = percentile(values, .99) if values else None
                    allocated = sum(sample["operation"] == "allocate" for sample in samples)
                    active, reserved, logical = (metadata[key] for key in ["active_capacity_bytes", "reserved_bytes", "active_requested_bytes"])
                    summary["internal_waste_pct"] = 100 * (active - logical) / active if active else None
                    summary["reserve_slack_pct"] = 100 * (reserved - logical) / reserved if reserved else None
                    summary["reuse_hit_pct"] = 100 * metadata["reuse_count"] / allocated if allocated and metadata["reuse_count"] is not None else None
                    allocator.append(summary)
                if job["suite"] == "storage":
                    with base.with_suffix(".csv").open() as stream:
                        storage.extend(dict(cell=cell, repetition=repetition, **sample) for sample in csv.DictReader(stream))
                if job["suite"] == "e2e":
                    data = json.loads(base.with_suffix(".json").read_text())
                    peak = None
                    if base.with_suffix(".gpu.csv").exists():
                        with base.with_suffix(".gpu.csv").open() as stream:
                            values = [float(sample["used_MiB"]) for sample in csv.DictReader(stream) if sample["used_MiB"].replace(".", "", 1).isdigit()]
                        peak = max(values) if values else None
                    for sample in data["requests"]:
                        evidence = "verified" if job["workload"] == "A" else "host-path-unverified"
                        if job["workload"] == "A" and sample["phase"] == "repeat" and not sample.get("prefix_hit_tokens"):
                            evidence = "prefix-hit-unverified"
                        if job["workload"] == "B-host" and sample.get("host_read_bytes", 0) and sample.get("host_written_bytes", 0):
                            evidence = "pinned-host-read-write"
                        if job["workload"] == "B-nvme":
                            evidence = "ssd-prefill-read" if sample.get("prefill_ssd_read_bytes", 0) else "ssd-prefill-unverified"
                        if job["engine"] == "vllm" and job["workload"] != "A":
                            evidence = "standard-gpu-baseline-only"
                        if job["engine"] == "graph" and not sample.get("graph_replays"):
                            evidence = "graph-replay-unverified"
                        inference.append(dict(cell=cell, repetition=repetition, engine=job["engine"], workload=job["workload"],
                                              prompt_length=job["prompt_length"], device_peak_vram_mib=peak, evidence=evidence,
                                              decode_tokens_s=1000 / sample["tpot_ms"],
                                              output_tokens_s=1000 * sample["output_tokens"] / sample["request_ms"],
                                              **{key: value for key, value in sample.items() if key not in ["tokens", "decode_samples_ms"]},
                                              tokens=sample["tokens"]))
                # Keep reviewable partial results even if a subsequent cell fails.
                report(output, status, allocator, storage, inference, manifest["mock"])
        reference = {(row["repetition"], row["workload"], row["prompt_length"], row["phase"]): row["tokens"]
                     for row in inference if row["engine"] == "sync"}
        for row in inference:
            tokens = reference.get((row["repetition"], row["workload"], row["prompt_length"], row["phase"]))
            row["tokens_match_sync"] = row["tokens"] == tokens if tokens is not None else "unavailable"
            if tokens is not None and row["tokens"] != tokens:
                status.append(dict(cell=row["cell"], status="correctness-failed", reason=f"{row['phase']}: greedy tokens differ from sync"))
        return 1 if any(row["status"] in ["failed", "correctness-failed"] for row in status) else 0
    except Exception as error:
        status.append(dict(cell="runner", status="failed", reason=str(error)))
        (output / "runner_error.log").write_text(traceback.format_exc())
        print(f"Runner failed: {error}", file=sys.stderr)
        return 1
    finally:
        report(output, status, allocator, storage, inference, manifest["mock"])
        print(f"Report: {output / 'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smoke only: small prompts/dataset, one process repetition")
    parser.add_argument("--suites", nargs="+", choices=["allocator", "storage", "e2e"], default=["allocator", "storage", "e2e"])
    parser.add_argument("--cuda", choices=["auto", "on", "off"], default=os.environ.get("BENCH_CUDA", "auto"))
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("BENCH_RESULTS", str(HERE / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}")))))
    parser.add_argument("--build-dir", type=Path, default=Path(os.environ["BENCH_BUILD_DIR"]) if "BENCH_BUILD_DIR" in os.environ else None)
    parser.add_argument("--model-pair", type=Path, default=Path(os.environ["BENCH_MODEL_PAIR"]) if "BENCH_MODEL_PAIR" in os.environ else None)
    parser.add_argument("--io-directory", type=Path, default=Path(os.environ.get("BENCH_IO_DIR", "/tmp")))
    parser.add_argument("--model-io", choices=["posix", "uring", "spdk"], default=os.environ.get("BENCH_MODEL_IO", "uring"))
    parser.add_argument("--repetitions", type=int, default=env_int("BENCH_REPETITIONS", 5))
    parser.add_argument("--seed", type=int, default=env_int("BENCH_SEED", 42))
    parser.add_argument("--jobs", type=int, default=env_int("BENCH_JOBS", min(8, os.cpu_count() or 1)))
    parser.add_argument("--timeout", type=int, default=env_int("BENCH_TIMEOUT", 1800))
    parser.add_argument("--build-timeout", type=int, default=env_int("BENCH_BUILD_TIMEOUT", 1200))
    parser.add_argument("--gpu-id", default=os.environ.get("BENCH_GPU_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    parser.add_argument("--sample-interval", type=float, default=float(os.environ.get("BENCH_GPU_SAMPLE_SECONDS", ".1")))
    parser.add_argument("--spdk-config", type=Path, default=Path(os.environ["SPDK_JSON"]) if "SPDK_JSON" in os.environ else None)
    parser.add_argument("--spdk-bdev", default=os.environ.get("SPDK_BDEV", ""))
    parser.add_argument("--spdk-offset", type=int, default=env_int("SPDK_OFFSET_BYTES", -1))
    parser.add_argument("--spdk-capacity", type=int, default=env_int("SPDK_CAPACITY_BYTES", 0))
    parser.add_argument("--allow-spdk-write", default=os.environ.get("ALLOW_SPDK_WRITE", "NO"))
    parser.add_argument("--vllm-eager", action="store_true", default=os.environ.get("BENCH_VLLM_EAGER") == "1")
    args = parser.parse_args()
    args.threads = numbers("BENCH_THREADS", "1,4")
    args.slots = env_int("BENCH_SLOTS", 16)
    args.steps = env_int("BENCH_ALLOCATOR_STEPS", 100 if args.quick else 10000)
    args.allocator_warmup = env_int("BENCH_ALLOCATOR_WARMUP", 16 if args.quick else 1000)
    args.blocks = numbers("BENCH_BLOCKS", "4096,65536,262144,1048576")
    args.depths = numbers("BENCH_DEPTHS", "1,4" if args.quick else "1,8,32")
    args.storage_bytes = env_int("BENCH_STORAGE_BYTES", (4 if args.quick else 256) * 1024 * 1024)
    args.storage_rounds = env_int("BENCH_STORAGE_ROUNDS", 1 if args.quick else 2)
    args.pattern = os.environ.get("BENCH_STORAGE_PATTERN", "random")
    args.prompts = numbers("BENCH_PROMPTS", "64" if args.quick else "2048,4096")
    args.generate = env_int("BENCH_GENERATE", 8 if args.quick else 128)
    args.model_warmup = env_int("BENCH_MODEL_WARMUP", 1)
    args.workloads = os.environ.get("BENCH_WORKLOADS", "A,B-host,B-nvme").split(",")
    args.page_tokens = env_int("BENCH_PAGE_TOKENS", 16)
    args.gpu_bytes = env_int("BENCH_GPU_BYTES", 0)
    args.ballast_mib = env_int("BENCH_BALLAST_MIB", 0)
    args.vllm_memory = float(os.environ.get("BENCH_VLLM_MEMORY", ".8"))
    args.vllm_kv_bytes = env_int("BENCH_VLLM_KV_BYTES", 0)
    if args.quick and "BENCH_REPETITIONS" not in os.environ and "--repetitions" not in sys.argv:
        args.repetitions = 1
    if args.spdk_config and (not args.spdk_bdev or args.spdk_offset < 0 or args.allow_spdk_write != "YES"):
        parser.error("SPDK requires SPDK_BDEV, an explicit nonnegative SPDK_OFFSET_BYTES, and ALLOW_SPDK_WRITE=YES")
    if args.model_io == "spdk" and (not args.spdk_config or args.spdk_capacity <= 0):
        parser.error("SPDK inference also requires SPDK_JSON and an explicit SPDK_CAPACITY_BYTES")
    if min(args.repetitions, args.steps, args.slots, args.page_tokens, args.jobs, args.storage_rounds,
           args.storage_bytes, *args.prompts, *args.threads, *args.depths, *args.blocks) < 1 or args.generate < 2:
        parser.error("Counts must be positive; output generation must contain at least two tokens")
    if (max(args.depths) > 128 or any(block % 4096 or args.storage_bytes % block for block in args.blocks)
            or args.allocator_warmup < 0 or args.model_warmup < 0 or args.ballast_mib < 0
            or args.gpu_bytes < 0 or args.vllm_kv_bytes < 0 or args.sample_interval <= 0
            or not 0 < args.vllm_memory < 1):
        parser.error("Use aligned block/dataset sizes, QD<=128, nonnegative budgets/warmup, positive sampling interval, 0<vLLM memory<1")
    if not args.gpu_id or "," in args.gpu_id:
        parser.error("Select exactly one physical GPU ID/UUID for BENCH_GPU_ID")
    if set(args.workloads) - {"A", "B-host", "B-nvme"} or args.pattern not in ["random", "sequential"]:
        parser.error("Invalid workloads/storage pattern")
    if not args.io_directory.is_dir():
        parser.error("--io-directory must exist")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
