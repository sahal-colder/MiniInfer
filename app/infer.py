#!/usr/bin/env python3
"""One startup configuration for the existing Native and vLLM integrations."""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "framework": "native", "storage-engine": "posix", "memory-mode": "pool",
    "cuda": True, "device": "cuda", "c-api": True, "preload": True,
    "build-dir": "build", "build-type": "Release", "jobs": 6, "cmake": "cmake",
    "model": "", "tokenizer": "", "prompt": "Once upon a time", "max-tokens": 64,
    "temperature": 0.0, "top-k": 0, "seed": 42,
    "max-model-len": 2048, "gpu-memory-utilization": 0.8, "enforce-eager": True,
    "cpu-cache-bytes": 256 << 20, "ssd-bytes": 32 << 30,
    "ssd-directory": "/tmp/miniinfer-kv", "disk-buffer-slots": 64,
    "io-depth": 8, "file-shards": 4, "write-concurrency": 4, "ssd-verify": "on_read",
    "spdk-sdk": "", "spdk-config": "", "spdk-bdev": "",
    "spdk-reactor-mask": "0x1", "spdk-mem-mib": 256,
    "spdk-iova-mode": "auto",
    "gpu-cache-bytes": 0, "cufile": False,
    "spdk-no-huge": False, "spdk-no-pci": False, "spdk-chunk-kib": 256,
}


def load_config(path, overrides, *, building=False):
    config = DEFAULTS.copy()
    for index, line in enumerate(path.read_text().splitlines(), 1):
        try:
            words = shlex.split(line, comments=True)
            if not words:
                continue
            if len(words) != 2:
                raise ValueError("expected key and one value")
            set_value(config, *words)
        except ValueError as error:
            raise ValueError(f"{path}:{index}: {error}") from error
    for key, value in overrides.items():
        if value is not None:
            set_value(config, key.replace("_", "-"), value)
    validate(config, building=building)
    for key in ("build-dir", "ssd-directory", "spdk-sdk", "spdk-config"):
        if config[key]:
            config[key] = str((path.parent / config[key]).resolve())
    for key in ("model", "tokenizer"):
        if config[key] and (
            config["framework"] == "native" or config[key].startswith(("/", "."))
            or (path.parent / config[key]).exists()
        ):
            config[key] = str((path.parent / config[key]).resolve())
    if "/" in config["cmake"]:
        config["cmake"] = str((path.parent / config["cmake"]).resolve())
    return config


def set_value(config, key, value):
    if key not in DEFAULTS:
        raise ValueError(f"unknown directive: {key}")
    default = DEFAULTS[key]
    if type(default) is bool:
        if value not in ("yes", "no"):
            raise ValueError(f"{key} expects yes or no")
        config[key] = value == "yes"
        return
    if type(default) is int:
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"{key} expects a nonnegative integer")
        config[key] = int(value)
        return
    if type(default) is float:
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"invalid {key}: {value}") from error
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite")
        config[key] = number
        return
    config[key] = value


def validate(config, *, building):
    for key, choices in (
        ("framework", ("native", "vllm", "storage")),
        ("storage-engine", ("posix", "uring", "auto", "spdk")),
        ("memory-mode", ("sync", "async", "pool")),
        ("device", ("cpu", "cuda")),
        ("build-type", ("Release", "Debug", "RelWithDebInfo", "MinSizeRel")),
        ("ssd-verify", ("on_read", "read_back")),
        ("spdk-iova-mode", ("auto", "va", "pa")),
    ):
        if config[key] not in choices:
            raise ValueError(f"{key} must be one of {', '.join(choices)}")
    for key, low, high in (
        ("gpu-cache-bytes", 0, 2**63 - 1),
        ("jobs", 1, 1024), ("max-tokens", 0, 2**31 - 1),
        ("top-k", 0, 2**31 - 1), ("seed", 0, 2**32 - 1),
        ("max-model-len", 1, 2**31 - 1),
        ("cpu-cache-bytes", 1, 2**63 - 1), ("ssd-bytes", 4096, 2**63 - 1),
        ("disk-buffer-slots", 2, 65536), ("io-depth", 1, 64),
        ("write-concurrency", 1, 16), ("spdk-mem-mib", 64, 4096),
        ("spdk-chunk-kib", 1, 8192),
    ):
        if not low <= config[key] <= high:
            raise ValueError(f"{key} must be in {low}..{high}")
    if config["file-shards"] not in (1, 4):
        raise ValueError("file-shards must be 1 or 4")
    if config["temperature"] < 0 or not 0 < config["gpu-memory-utilization"] < 1:
        raise ValueError("temperature must be >=0 and gpu-memory-utilization in (0,1)")
    if config["framework"] != "storage" and not config["cuda"]:
        raise ValueError("Native and vLLM builds currently require cuda yes")
    if config["framework"] == "vllm" and config["device"] != "cuda":
        raise ValueError("vLLM integration requires device cuda")
    if config["disk-buffer-slots"] % 2:
        raise ValueError("disk-buffer-slots must be even")
    if config["framework"] == "vllm" and not config["max-tokens"]:
        raise ValueError("vLLM requires max-tokens >=1")
    if not building and config["storage-engine"] == "spdk" and (
        not config["spdk-config"] or not config["spdk-bdev"]
    ):
        raise ValueError("SPDK requires spdk-config and spdk-bdev")
    if config["storage-engine"] == "spdk":
        if config["ssd-bytes"] % 4096 or config["spdk-chunk-kib"] % 4:
            raise ValueError("SPDK requires 4096-byte aligned ssd-bytes and 4-KiB chunks")
        if config["io-depth"] * config["spdk-chunk-kib"] > config["spdk-mem-mib"] * 1024:
            raise ValueError("SPDK DMA buffers exceed spdk-mem-mib")
    for key in ("build-dir", "cmake", "ssd-directory", "spdk-reactor-mask"):
        if not config[key]:
            raise ValueError(f"{key} must not be empty")


def build_plan(config):
    framework = config["framework"]
    flags = {
        "CMAKE_BUILD_TYPE": config["build-type"],
        "INFER_ENABLE_CUDA": config["cuda"],
        "INFER_BUILD_RUNTIME": True,
        "INFER_BUILD_NATIVE": framework == "native",
        "INFER_BUILD_VLLM": framework == "vllm",
        "INFER_BUILD_C_API": config["c-api"],
        "INFER_BUILD_PRELOAD": config["preload"],
        "INFER_MEMORY_MODE": ("sync", "async", "pool").index(config["memory-mode"]),
        "INFER_ENABLE_SPDK": config["storage-engine"] == "spdk",
        "INFER_ENABLE_IO_URING": framework == "native" or config["storage-engine"] in ("uring", "auto"),
        "INFER_ENABLE_CUFILE": config["cufile"],
        "INFER_IO_URING": config["storage-engine"] in ("uring", "auto"),
    }
    flags["INFER_SPDK_ROOT"] = config["spdk-sdk"]
    if framework == "native":
        flags["CMAKE_CUDA_ARCHITECTURES"] = "native"
        flags["MINI_INFER_BUILD_APP"] = True
    if framework == "vllm":
        flags["Python3_EXECUTABLE"] = sys.executable
    configure = [config["cmake"], "-S", str(ROOT), "-B", config["build-dir"], "-G", "Ninja"]
    for key, value in flags.items():
        if type(value) is bool:
            value = "ON" if value else "OFF"
        configure.append(f"-D{key}={value}")
    return {"framework": framework, "commands": [configure,
        [config["cmake"], "--build", config["build-dir"], "--parallel", str(config["jobs"])]]}


def run_plan(config):
    if config["framework"] == "storage":
        raise ValueError("framework storage is build-only; use the C/C++ storage API")
    if not config["model"]:
        raise ValueError("model is required for run")
    if config["framework"] == "native":
        if not config["tokenizer"]:
            raise ValueError("Native run requires tokenizer")
        command = [str(Path(config["build-dir"]) / "mini_infer_main")]
        for flag, key in (("model", "model"), ("tokenizer", "tokenizer"),
                          ("device", "device"), ("prompt", "prompt"),
                          ("steps", "max-tokens"), ("temperature", "temperature"),
                          ("top-k", "top-k"), ("seed", "seed")):
            command.extend([f"--{flag}", str(config[key])])
        keys = ["storage-engine", "gpu-cache-bytes", "cpu-cache-bytes", "ssd-bytes",
                "ssd-directory", "io-depth", "file-shards", "ssd-verify"]
        if config["storage-engine"] == "spdk":
            keys += [f"spdk-{key}" for key in
                     ("config", "bdev", "reactor-mask", "mem-mib", "iova-mode", "no-huge", "no-pci", "chunk-kib")]
        for key in keys:
            value = int(config[key]) if type(config[key]) is bool else config[key]
            command.extend([f"--{key}", str(value)])
        return {"framework": "native", "commands": [command]}
    engine = config["storage-engine"]
    environment = {
        "MINIINFER_BRIDGE": str(Path(config["build-dir"]) / "libminiinfer_vllm_bridge.so"),
        "MINIINFER_POOL_BRIDGE": str(Path(config["build-dir"]) / "libminiinfer_pool_bridge.so"),
        "MINIINFER_FILE_IO": "posix" if engine == "spdk" else engine,
        "MINIINFER_IO_DEPTH": str(config["io-depth"]),
        "MINIINFER_FILE_SHARDS": str(config["file-shards"]),
        "MINIINFER_WRITE_CONCURRENCY": str(config["write-concurrency"]),
        "MINIINFER_SSD_VERIFY": config["ssd-verify"],
    }
    if engine == "spdk":
        for key in ("config", "bdev", "reactor-mask", "mem-mib", "iova-mode", "no-huge", "no-pci", "chunk-kib"):
            value = config[f"spdk-{key}"]
            if type(value) is bool:
                value = int(value)
            environment[f"MINIINFER_SPDK_{key.upper().replace('-', '_')}"] = str(value)
        environment["MINIINFER_SPDK_CAPACITY_BYTES"] = str(config["ssd-bytes"])
    llm = {
        "model": config["model"], "max_model_len": config["max-model-len"],
        "gpu_memory_utilization": config["gpu-memory-utilization"],
        "enforce_eager": config["enforce-eager"], "enable_prefix_caching": True,
        "kv_transfer_config": {
            "kv_connector": "MiniInferTieredConnector",
            "kv_connector_module_path": "tiered_connector", "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lazy_offload": False, "kv_offload_backend": "disk", "use_page_cache": True,
                "disk_path": str(Path(config["ssd-directory"]) / "kv.cache"),
                "cpu_bytes_to_use": config["ssd-bytes"],
                "disk_capacity_bytes": config["ssd-bytes"],
                "miniinfer_ssd_bytes": config["ssd-bytes"],
                "miniinfer_cpu_bytes": config["cpu-cache-bytes"],
                "disk_buffer_slots": config["disk-buffer-slots"], "miniinfer_min_reuse": 1,
            },
        },
    }
    if config["tokenizer"]:
        llm["tokenizer"] = config["tokenizer"]
    return {"framework": "vllm", "commands": [], "environment": environment, "llm": llm,
            "sampling": {"temperature": config["temperature"], "max_tokens": config["max-tokens"],
                         "top_k": config["top-k"] or -1, "seed": config["seed"]}}


def check_build(config):
    path = Path(config["build-dir"]) / "CMakeCache.txt"
    if not path.is_file():
        raise ValueError("build not found; run this command with build first")
    cache = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
    for key, expected in (
        (f"INFER_BUILD_{config['framework'].upper()}", "ON"),
        ("INFER_ENABLE_CUDA", "ON" if config["cuda"] else "OFF"),
        ("INFER_MEMORY_MODE", str(("sync", "async", "pool").index(config["memory-mode"]))),
    ):
        if cache.get(key) != expected:
            raise ValueError(f"build does not match {key}={expected}; rebuild with the current config")
    engine = config["storage-engine"]
    if engine in ("uring", "spdk"):
        key = "INFER_ENABLE_SPDK" if engine == "spdk" else "INFER_ENABLE_IO_URING"
        if cache.get(key) != "ON":
            raise ValueError(f"{engine} is disabled in this build; rebuild with the current config")


def run(config, plan):
    check_build(config)
    if config["framework"] == "native":
        command = plan["commands"][0]
        if not Path(command[0]).is_file():
            raise ValueError("Native executable missing; run build first")
        if config["storage-engine"] != "spdk":
            Path(config["ssd-directory"]).mkdir(parents=True, exist_ok=True)
        os.execv(command[0], command)
    # Imports and environment changes belong only to the selected vLLM process.
    for key in tuple(os.environ):
        if key.startswith("MINIINFER_"):
            del os.environ[key]
    os.environ.update(plan["environment"])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(ROOT / "integrations/vllm"))
    if not Path(plan["environment"]["MINIINFER_BRIDGE"]).is_file():
        raise ValueError("vLLM bridge missing; run build first")
    Path(config["ssd-directory"]).mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version

    if version("vllm") != "0.31.0":
        raise ValueError("vLLM integration requires the tested vllm==0.31.0")
    from vllm import LLM, SamplingParams

    llm = LLM(**plan["llm"])
    outputs = llm.generate([config["prompt"]], SamplingParams(**plan["sampling"]))
    for output in outputs:
        print(output.outputs[0].text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=ROOT / "infer.conf")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the launch/build plan without executing")
    for key in DEFAULTS:
        parser.add_argument(f"--{key}", metavar="VALUE")
    args = parser.parse_args()
    config = load_config(args.config.resolve(),
                         {key: getattr(args, key.replace("-", "_")) for key in DEFAULTS},
                         building=args.command == "build")
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if args.check_config:
        print("Infer configuration is valid")
        return
    plan = build_plan(config) if args.command == "build" else run_plan(config)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.command == "build":
        for command in plan["commands"]:
            print("+ " + shlex.join(command), flush=True)
            subprocess.run(command, check=True)
        return
    run(config, plan)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        print(f"Infer config error: {error}", file=sys.stderr)
        sys.exit(1)
