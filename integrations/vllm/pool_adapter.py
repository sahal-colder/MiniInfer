"""Process-local allocation injection; preserve installed vLLM layout/copy code.

The cloned function has a private globals dict. Its torch allocation facade is
never installed in torch itself or in the original module globals. No installed
file is changed. This adapter intentionally supports the tested vLLM version.
"""

import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path
import types
import torch


class AllocationTorch:
    def __init__(self, pool, kind, role, audit):
        self.pool, self.kind, self.role, self.audit = pool, kind, role, audit

    def __getattr__(self, name):
        return getattr(torch, name)

    def _allocate(self, shape, dtype, device, zero):
        device = torch.device(device or "cpu")
        expected = "cuda" if self.kind == "gpu" else "cpu"
        if device.type != expected or (expected == "cuda" and device.index not in (None, 0)):
            raise ValueError("Unexpected device in scoped storage allocation")
        t = self.pool.tensor(self.kind, shape, dtype)
        if zero:
            t.zero_()
        if self.kind == "pinned" and not t.is_pinned():
            raise RuntimeError("Project Pinned allocation is not CUDA pinned memory")
        self.audit.append(
            dict(
                role=self.role,
                pointer=t.data_ptr(),
                bytes=t.nbytes,
                shape=list(t.shape),
                stride=list(t.stride()),
                pool=self.kind,
            )
        )
        return t

    def zeros(self, size, *, dtype=None, device=None, **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported zeros arguments: {kwargs}")
        return self._allocate(size, dtype or torch.get_default_dtype(), device, True)

    def empty(self, size, *, dtype=None, device=None, pin_memory=False, **kwargs):
        if kwargs or not pin_memory:
            raise ValueError("Only the pinned staging allocation is supported")
        return self._allocate(size, dtype or torch.get_default_dtype(), device, False)

    def empty_like(self, other, *, pin_memory=False, **kwargs):
        if kwargs or not pin_memory:
            raise ValueError("Unexpected staging empty_like")
        return self._allocate(other.shape, other.dtype, other.device, False)


class PinnedArenaTorch(AllocationTorch):
    """One pooled allocation avoids per-layer size-class padding exceeding budget."""

    def __init__(self, pool, total, audit):
        super().__init__(pool, "pinned", "native_cpu_offload", audit)
        self.arena = (
            super()._allocate((total,), torch.uint8, "cpu", False)
            if pool
            else torch.empty(total, dtype=torch.uint8, pin_memory=True)
        )
        self.offset = 0

    def _allocate(self, shape, dtype, device, zero):
        import math

        size = math.prod(shape) * dtype.itemsize
        if torch.device(device).type != "cpu" or self.offset + size > self.arena.nbytes:
            raise ValueError("Native CPU arena bounds or device mismatch")
        result = self.arena[self.offset : self.offset + size].view(dtype).reshape(shape)
        self.offset += size
        if zero:
            result.zero_()
        return result


def clone_function(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    result = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class AdapterSession:
    def __init__(
        self,
        pool,
        gpu,
        native_cpu,
        staging,
        native_torch_pinned=False,
        native_disk_torch_pinned=False,
    ):
        if version("vllm") not in ("0.31.0", "0.32.0"):
            raise RuntimeError("Allocation adapter requires vLLM 0.31.0 or 0.32.0")
        self.pool = pool
        self.allocations = []
        self.kv_views = []
        self.sources = {}
        self.originals = []
        try:
            if gpu:
                import vllm.v1.worker.gpu.attn_utils as runner

                original = runner.allocate_kv_cache
                if (list(inspect.signature(original).parameters) !=
                        ["kv_cache_config", "device", "layout", "kernel_block_sizes"] or
                        "buf = torch.zeros" not in inspect.getsource(original)):
                    raise RuntimeError("vLLM KV allocation implementation changed")
                gpu_allocate = clone_function(
                    original, torch=AllocationTorch(pool, "gpu", "gpu_kv", self.allocations)
                )

                def allocate(*args, **kwargs):
                    result = gpu_allocate(*args, **kwargs)
                    for layer, tensor in result.items():
                        if not pool.owns("gpu", tensor.data_ptr()):
                            raise RuntimeError("Real vLLM KV view escaped project GPU pool")
                        self.kv_views.append(
                            dict(
                                layer=layer,
                                pointer=tensor.data_ptr(),
                                shape=list(tensor.shape),
                                stride=list(tensor.stride()),
                                dtype=str(tensor.dtype),
                            )
                        )
                    return result

                self.replace(runner, "allocate_kv_cache", allocate)
            if native_cpu or native_torch_pinned:
                from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

                original = SimpleCPUOffloadWorker._init_cpu_mode
                if list(inspect.signature(original).parameters) != [
                    "self", "unique_gpu_caches", "total_bytes_per_block", "device"
                ]:
                    raise RuntimeError("vLLM CPU offload allocation interface changed")

                def native_init(worker, caches, total_bytes_per_block, device):
                    facade = PinnedArenaTorch(
                        pool, worker.num_cpu_blocks * total_bytes_per_block, self.allocations
                    )
                    allocated = clone_function(
                        original, torch=facade, pin_tensor=lambda t: None, PIN_MEMORY=True
                    )
                    allocated(worker, caches, total_bytes_per_block, device)
                    if facade.offset != facade.arena.nbytes:
                        raise RuntimeError("Incomplete native CPU arena layout")

                self.replace(SimpleCPUOffloadWorker, "_init_cpu_mode", native_init)
            if staging:
                from miniinfer_connector import MiniInferDiskBackend

                allocated = clone_function(
                    MiniInferDiskBackend._prepare_staging,
                    torch=AllocationTorch(pool, "pinned", "staging", self.allocations),
                )
                self.replace(MiniInferDiskBackend, "_prepare_staging", allocated)
            if native_disk_torch_pinned:
                from vllm.v1.simple_kv_offload.disk_backend import DiskBackend

                def aligned(slots, bpb):
                    size = slots * bpb
                    raw = torch.empty(size + 4096, dtype=torch.int8, pin_memory=True)
                    offset = -raw.data_ptr() % 4096
                    return raw[offset : offset + size].view(slots, bpb)

                self.replace(
                    DiskBackend,
                    "init",
                    clone_function(
                        DiskBackend.init, _alloc_aligned=aligned, pin_tensor=lambda t: None
                    ),
                )
        except BaseException:
            self.restore()
            raise

    def replace(self, target, name, function):
        original = getattr(target, name)
        path = Path(inspect.getfile(original)).resolve()
        self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.originals.append((target, name, original))
        setattr(target, name, function)

    def restore(self):
        for target, name, original in reversed(self.originals):
            setattr(target, name, original)
        self.originals.clear()

    def audit(self):
        return dict(
            allocations=self.allocations,
            kv_views=self.kv_views,
            sources=self.sources,
            pools=self.pool.stats(),
        )


def install(
    pool,
    *,
    gpu=True,
    native_cpu=False,
    staging=False,
    native_torch_pinned=False,
    native_disk_torch_pinned=False,
):
    return AdapterSession(
        pool, gpu, native_cpu, staging, native_torch_pinned, native_disk_torch_pinned
    )
