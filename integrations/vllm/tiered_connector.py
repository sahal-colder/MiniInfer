"""Experimental vLLM adapter: repeat-request admission + native CPU/SSD cache.

vLLM owns GPU KV and logical offload-slot eviction. MiniInfer owns each slot's
CPU/SSD placement through TieredStorage. No installation files are patched.
"""

import ctypes
import os
import types
from functools import partial
from pathlib import Path

from admission import CapacityAdmission, FrequencyAdmission, install_admission
from cache_cost import CacheCostModel, release_pending_hit
from host_budget import (
    host_cache_budget,
    kv_slot_bytes,
    pool_usable_budget,
    ssd_payload_capacity,
)
from miniinfer_connector import (
    MiniInferConnector,
    MiniInferDiskBackend,
    NativeStore,
)
from prefix_policy import PromptOffloadPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec

POLICIES = []
PROMPT_POLICIES = []


class NativeTieredStore(NativeStore):
    def __init__(self, payload, count, directory, cpu_budget_bytes):
        compact = os.environ.get("MINIINFER_COMPACT_SLOTS", "0")
        if compact not in ("0", "1"):
            raise ValueError("MINIINFER_COMPACT_SLOTS must be 0 or 1")
        hot = os.environ.get("MINIINFER_BATCH_HOT", "0")
        if hot not in ("0", "1"):
            raise ValueError("MINIINFER_BATCH_HOT must be 0 or 1")
        options = int(compact) | (int(hot) << 1)
        overlap = os.environ.get("MINIINFER_OVERLAP_WRITEBACK", "0")
        if overlap not in ("0", "1"):
            raise ValueError("MINIINFER_OVERLAP_WRITEBACK must be 0 or 1")
        options |= int(overlap) << 2
        protect = os.environ.get("MINIINFER_CPU_HOT", "0")
        if protect not in ("0", "1"):
            raise ValueError("MINIINFER_CPU_HOT must be 0 or 1")
        options |= int(protect) << 3
        slot_bytes = kv_slot_bytes(payload, compact == "1")
        self.file_slot_bytes = slot_bytes + 64
        if os.environ.get("MINIINFER_SPDK_BDEV"):
            self.file_slot_bytes = (self.file_slot_bytes + 4095) // 4096 * 4096
        cpu_slots = min(count, cpu_budget_bytes // slot_bytes)
        if cpu_slots < 1:
            raise ValueError("CPU cache budget cannot hold one physical KV slot")
        path = Path(
            os.environ.get(
                "MINIINFER_BRIDGE",
                Path(__file__).resolve().parents[2] / "build/libminiinfer_vllm_bridge.so",
            )
        )
        self.lib = ctypes.CDLL(str(path))
        self.environment_after_close_bytes = 0
        self.lib.mi_spdk_environment_bytes.argtypes = []
        self.lib.mi_spdk_environment_bytes.restype = ctypes.c_size_t
        self.lib.mi_spdk_available.argtypes = []
        self.lib.mi_spdk_available.restype = ctypes.c_int
        if os.environ.get("MINIINFER_SPDK_BDEV") and not self.lib.mi_spdk_available():
            raise RuntimeError("Bridge does not support SPDK; rebuild with INFER_ENABLE_SPDK=ON")
        self._bind("mi_tiered", self.lib)
        self.lib.mi_tiered_create_options.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.lib.mi_tiered_create_options.restype = ctypes.c_void_p
        prefetch = os.environ.get("MINIINFER_PREFETCH", "0")
        if prefetch not in ("0", "1"):
            raise ValueError("MINIINFER_PREFETCH must be 0 or 1")
        self.prefetch_enabled = prefetch == "1"
        self.lib.mi_tiered_prefetch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.mi_tiered_prefetch.restype = ctypes.c_int
        self.lib.mi_tiered_prefetch_poll.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.mi_tiered_prefetch_poll.restype = ctypes.c_int
        self.lib.mi_tiered_prefetch_drain.argtypes = [ctypes.c_void_p]
        self.lib.mi_tiered_prefetch_drain.restype = ctypes.c_int
        self.batch_io = os.environ.get("MINIINFER_BATCH_IO", "1") != "0"
        mode = {"auto": 0, "posix": 1, "uring": 2, "direct": 2}[os.environ.get("MINIINFER_FILE_IO", "posix")]
        depth = int(os.environ.get("MINIINFER_IO_DEPTH", "8"))
        writes = int(os.environ.get("MINIINFER_WRITE_CONCURRENCY", "4"))
        self.ptr = self.lib.mi_tiered_create_options(
            payload, count, cpu_slots, os.fsencode(directory), mode, depth, writes, options
        )
        if not self.ptr:
            raise RuntimeError(self.lib.mi_storage_error().decode())
        self._payload, self._count = payload, count

    def prefetch(self, slot):
        if not self.ptr:
            raise RuntimeError("Tiered store is closed")
        operation = ctypes.c_uint64()
        status = self.lib.mi_tiered_prefetch(self.ptr, slot, ctypes.byref(operation))
        if status < 0:
            raise RuntimeError(self.lib.mi_storage_error().decode())
        return None if status == 1 else operation.value

    def poll_prefetch(self, operation, wait=False):
        if not self.ptr:
            raise RuntimeError("Tiered store is closed")
        resident = ctypes.c_int()
        status = self.lib.mi_tiered_prefetch_poll(self.ptr, operation, wait, ctypes.byref(resident))
        if status < 0:
            raise RuntimeError(self.lib.mi_storage_error().decode())
        return None if status == 1 else bool(resident.value)

    def drain_prefetch(self):
        if not self.ptr:
            raise RuntimeError("Tiered store is closed")
        if self.lib.mi_tiered_prefetch_drain(self.ptr):
            raise RuntimeError(self.lib.mi_storage_error().decode())

    def close(self):
        if self.ptr:
            try:
                self.drain_prefetch()
            finally:
                self.lib.mi_tiered_destroy(self.ptr)
                self.ptr = None
                self.environment_after_close_bytes = self.lib.mi_spdk_environment_bytes()


class MiniInferTieredBackend(MiniInferDiskBackend):
    def __init__(self, cpu_budget_bytes):
        super().__init__()
        self.cpu_budget_bytes = cpu_budget_bytes

    def init(self, *args, **kwargs):
        kwargs["store_factory"] = partial(NativeTieredStore, cpu_budget_bytes=self.cpu_budget_bytes)
        super().init(*args, **kwargs)


class MiniInferTieredConnector(MiniInferConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        if extra.get("lazy_offload", False):
            raise ValueError("Tiered admission currently requires eager offload")
        if extra.get("kv_offload_backend") != "disk" and not extra.get(
            "miniinfer_pinned_resident", False
        ):
            raise ValueError("Tiered connector requires disk backend configuration")
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError("Tiered admission supports one FullAttention KV group")
        if "miniinfer_ssd_bytes" in extra:
            payload = sum(t.size for t in kv_cache_config.kv_cache_tensors) // kv_cache_config.num_blocks
            capacity = ssd_payload_capacity(
                extra["miniinfer_ssd_bytes"], payload,
                os.environ.get("MINIINFER_COMPACT_SLOTS", "0") == "1",
                4096 if os.environ.get("MINIINFER_SPDK_BDEV") else 1,
            )
            extra["disk_capacity_bytes"] = capacity
            extra["cpu_bytes_to_use"] = capacity
        cpu_bytes = extra.get("miniinfer_cpu_bytes", 512 * 1024**2)
        if type(cpu_bytes) is not int or cpu_bytes <= 0:
            raise ValueError("miniinfer_cpu_bytes must be a positive integer")
        self.cpu_resident = bool(extra.get("miniinfer_cpu_resident", False))
        self.pinned_resident = bool(extra.get("miniinfer_pinned_resident", False))
        if self.pinned_resident and not self.cpu_resident:
            raise ValueError("Pinned resident mode requires capacity admission")
        if self.pinned_resident and extra.get("kv_offload_backend", "cpu") != "cpu":
            raise ValueError("Pinned resident mode requires the native CPU DMA backend")
        if "miniinfer_host_total_bytes" in extra:
            total = extra["miniinfer_host_total_bytes"]
            if type(total) is not int or total <= 0:
                raise ValueError("Total host budget must be a positive integer")
            payload = (
                sum(t.size for t in kv_cache_config.kv_cache_tensors) // kv_cache_config.num_blocks
            )
            cpu_bytes = (
                total
                if self.pinned_resident
                else host_cache_budget(
                    total,
                    payload,
                    extra.get("disk_buffer_slots", 2),
                    extra.get("miniinfer_pooled_staging", False),
                    os.environ.get("MINIINFER_COMPACT_SLOTS", "0") == "1",
                    (int(os.environ.get("MINIINFER_SPDK_MEM_MIB", "256")) << 20)
                    if os.environ.get("MINIINFER_SPDK_BDEV")
                    else 0,
                    staging_in_device=bool(os.environ.get("MINIINFER_SPDK_BDEV")),
                )
            )
        if self.cpu_resident:
            payload = (
                sum(t.size for t in kv_cache_config.kv_cache_tensors) // kv_cache_config.num_blocks
            )
            # arena 内按 payload 紧排；整块采用现有 CPU/Pinned 池的四分之一档位。
            slot = (
                payload
                if self.pinned_resident
                else kv_slot_bytes(payload, os.environ.get("MINIINFER_COMPACT_SLOTS", "0") == "1")
            )
            available = pool_usable_budget(cpu_bytes) if self.pinned_resident else cpu_bytes
            slots = available // slot
            if slots < 2:
                raise ValueError("CPU-resident profile requires a null slot and a reusable slot")
            # 外部逻辑槽与物理 CPU 容量匹配，替换时复用原 CPU 槽，不制造无价值 SSD 写回。
            extra["disk_capacity_bytes"] = slots * payload
            extra["cpu_bytes_to_use"] = slots * payload
        super().__init__(vllm_config, role, kv_cache_config)
        self.admission = None
        self.cost_model = None
        self.prompt_policy = None
        self._restore_decisions = {}
        if self.scheduler_manager is not None:
            if self.cpu_resident:
                manager = self.scheduler_manager
                capacity = (
                    (manager.num_cpu_blocks - 1) * manager.block_size // manager.hash_block_size
                )
                self.admission = CapacityAdmission(
                    capacity,
                    extra.get("miniinfer_min_reuse", 2),
                    extra.get("miniinfer_history_capacity", 16384),
                )
            else:
                self.admission = FrequencyAdmission(
                    extra.get("miniinfer_min_reuse", 2),
                    extra.get("miniinfer_history_capacity", 16384),
                )
                self.admission.prompt_only = bool(extra.get("miniinfer_prompt_only", False))
            if extra.get("miniinfer_cost_profile") is not None:
                self.cost_model = CacheCostModel(
                    extra["miniinfer_cost_profile"], extra.get("miniinfer_cost_identity")
                )
                self.admission.cost_model = self.cost_model
                self.admission.tokens_per_block = self.scheduler_manager.hash_block_size
            install_admission(self.scheduler_manager, self.admission)
            if (
                extra.get("miniinfer_prompt_offload", True)
                and not self.cpu_resident
                and not self.admission.prompt_only
            ):
                self.prompt_policy = PromptOffloadPolicy(
                    self.scheduler_manager,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    max(1024, 4 * kv_cache_config.num_blocks),
                )
                self.prompt_policy.install()
                PROMPT_POLICIES.append(self.prompt_policy)
            POLICIES.append(self.admission)
        if self.worker_handler is not None and not self.pinned_resident:
            self.worker_handler._miniinfer_cpu_bytes = cpu_bytes
            self.worker_handler._init_disk_mode = types.MethodType(
                _init_tiered_mode, self.worker_handler
            )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        hit, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if hit and self.cost_model is not None:
            rid = request.request_id
            if rid not in self._restore_decisions:
                self._restore_decisions[rid] = self.cost_model.restore(
                    hit, hit // self.scheduler_manager.hash_block_size
                )
            if self._restore_decisions[rid] is False:
                release_pending_hit(self.scheduler_manager, rid)
                return 0, False
        return hit, asynchronous

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self.admission is not None:
            manager = self.scheduler_manager
            count = (
                request.num_prompt_tokens - int(self.admission.prompt_only)
            ) // manager.hash_block_size
            key = (
                (count, request.block_hashes[count - 1])
                if count and not request.skip_reading_prefix_cache
                else None
            )
            allowed = self.admission.observe(request.request_id, key)
            if self.prompt_policy is not None:
                self.prompt_policy.observe(request, allowed)
        return super().update_state_after_alloc(request, blocks, num_external_tokens)

    def request_finished(self, request, block_ids):
        try:
            return super().request_finished(request, block_ids)
        finally:
            self._restore_decisions.pop(request.request_id, None)
            if self.admission is not None:
                self.admission.finish(request.request_id)

    def request_finished_all_groups(self, request, block_ids):
        try:
            # finish-time tail selection must see the same decision as earlier steps.
            return super().request_finished_all_groups(request, block_ids)
        finally:
            self._restore_decisions.pop(request.request_id, None)
            if self.admission is not None:
                self.admission.finish(request.request_id)

    def build_connector_meta(self, scheduler_output):
        state = scheduler_output.kv_connector_block_state
        if state is not None and state.boundary_state_offloads:
            raise ValueError("Boundary-state offload is unsupported by tiered admission")
        return super().build_connector_meta(scheduler_output)

    def reset_cache(self):
        result = super().reset_cache()
        if result and self.admission is not None:
            self.admission.reset()
        if result:
            self._restore_decisions.clear()
            if self.prompt_policy is not None:
                self.prompt_policy.reset()
        return result


def _init_tiered_mode(worker, caches, bytes_per_block, device):
    worker.num_cpu_blocks = max(1, worker.disk_capacity_bytes // bytes_per_block)
    worker._backend = MiniInferTieredBackend(worker._miniinfer_cpu_bytes)
    worker._backend.init(
        caches,
        device,
        worker.load_stream,
        worker.store_stream,
        worker.disk_path,
        worker.num_cpu_blocks,
        bytes_per_block,
        worker.disk_buffer_slots,
        worker.use_page_cache,
    )
