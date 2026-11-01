"""Experimental single-rank vLLM 0.31 adapter; no changes to installed vLLM.

Reuses vLLM's scheduler, block matching and batch DMA primitive, with packed
Pinned staging and reusable descriptors/events. The backing store belongs to
MiniInfer; no installed vLLM file or model operator is changed.
"""

import ctypes
from collections import deque
from importlib.metadata import version
import json
import operator
import os
from pathlib import Path
import threading
import time
import types

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    build_params,
    copy_blocks,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
)
from packed_dma import PackedDMACopy
from timing import CUDASpanMetrics

BACKENDS = []


class NativeOperation:
    """A batch token; native completion returns buffer ownership even on failure."""

    def __init__(self, store, token):
        self.store, self.token = store, token
        self._done, self._error = False, None

    def result(self):
        if not self._done:
            if not self.store.ptr:
                raise RuntimeError("Native store is closed")
            status = self.store._native_wait(self.store.ptr, self.token)
            self._done = True
            if status:
                self._error = RuntimeError(self.store.lib.mi_storage_error().decode())
        if self._error is not None:
            raise self._error


class NativeStore:
    prefetch_enabled = False
    batch_io = True

    def __init__(self, payload, count, directory):
        self._bind("mi_storage")
        self.lib.mi_storage_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p]
        self.lib.mi_storage_create.restype = ctypes.c_void_p
        self.ptr = self.lib.mi_storage_create(payload, count, os.fsencode(directory))
        if not self.ptr:
            raise RuntimeError(self.lib.mi_storage_error().decode())
        self._payload, self._count = payload, count
        self.file_slot_bytes = max(4096, 1 << (payload - 1).bit_length()) + 64

    def _bind(self, prefix, library=None):
        path = Path(
            os.environ.get(
                "MINIINFER_BRIDGE",
                Path(__file__).resolve().parents[2] / "build/libminiinfer_vllm_bridge.so",
            )
        )
        self.lib = library if library is not None else ctypes.CDLL(str(path))
        self._native_read = getattr(self.lib, prefix + "_read")
        self._native_write = getattr(self.lib, prefix + "_write")
        for method in (self._native_read, self._native_write):
            method.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            method.restype = ctypes.c_int
        self.lib.mi_storage_error.restype = ctypes.c_char_p
        self._native_io_many = getattr(self.lib, prefix + "_io_many")
        self._native_io_many.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._native_io_many.restype = ctypes.c_int
        if prefix == "mi_storage":
            for name in ("mi_storage_read_many", "mi_storage_write_many"):
                method = getattr(self.lib, name)
                method.argtypes = [ctypes.c_void_p] + self._native_io_many.argtypes[2:]
                method.restype = ctypes.c_int
        self._native_submit = getattr(self.lib, prefix + "_submit")
        self._native_submit.argtypes = self._native_io_many.argtypes + [ctypes.POINTER(ctypes.c_uint64)]
        self._native_submit.restype = ctypes.c_int
        self._native_wait = getattr(self.lib, prefix + "_wait")
        self._native_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._native_wait.restype = ctypes.c_int
        self._native_stats = getattr(self.lib, prefix + "_stats")
        self._native_stats.argtypes = [ctypes.c_void_p]
        self._native_stats.restype = ctypes.c_char_p
        self._native_destroy = getattr(self.lib, prefix + "_destroy")
        self._native_destroy.argtypes = [ctypes.c_void_p]
        self._native_destroy.restype = None

    def io(self, write, slot, pointer):
        method = self._native_write if write else self._native_read
        if method(self.ptr, slot, pointer):
            raise RuntimeError(self.lib.mi_storage_error().decode())

    def _validate_batch(self, write, slots, pointers):
        if not self.ptr:
            raise RuntimeError("Native store is closed")
        try:
            slots = [operator.index(slot) for slot in slots]
            pointers = [operator.index(pointer) for pointer in pointers]
        except TypeError as error:
            raise ValueError("Invalid batch slot or buffer") from error
        if len(slots) != len(pointers):
            raise ValueError("Mismatched batch slots/buffers")
        if len(set(slots)) != len(slots):
            raise ValueError("Duplicate batch slot")
        maximum = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
        if any(slot < 0 or slot >= self._count for slot in slots) or any(
            pointer <= 0 or pointer > maximum - self._payload for pointer in pointers
        ):
            raise ValueError("Invalid batch slot or buffer")
        if not write:
            destinations = sorted(pointers)
            if any(b < a + self._payload for a, b in zip(destinations, destinations[1:])):
                raise ValueError("Overlapping batch destinations")
        return slots, pointers

    def io_many(self, write, slots, pointers):
        slots, pointers = self._validate_batch(write, slots, pointers)
        if not self.batch_io:
            for slot, pointer in zip(slots, pointers):
                self.io(write, slot, pointer)
            return
        for begin in range(0, len(slots), 64):
            indices, addresses = slots[begin : begin + 64], pointers[begin : begin + 64]
            if self._native_io_many(
                self.ptr,
                write,
                len(indices),
                (ctypes.c_size_t * len(indices))(*indices),
                (ctypes.c_void_p * len(indices))(*addresses),
            ):
                raise RuntimeError(self.lib.mi_storage_error().decode())

    def submit_io(self, write, slots, pointers):
        slots, pointers = self._validate_batch(write, slots, pointers)
        if not slots or len(slots) > 64:
            raise ValueError("Native submission requires 1..64 slots/buffers")
        token = ctypes.c_uint64()
        if self._native_submit(
            self.ptr,
            write,
            len(slots),
            (ctypes.c_size_t * len(slots))(*slots),
            (ctypes.c_void_p * len(slots))(*pointers),
            ctypes.byref(token),
        ):
            raise RuntimeError(self.lib.mi_storage_error().decode())
        return NativeOperation(self, token.value)

    def stats(self):
        if not self.ptr:
            raise RuntimeError("Native store is closed")
        result = self._native_stats(self.ptr)
        if result is None:
            raise RuntimeError(self.lib.mi_storage_error().decode())
        return json.loads(result)

    def close(self):
        if self.ptr:
            self._native_destroy(self.ptr)
            self.ptr = None


class MiniInferDiskBackend(DiskBackend):
    def __init__(self, store_factory=NativeStore):
        super().__init__()
        self.store_factory = store_factory

    def init(
        self,
        gpu_caches,
        device,
        load_stream,
        store_stream,
        disk_path,
        num_disk_slots,
        total_block_bytes,
        num_buffer_slots=2,
        use_page_cache=False,
        store_factory=None,
    ):
        if num_buffer_slots < 2 or num_buffer_slots % 2:
            raise ValueError("Pinned slots must be an even number >= 2")
        if not use_page_cache:
            raise ValueError("MiniInfer currently supports buffered file I/O only")
        reuse_dma = os.environ.get("MINIINFER_REUSE_DMA", "1")
        if reuse_dma not in ("0", "1"):
            raise ValueError("MINIINFER_REUSE_DMA must be 0 or 1")
        self._reuse_dma = reuse_dma == "1"
        self.timings = CUDASpanMetrics(os.environ.get("MINIINFER_PROFILE_TRANSFERS", "0") == "1")
        self._load_stream, self._store_stream = load_stream, store_stream
        self._total_block_bytes, self._num_buffer_slots = total_block_bytes, num_buffer_slots
        self._tensor_names = list(gpu_caches)
        self._per_tensor_bpb = [t.stride(0) * t.element_size() for t in gpu_caches.values()]
        assert sum(self._per_tensor_bpb) == total_block_bytes
        # 一个连续 Pinned 槽按 layer 切视图：不再打包/解包普通 CPU 临时缓冲。
        self._store_buffer = torch.empty(
            (num_buffer_slots, total_block_bytes), dtype=torch.int8, pin_memory=True
        )
        self._load_buffer = torch.empty_like(self._store_buffer, pin_memory=True)
        # 每个缓冲槽建一次 descriptor，槽内各 layer 的长度与 GPU block stride 相同。
        # copy_blocks 的 CPU block id 固定为 0；实际槽由 descriptor 的基址选择。
        self.store_params, self.load_params = [], []
        for slot in range(1 if self._reuse_dma else num_buffer_slots):
            store_views, load_views, offset = {}, {}, 0
            for name, size in zip(self._tensor_names, self._per_tensor_bpb):
                store_views[name] = self._store_buffer[slot, offset : offset + size].view(1, size)
                load_views[name] = self._load_buffer[slot, offset : offset + size].view(1, size)
                offset += size
            self.store_params.append(
                build_params(
                    gpu_caches,
                    store_views,
                    store_stream,
                    src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
                )
            )
            self.load_params.append(
                build_params(
                    load_views,
                    gpu_caches,
                    load_stream,
                    src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
                )
            )
        self._packed_store = (
            PackedDMACopy(self.store_params[0], total_block_bytes, num_buffer_slots, True)
            if self._reuse_dma
            else None
        )
        self._packed_load = (
            PackedDMACopy(self.load_params[0], total_block_bytes, num_buffer_slots, False)
            if self._reuse_dma
            else None
        )
        # 缓冲半区只在同步并消费旧内容后复用，对应的内部事件也可安全复用。
        # 对调度器发布的完成事件仍独立创建，不能在其消费之前重新 record。
        self._store_buffer_events = [torch.Event(), torch.Event()] if self._reuse_dma else None
        self._load_buffer_events = [torch.Event(), torch.Event()] if self._reuse_dma else None
        self.store = (store_factory or self.store_factory)(
            total_block_bytes, num_disk_slots, str(Path(disk_path).parent)
        )
        self._native_read_ahead = self.store.prefetch_enabled
        self.counts = dict(
            store_blocks=0,
            load_blocks=0,
            store_bytes=0,
            load_bytes=0,
            read_ahead_batches=0,
            read_ahead_bytes=0,
        )
        self.completed_timings = deque(maxlen=256)
        self._failure = None
        self._pending = 0
        self._pending_condition = threading.Condition()
        self.capacity = dict(
            payload_bytes_per_block=total_block_bytes,
            disk_slots=num_disk_slots,
            file_slot_bytes=self.store.file_slot_bytes,
            reuse_dma=self._reuse_dma,
            pinned_read_ahead=self._native_read_ahead,
            pinned_tensor_bytes=self._store_buffer.nbytes + self._load_buffer.nbytes,
        )
        self._store_thread = threading.Thread(
            target=self._guard, args=(self._transfer_loop, device, store_stream, True), daemon=True
        )
        self._load_thread = threading.Thread(
            target=self._guard, args=(self._transfer_loop, device, load_stream, False), daemon=True
        )
        self._store_thread.start()
        self._load_thread.start()
        BACKENDS.append(self)

    def _guard(self, method, *args):
        try:
            method(*args)
        except BaseException as error:
            with self._pending_condition:
                self._failure = error
                self._pending_condition.notify_all()

    def check(self):
        if self._failure:
            raise RuntimeError("MiniInfer background transfer failed") from self._failure

    def launch_copy(self, *args, **kwargs):
        with self._pending_condition:
            self.check()
            if self._shutdown:
                raise RuntimeError("MiniInfer backend is shut down")
            self._pending += 1
            try:
                super().launch_copy(*args, **kwargs)
            except BaseException:
                self._pending -= 1
                self._pending_condition.notify_all()
                raise

    def _transfer_loop(self, device, stream, store):
        torch.cuda.set_device(device)
        queue = self._store_queue if store else self._load_queue
        while True:
            item = queue.get()
            if item is None:
                return
            src, dst, event_idx, events_list, wait_event = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            begin = time.perf_counter_ns()
            (self._do_store if store else self._do_load)(src, dst, stream)
            elapsed = time.perf_counter_ns() - begin
            event = torch.Event()
            event.record(stream)
            # drain 必须等到完成事件也发布；仅 DMA 退出还不足以交回调度器。
            with self._pending_condition:
                self.completed_timings.append(
                    dict(
                        operation="store" if store else "load",
                        blocks=len(src),
                        bytes=len(src) * self._total_block_bytes,
                        elapsed_ns=elapsed,
                    )
                )
                events_list.append((event_idx, event))
                self._pending -= 1
                self._pending_condition.notify_all()

    def _do_store(self, gpu_blocks, disk_slots, stream):
        assert len(gpu_blocks) == len(disk_slots)
        batch = self._num_buffer_slots // 2
        dma, operations = [None, None], [None, None]
        try:
            for i in range(0, len(gpu_blocks), batch):
                group = (i // batch) % 2
                if operations[group] is not None:
                    operation = operations[group]
                    operations[group] = None
                    self._finish_window(operation, True)
                first = group * batch
                gpu = gpu_blocks[i : i + batch]
                self._timed_copy_batch(gpu, first, True, stream)
                event = self._store_buffer_events[group] if self._reuse_dma else torch.Event()
                event.record(stream)
                dma[group] = (
                    event,
                    list(zip(range(first, first + len(gpu)), disk_slots[i : i + batch])),
                )
                # The next D2H window is already submitted while native I/O consumes the last.
                other = 1 - group
                if dma[other] is not None:
                    event, targets = dma[other]
                    self._wait_dma(event, True)
                    operations[other] = self._submit_io_window(targets, True)
                    dma[other] = None
            for group in range(2):
                if dma[group] is not None:
                    event, targets = dma[group]
                    self._wait_dma(event, True)
                    operations[group] = self._submit_io_window(targets, True)
                    dma[group] = None
            for group in range(2):
                if operations[group] is not None:
                    operation = operations[group]
                    operations[group] = None
                    self._finish_window(operation, True)
        finally:
            # Drain every accepted native job before returning any Pinned half, even on failure.
            try:
                failure = None
                for operation in operations:
                    if operation is not None:
                        try:
                            operation[0].result()
                        except BaseException as error:
                            if failure is None:
                                failure = error
                if failure is not None:
                    raise failure
            finally:
                stream.synchronize()

    def _do_load(self, disk_slots, gpu_blocks, stream):
        assert len(gpu_blocks) == len(disk_slots)
        batch = self._num_buffer_slots // 2
        events = [None, None]
        pending = None

        def targets(offset):
            first = ((offset // batch) % 2) * batch
            slots = disk_slots[offset : offset + batch]
            return list(zip(range(first, first + len(slots)), slots))

        try:
            if self._native_read_ahead and disk_slots:
                pending = self._submit_window(targets(0), None)
            for i in range(0, len(disk_slots), batch):
                group = (i // batch) % 2
                first_slot = group * batch
                gpu = gpu_blocks[i : i + batch]
                if pending is not None:
                    started = time.perf_counter_ns()
                    self._finish_window(pending)
                    self.timings.add_host("load_read_wait_ns", time.perf_counter_ns() - started)
                    pending = None
                    if i:
                        self.counts["read_ahead_batches"] += 1
                        self.counts["read_ahead_bytes"] += len(gpu) * self._total_block_bytes
                else:
                    self._read_window(targets(i), events[group])
                if self._native_read_ahead and i + batch < len(disk_slots):
                    # Only the next window in this scheduler-owned load plan.
                    # Read directly into existing Pinned memory, even when the
                    # ordinary CPU cache is full; no speculative hot eviction.
                    pending = self._submit_window(targets(i + batch), events[1 - group])
                self._timed_copy_batch(gpu, first_slot, False, stream)
                event = self._load_buffer_events[group] if self._reuse_dma else torch.Event()
                event.record(stream)
                events[group] = event
        finally:
            # A failed DMA/submission must not leave a reader writing into a
            # window that shutdown or the scheduler can reclaim.
            try:
                if pending is not None:
                    pending[0].result()
            finally:
                # Also cover a copy enqueued before event.record failed.
                stream.synchronize()

    def _submit_window(self, targets, previous_dma):
        if previous_dma is not None:
            self._wait_dma(previous_dma, False)
        return self._submit_io_window(targets, False)

    def _submit_io_window(self, targets, write):
        buffer = self._store_buffer if write else self._load_buffer
        base = buffer.data_ptr()
        started = time.perf_counter_ns()
        operation = self.store.submit_io(
            write,
            [disk for _, disk in targets],
            [base + slot * self._total_block_bytes for slot, _ in targets],
        )
        return operation, len(targets), started

    def _finish_window(self, pending, write=False):
        operation, count, started = pending
        try:
            operation.result()
            direction = "store" if write else "load"
            self.counts[direction + "_blocks"] += count
            self.counts[direction + "_bytes"] += count * self._total_block_bytes
        finally:
            # Submission-to-completion host span includes queueing and overlap.
            self.timings.add_host(
                "store_io_ns" if write else "load_io_ns", time.perf_counter_ns() - started
            )

    def _read_window(self, targets, previous_dma):
        if previous_dma is not None:
            self._wait_dma(previous_dma, False)
        self._io_slots(targets, False)

    def _wait_dma(self, event, store):
        started = time.perf_counter_ns()
        event.synchronize()
        self.timings.add_host(
            "store_dma_wait_ns" if store else "load_dma_wait_ns",
            time.perf_counter_ns() - started,
        )

    def _copy_batch(self, gpu_blocks, first_slot, store):
        if self._reuse_dma:
            (self._packed_store if store else self._packed_load).copy(gpu_blocks, first_slot)
            return
        # 把若干 token blocks 的所有 layer 合并为一次成熟的 cuMemcpyBatchAsync。
        # GPU layer stride 与 packed host stride 不同，因此显式生成每项基址；
        # 传给 copy_blocks 的 block id 为 0，bpb 只充当本项复制长度。
        proto = (self.store_params if store else self.load_params)[0]
        ids = np.asarray(gpu_blocks, dtype=np.uint64)
        slots = np.arange(first_slot, first_slot + len(ids), dtype=np.uint64)
        gpu_bases = proto.src_bases if store else proto.dst_bases
        host_bases = proto.dst_bases if store else proto.src_bases
        gpu_ptrs = (gpu_bases[None, :] + ids[:, None] * proto.bpb[None, :]).ravel()
        host_ptrs = (host_bases[None, :] + slots[:, None] * self._total_block_bytes).ravel()
        params = proto._replace(
            src_bases=gpu_ptrs if store else host_ptrs,
            dst_bases=host_ptrs if store else gpu_ptrs,
            bpb=np.tile(proto.bpb, len(ids)),
            num_layers=proto.num_layers * len(ids),
        )
        copy_blocks([0], [0], params)

    def _timed_copy_batch(self, gpu_blocks, first_slot, store, stream):
        marker = self.timings.start("store_dma" if store else "load_dma", stream)
        try:
            self._copy_batch(gpu_blocks, first_slot, store)
        finally:
            self.timings.finish(marker, stream)

    def _io_slots(self, targets, write):
        started = time.perf_counter_ns()
        try:
            buffer = self._store_buffer if write else self._load_buffer
            # 缓冲连续且槽间距固定；避免逐块创建 Tensor view 和 ctypes 往返。
            base = buffer.data_ptr()
            self.store.io_many(
                write,
                [disk for _, disk in targets],
                [base + slot * self._total_block_bytes for slot, _ in targets],
            )
            prefix = "store" if write else "load"
            self.counts[prefix + "_blocks"] += len(targets)
            self.counts[prefix + "_bytes"] += len(targets) * self._total_block_bytes
        finally:
            self.timings.add_host(
                "store_io_ns" if write else "load_io_ns", time.perf_counter_ns() - started
            )

    def drain(self):
        with self._pending_condition:
            while self._pending:
                self.check()
                self._pending_condition.wait()
            self.check()
        self._store_stream.synchronize()
        self._load_stream.synchronize()
        self.timings.collect(wait=True)
        if hasattr(self.store, "drain_prefetch"):
            self.store.drain_prefetch()

    def shutdown(self):
        with self._pending_condition:
            if self._shutdown:
                return
            self._shutdown = True
            # 与 launch_copy 串行，不能把新任务放在退出标记后面。
            self._store_queue.put(None)
            self._load_queue.put(None)
        for thread in (self._store_thread, self._load_thread):
            thread.join()
        self._store_stream.synchronize()
        self._load_stream.synchronize()
        self.store.close()
        # 完成等待之后放掉 tensor 所有权；全局统计档案不应保活 Pinned 缓冲。
        self._store_buffer = self._load_buffer = None
        self._packed_store = self._packed_load = None
        self._store_buffer_events = self._load_buffer_events = None
        self._store_buffer_caches.clear()
        self._load_buffer_caches.clear()


class MiniInferConnector(SimpleCPUOffloadConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        if version("vllm") != "0.31.0":
            raise RuntimeError("Experimental connector requires tested vLLM 0.31.0")
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("Benchmark connector validated only for world_size=1")
        super().__init__(vllm_config, role, kv_cache_config)
        if self.worker_handler is not None:
            self.worker_handler._init_disk_mode = types.MethodType(
                _init_disk_mode, self.worker_handler
            )

    def get_finished(self, finished_req_ids):
        if self.worker_handler and isinstance(self.worker_handler._backend, MiniInferDiskBackend):
            self.worker_handler._backend.check()
        return super().get_finished(finished_req_ids)

    def handle_preemptions(self, kv_connector_metadata):
        if (
            kv_connector_metadata.need_flush
            and self.worker_handler
            and isinstance(self.worker_handler._backend, MiniInferDiskBackend)
        ):
            # 父类只等待已登记事件，后台队列中的任务尚未登记；先排空再复用 KV 块。
            self.worker_handler._backend.drain()
        return super().handle_preemptions(kv_connector_metadata)


def _init_disk_mode(worker, caches, bytes_per_block, device, *, store_factory=NativeStore):
    worker.num_cpu_blocks = max(1, worker.disk_capacity_bytes // bytes_per_block)
    worker._backend = MiniInferDiskBackend(store_factory)
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
