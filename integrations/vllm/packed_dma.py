"""Reuse packed-host DMA descriptors with vLLM 0.31's batch-copy primitive.

Only GPU block addresses change between submissions. Host addresses, lengths,
attributes and descriptor storage remain bounded by the staging slot count.
The driver consumes the host descriptor arrays before returning, just as in
vLLM's cuda_mem_ops.copy_blocks; the payload stays live until stream completion.
"""

import ctypes

import numpy as np
from vllm.v1.simple_kv_offload import cuda_mem_ops


class PackedDMACopy:
    def __init__(self, proto, slot_bytes, slots, store):
        self.proto, self.store = proto, store
        self.slots, self.layers = slots, proto.num_layers
        self.gpu_bases = proto.src_bases if store else proto.dst_bases
        host_bases = proto.dst_bases if store else proto.src_bases
        self.host = host_bases[None, :] + np.arange(slots, dtype=np.uint64)[:, None] * slot_bytes
        self.gpu = np.empty_like(self.host)
        self.lengths = np.tile(proto.bpb, slots)
        self.function = cuda_mem_ops._batch_memcpy[0]  # build_params resolved it.
        self.max_descriptors = cuda_mem_ops._resolve_max_batch_descriptors()

    def copy(self, blocks, first_slot):
        count = len(blocks)
        if first_slot < 0 or first_slot + count > self.slots:
            raise ValueError("Packed DMA exceeds staging slots")
        if not count:
            return
        # 每个方向只由一个 worker 提交；两个方向有独立的地址数组和属性。
        target = self.gpu[first_slot : first_slot + count]
        ids = np.asarray(blocks, dtype=np.uint64)
        np.multiply(ids[:, None], self.proto.bpb[None, :], out=target)
        np.add(target, self.gpu_bases[None, :], out=target)
        begin, total = first_slot * self.layers, count * self.layers
        gpu = target.reshape(-1)
        host = self.host[first_slot : first_slot + count].reshape(-1)
        lengths = self.lengths[begin : begin + total]
        src, dst = (gpu, host) if self.store else (host, gpu)
        step = self.max_descriptors or total
        # 保留 vLLM 的 descriptor 上限、源访问顺序和错误位置语义。
        for offset in range(0, total, step):
            end = min(offset + step, total)
            error = self.function(
                dst[offset:end].ctypes.data,
                src[offset:end].ctypes.data,
                lengths[offset:end].ctypes.data,
                end - offset,
                ctypes.addressof(self.proto.attrs),
                ctypes.byref(self.proto.attrs_idx),
                self.proto.num_attrs,
                ctypes.byref(self.proto.fail_idx),
                self.proto.stream_handle,
            )
            if error:
                raise RuntimeError(
                    f"batch memcpy failed: err={error} failIdx={self.proto.fail_idx.value}"
                )
