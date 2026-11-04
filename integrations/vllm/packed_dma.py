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
        self.src_address = (self.gpu if store else self.host).ctypes.data
        self.dst_address = (self.host if store else self.gpu).ctypes.data
        self.lengths_address = self.lengths.ctypes.data
        self.attrs = ctypes.addressof(proto.attrs)
        self.attrs_idx = ctypes.byref(proto.attrs_idx)
        self.fail_idx = ctypes.byref(proto.fail_idx)
        self.direct = None
        if self.layers == 1 and slot_bytes == int(proto.bpb[0]) and cuda_mem_ops.current_platform.is_cuda():
            from cuda.bindings import driver

            error, pointer, _ = driver.cuGetProcAddress(b"cuMemcpyAsync", 4000, 0)
            if error != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuGetProcAddress(cuMemcpyAsync) failed: {error}")
            self.direct = ctypes.CFUNCTYPE(
                ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p
            )(pointer)
            self.direct_gpu_base = int(self.gpu_bases[0])
            self.direct_host_base = int(host_bases[0])
            self.direct_block_bytes = slot_bytes

    def copy(self, blocks, first_slot):
        count = len(blocks)
        if first_slot < 0 or first_slot + count > self.slots:
            raise ValueError("Packed DMA exceeds staging slots")
        if not count:
            return
        if self.direct is not None and blocks[0] >= 0 and (
            count == 1 or all(block == blocks[0] + index for index, block in enumerate(blocks))
        ):
            gpu = self.direct_gpu_base + int(blocks[0]) * self.direct_block_bytes
            host = self.direct_host_base + first_slot * self.direct_block_bytes
            src, dst = (gpu, host) if self.store else (host, gpu)
            # A contiguous one-layer window needs one stream-ordered copy.
            # STREAM also satisfies the stable-host-source ANY contract.
            error = self.direct(dst, src, count * self.direct_block_bytes, self.proto.stream_handle)
            if error:
                raise RuntimeError(f"async memcpy failed: err={error}")
            return
        # 每个方向只由一个 worker 提交；两个方向有独立的地址数组和属性。
        target = self.gpu[first_slot : first_slot + count]
        ids = np.asarray(blocks, dtype=np.uint64)
        np.multiply(ids[:, None], self.proto.bpb[None, :], out=target)
        np.add(target, self.gpu_bases[None, :], out=target)
        begin, total = first_slot * self.layers, count * self.layers
        step = self.max_descriptors or total
        # 保留 vLLM 的 descriptor 上限、源访问顺序和错误位置语义。
        for offset in range(0, total, step):
            end = min(offset + step, total)
            address_offset = (begin + offset) * 8
            error = self.function(
                self.dst_address + address_offset,
                self.src_address + address_offset,
                self.lengths_address + address_offset,
                end - offset,
                self.attrs,
                self.attrs_idx,
                self.proto.num_attrs,
                self.fail_idx,
                self.proto.stream_handle,
            )
            if error:
                raise RuntimeError(
                    f"batch memcpy failed: err={error} failIdx={self.proto.fail_idx.value}"
                )
