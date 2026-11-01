"""Torch tensor ownership over the project's actual C++ pools; no payload copy."""

import ctypes
import json
import math
import os
from pathlib import Path
import torch

KINDS = {"cpu": 0, "gpu": 1, "pinned": 2}
_LIBRARIES = {}
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_capsule_new.restype = ctypes.py_object
_capsule_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_valid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_capsule_valid.restype = ctypes.c_int
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_capsule_pointer.restype = ctypes.c_void_p


def library():
    path = str(
        Path(
            os.environ.get(
                "MINIINFER_POOL_BRIDGE",
                Path(__file__).resolve().parents[2] / "build/libminiinfer_pool_bridge.so",
            )
        ).resolve()
    )
    if path in _LIBRARIES:
        return _LIBRARIES[path]
    lib = ctypes.CDLL(path)
    lib.mi_pool_create.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.mi_pool_create.restype = ctypes.c_void_p
    lib.mi_pool_close.argtypes = [ctypes.c_void_p]
    lib.mi_pool_close.restype = None
    lib.mi_pool_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.mi_pool_tensor.restype = ctypes.c_void_p
    lib.mi_pool_delete_tensor.argtypes = [ctypes.c_void_p]
    lib.mi_pool_delete_tensor.restype = None
    lib.mi_pool_owns.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.mi_pool_owns.restype = ctypes.c_int
    lib.mi_pool_stats.argtypes = [ctypes.c_void_p]
    lib.mi_pool_stats.restype = ctypes.c_char_p
    lib.mi_pool_error.restype = ctypes.c_char_p

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def capsule_destructor(capsule):
        if _capsule_valid(capsule, b"dltensor"):
            lib.mi_pool_delete_tensor(_capsule_pointer(capsule, b"dltensor"))

    # Keep both the native library and unused-capsule destructor alive.
    lib._capsule_destructor = capsule_destructor
    _LIBRARIES[path] = lib
    return lib


class Pools:
    def __init__(self, gpu_limit=512 << 20, cpu_limit=512 << 20, pinned_limit=64 << 20, device=0):
        self.lib = library()
        self.handle = self.lib.mi_pool_create(device, gpu_limit, cpu_limit, pinned_limit)
        if not self.handle:
            raise RuntimeError(self.lib.mi_pool_error().decode())
        self.allocations = []

    def tensor(self, kind, shape, dtype=torch.uint8):
        if not self.handle:
            raise RuntimeError("Pool is closed")
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if kind not in KINDS or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("Positive tensor dimensions and supported pool kind required")
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes >= 2**63:
            raise ValueError("Tensor size overflow")
        pointer = self.lib.mi_pool_tensor(self.handle, KINDS[kind], nbytes)
        if not pointer:
            raise RuntimeError(self.lib.mi_pool_error().decode())
        try:
            capsule = _capsule_new(pointer, b"dltensor", self.lib._capsule_destructor)
        except BaseException:
            self.lib.mi_pool_delete_tensor(pointer)
            raise
        tensor = torch.utils.dlpack.from_dlpack(capsule).view(dtype).reshape(shape)
        if not self.owns(kind, tensor.data_ptr()):
            raise RuntimeError("Tensor does not alias its project-pool allocation")
        self.allocations.append(
            dict(
                kind=kind,
                pointer=tensor.data_ptr(),
                bytes=nbytes,
                shape=list(shape),
                dtype=str(dtype),
            )
        )
        return tensor

    def owns(self, kind, pointer):
        result = self.lib.mi_pool_owns(self.handle, KINDS[kind], pointer)
        if result < 0:
            raise RuntimeError(self.lib.mi_pool_error().decode())
        return bool(result)

    def stats(self):
        result = self.lib.mi_pool_stats(self.handle)
        if result is None:
            raise RuntimeError(self.lib.mi_pool_error().decode())
        return json.loads(result)

    def close(self):
        if self.handle:
            self.lib.mi_pool_close(self.handle)
            self.handle = None
