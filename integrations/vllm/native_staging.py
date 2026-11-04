"""DLPack views retaining the native SPDK allocation through all Torch aliases."""

import ctypes
import math
import torch

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


def tensor(lib, handle, shape, dtype=torch.uint8, device=None, prefix="mi_store"):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Positive staging tensor dimensions required")
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes >= 2**63:
        raise ValueError("Staging tensor size overflow")
    if lib._name not in _LIBRARIES:
        lib.mi_store_delete_staging_tensor.argtypes = [ctypes.c_void_p]
        lib.mi_store_delete_staging_tensor.restype = None

        @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        def capsule_destructor(capsule):
            if _capsule_valid(capsule, b"dltensor"):
                lib.mi_store_delete_staging_tensor(_capsule_pointer(capsule, b"dltensor"))

        lib._staging_capsule_destructor = capsule_destructor
        _LIBRARIES[lib._name] = lib
    lib = _LIBRARIES[lib._name]
    factory = getattr(lib, prefix + "_staging_tensor")
    factory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    factory.restype = ctypes.c_void_p
    # Initialize Torch's runtime too when the bridge uses a different cudart.
    torch.cuda.init()
    pointer = factory(handle, nbytes, torch.cuda.current_device() if device is None else device)
    if not pointer:
        raise RuntimeError(lib.mi_storage_error().decode())
    try:
        capsule = _capsule_new(pointer, b"dltensor", lib._staging_capsule_destructor)
    except BaseException:
        lib.mi_store_delete_staging_tensor(pointer)
        raise
    return torch.utils.dlpack.from_dlpack(capsule).view(dtype).reshape(shape)
