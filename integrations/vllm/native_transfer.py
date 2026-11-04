"""One native call per transfer plan; CUDA and disk keep the staging ownership."""
import ctypes


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('blocks', 'windows', 'prefetched_blocks', 'prefetched_windows')]


class NativeTransfer:
    @staticmethod
    def available(store):
        available = getattr(store.lib, 'mi_transfer_available', None)
        if available is None:
            return False
        available.argtypes, available.restype = [], ctypes.c_int
        return bool(available())

    def __init__(self, store, caches, slots, store_buffer, load_buffer, store_stream,
                 load_stream, descriptor_cap):
        self.store = store
        self.create = store.lib.mi_transfer_create
        self.create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.create.restype = ctypes.c_void_p
        self.execute_call = store.lib.mi_transfer_execute
        self.execute_call.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(Result)]
        self.execute_call.restype = ctypes.c_int
        self.destroy = store.lib.mi_transfer_destroy
        self.destroy.argtypes, self.destroy.restype = [ctypes.c_void_p], None
        tensors = list(caches.values())
        self.handle = self.create(store.ptr, store._prefix == 'mi_tiered',
            min(t.shape[0] for t in tensors), slots, len(tensors),
            (ctypes.c_uint64 * len(tensors))(*(t.data_ptr() for t in tensors)),
            (ctypes.c_size_t * len(tensors))(*(t.stride(0) * t.element_size() for t in tensors)),
            store_buffer.data_ptr(), load_buffer.data_ptr(), store_stream.cuda_stream,
            load_stream.cuda_stream, descriptor_cap, store.prefetch_enabled)
        if not self.handle:
            raise RuntimeError(store.lib.mi_storage_error().decode())

    def execute(self, source, destination, write, counts, payload):
        if len(source) != len(destination):
            raise ValueError('Native transfer source/destination lengths differ')
        gpu, disk = (source, destination) if write else (destination, source)
        result = Result()
        status = self.execute_call(self.handle, write, len(gpu),
            (ctypes.c_size_t * len(gpu))(*gpu), (ctypes.c_size_t * len(disk))(*disk),
            ctypes.byref(result))
        prefix = 'store' if write else 'load'
        counts[prefix + '_blocks'] += result.blocks
        counts[prefix + '_bytes'] += result.blocks * payload
        counts['read_ahead_batches'] += result.prefetched_windows
        counts['read_ahead_bytes'] += result.prefetched_blocks * payload
        counts['native_transfer_calls'] += 1
        if status:
            raise RuntimeError(self.store.lib.mi_storage_error().decode())

    def close(self):
        if self.handle:
            self.destroy(self.handle)
            self.handle = None
