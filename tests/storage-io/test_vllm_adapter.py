"""Verify native offload and real GPU→host→SSD→GPU use project allocations."""

import gc
import sys
import tempfile
import time
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations" / "vllm"))
from pool_runtime import Pools
from pool_adapter import install


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.pool = Pools(gpu_limit=16 << 20, cpu_limit=8 << 20, pinned_limit=8 << 20)
        self.addCleanup(self.pool.close)
        self.addCleanup(gc.collect)
        self.patch = install(self.pool, gpu=True, native_cpu=True, staging=True)
        self.addCleanup(self.patch.restore)

    def test_native_cpu_offload_allocation(self):
        from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

        caches = {"layer0": self.pool.tensor("gpu", (16, 4096), torch.int8)}
        worker = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
        worker.num_cpu_blocks = 16
        worker.load_stream, worker.store_stream = torch.cuda.Stream(), torch.cuda.Stream()
        worker._init_cpu_mode(caches, 4096, torch.device("cuda:0"))
        t = worker.cpu_kv_caches["layer0"]
        self.assertTrue(t.is_pinned())
        self.assertTrue(self.pool.owns("pinned", t.data_ptr()))
        self.assertEqual(t.shape, (16, 4096))
        worker._backend.shutdown()

    def test_native_layers_share_bounded_pinned_arena(self):
        from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

        caches = {str(i): self.pool.tensor("gpu", (292, 1024), torch.int8) for i in range(28)}
        worker = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
        worker.num_cpu_blocks = 292
        worker.load_stream, worker.store_stream = torch.cuda.Stream(), torch.cuda.Stream()
        worker._init_cpu_mode(caches, 28 * 1024, torch.device("cuda:0"))
        self.assertEqual(len(worker.cpu_kv_caches), 28)
        self.assertLessEqual(self.pool.stats()["pinned"]["reserved_bytes"], 8 << 20)
        for tensor in worker.cpu_kv_caches.values():
            self.assertTrue(self.pool.owns("pinned", tensor.data_ptr()))
        worker._backend.shutdown()

    def test_native_pinned_dma_sparse_roundtrip(self):
        from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

        caches = {
            str(i): self.pool.tensor("gpu", (40, size), torch.int8)
            for i, size in enumerate((4096, 6144, 2048))
        }
        for i, t in enumerate(caches.values()):
            t.copy_((torch.arange(t.numel(), device="cuda").reshape(t.shape) + i).to(torch.int8))
        reference = {k: t.cpu() for k, t in caches.items()}
        worker = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
        worker.num_cpu_blocks = 40
        worker.load_stream, worker.store_stream = torch.cuda.Stream(), torch.cuda.Stream()
        worker._init_cpu_mode(caches, 12288, torch.device("cuda:0"))
        original = list(range(37, 0, -2))
        slots = list(range(0, 38, 2))
        restored = list(range(1, 38, 2))

        def finish(events):
            deadline = time.monotonic() + 10
            while not events and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(events, "DMA completion event was not published")
            for _, event in events:
                event.synchronize()

        try:
            ready = torch.Event()
            ready.record()
            events = []
            worker._backend.launch_copy(original, slots, True, 1, events, ready)
            finish(events)
            for t in caches.values():
                t.zero_()
            torch.cuda.synchronize()
            events = []
            worker._backend.launch_copy(slots, restored, False, 2, events)
            finish(events)
            for name, t in caches.items():
                actual = t.cpu()
                for old, new in zip(original, restored):
                    self.assertTrue(torch.equal(actual[new], reference[name][old]))
                self.assertEqual(torch.count_nonzero(actual[::2]).item(), 0)
        finally:
            worker._backend.shutdown()

    def test_multilayer_tiered_roundtrip_and_neighbors(self):
        from tiered_connector import MiniInferTieredBackend

        tensors = {
            f"layer{i}": self.pool.tensor("gpu", (40, size), torch.int8)
            for i, size in enumerate((4096, 6144, 2048))
        }
        for i, t in enumerate(tensors.values()):
            t.copy_((torch.arange(t.numel(), device="cuda").reshape(t.shape) + i).to(torch.int8))
        reference = {k: t.cpu() for k, t in tensors.items()}
        load, save = torch.cuda.Stream(), torch.cuda.Stream()
        with tempfile.TemporaryDirectory() as directory:
            backend = MiniInferTieredBackend(cpu_budget_bytes=4 * 16384)
            backend.init(
                tensors,
                torch.device("cuda:0"),
                load,
                save,
                directory + "/cache",
                40,
                12288,
                num_buffer_slots=16,
                use_page_cache=True,
            )
            self.assertTrue(self.pool.owns("pinned", backend._store_buffer.data_ptr()))
            self.assertTrue(self.pool.owns("pinned", backend._load_buffer.data_ptr()))
            ready = torch.Event()
            ready.record()
            original = list(range(37, 0, -2))
            slots = list(range(0, 38, 2))
            events = []
            backend.launch_copy(original, slots, True, 1, events, ready)
            backend.drain()
            for tensor in tensors.values():
                tensor.zero_()
            torch.cuda.synchronize()
            restored = list(range(1, 38, 2))
            backend.launch_copy(slots, restored, False, 2, events)
            backend.drain()
            for name, t in tensors.items():
                actual = t.cpu()
                for old, new in zip(original, restored):
                    self.assertTrue(torch.equal(actual[new], reference[name][old]))
                self.assertEqual(torch.count_nonzero(actual[::2]).item(), 0)
            stats = backend.store.stats()
            self.assertEqual(backend.counts["store_blocks"], len(original))
            self.assertEqual(backend.counts["load_blocks"], len(restored))
            self.assertGreater(stats["cpu_hits"], 0)
            self.assertGreater(stats["ssd_hits"], 0)
            self.assertLessEqual(stats["cpu_pool_reserved_bytes"], stats["cpu_limit_bytes"])
            backend.shutdown()
        self.assertGreater(self.pool.stats()["gpu"]["upstream_allocations"], 0)

    def test_real_native_read_failure_is_terminal_and_fresh_backend_recovers(self):
        from miniinfer_connector import MiniInferDiskBackend

        tensor = self.pool.tensor("gpu", (4, 4096), torch.int8)
        tensor.fill_(17)
        torch.cuda.synchronize()
        reference = tensor.cpu()
        load, save = torch.cuda.Stream(), torch.cuda.Stream()
        with tempfile.TemporaryDirectory() as directory:
            def create(name):
                path = Path(directory) / name
                path.mkdir()
                backend = MiniInferDiskBackend()
                backend.init({"layer": tensor}, torch.device("cuda:0"), load, save,
                             str(path / "cache"), 4, 4096, num_buffer_slots=4, use_page_cache=True)
                self.assertTrue(backend.capacity["native_transfer_executor"])
                self.assertTrue(self.pool.owns("pinned", backend._load_buffer.data_ptr()))
                return backend

            failed = create("failed")
            events = []
            try:
                # No data has been written: the actual C++ storage read fails.
                failed.launch_copy([0], [0], False, 1, events)
                with self.assertRaisesRegex(RuntimeError, "background transfer failed") as error:
                    failed.drain()
                self.assertIsNotNone(error.exception.__cause__)
                self.assertEqual(str(error.exception.__cause__), "Slot has not been written")
                self.assertEqual(events, [])
                self.assertTrue(torch.equal(tensor.cpu(), reference))
                with self.assertRaisesRegex(RuntimeError, "background transfer failed"):
                    failed.launch_copy([0], [1], True, 2, [])
            finally:
                failed.shutdown()
            self.assertTrue(load.query())
            self.assertTrue(save.query())

            recovered = create("recovered")
            try:
                ready = torch.Event()
                ready.record()
                recovered.launch_copy([0], [0], True, 3, [], ready)
                recovered.drain()
                tensor[0].zero_()
                torch.cuda.synchronize()
                recovered.launch_copy([0], [0], False, 4, [])
                recovered.drain()
                self.assertEqual(recovered.counts["native_transfer_calls"], 2)
                self.assertTrue(torch.equal(tensor.cpu(), reference))
            finally:
                recovered.shutdown()


if __name__ == "__main__":
    unittest.main()
