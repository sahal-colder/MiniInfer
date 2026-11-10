"""Release bridge verification policies and real native queue ownership."""
import ctypes
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/vllm"))
from miniinfer_connector import NativeStore
from tiered_connector import NativeTieredStore


@pytest.fixture(autouse=True)
def file_backend(monkeypatch):
    for key in ["MINIINFER_SPDK_BDEV", "MINIINFER_COMPACT_SLOTS", "MINIINFER_BATCH_HOT",
                "MINIINFER_OVERLAP_WRITEBACK", "MINIINFER_SSD_VERIFY"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MINIINFER_FILE_IO", "posix")
    monkeypatch.setenv("MINIINFER_FILE_SHARDS", "1")


def stats(store):
    if isinstance(store, NativeTieredStore):
        return store.stats()
    store.lib.mi_storage_stats.argtypes = [ctypes.c_void_p]
    store.lib.mi_storage_stats.restype = ctypes.c_char_p
    return json.loads(store.lib.mi_storage_stats(store.ptr))


@pytest.mark.parametrize("tiered", [False, True])
@pytest.mark.parametrize("verification", [None, "none", "on_read", "read_back"])
def test_real_release_verification_contract(tmp_path, monkeypatch, tiered, verification):
    if verification is not None:
        monkeypatch.setenv("MINIINFER_SSD_VERIFY", verification)
    store = (NativeTieredStore(193, 2, str(tmp_path), cpu_budget_bytes=4096) if tiered
             else NativeStore(193, 2, str(tmp_path)))
    source = ctypes.create_string_buffer(b"x" * 193)
    output = ctypes.create_string_buffer(193)
    try:
        store.io(True, 0, ctypes.addressof(source))
        if tiered:
            store.io(True, 1, ctypes.addressof(source))
        before = stats(store)
        assert before["ssd_payload_written_bytes"] == 193
        assert before["ssd_payload_read_bytes"] == 0
        assert before["ssd_verification_read_bytes"] == (193 if verification == "read_back" else 0)
        store.io(False, 0, ctypes.addressof(output))
        assert output.raw == b"x" * 193
        after = stats(store)
        assert after["ssd_payload_read_bytes"] == 193
        assert after["ssd_verification_read_bytes"] == before["ssd_verification_read_bytes"]
    finally:
        store.close()


@pytest.mark.parametrize("verification", ["none", "on_read", "read_back"])
def test_native_queue_bound_drain_and_reuse_preserve_payload(tmp_path, monkeypatch, verification):
    monkeypatch.setenv("MINIINFER_SSD_VERIFY", verification)
    payload = 5003
    store = NativeStore(payload, 17, str(tmp_path))
    source = ctypes.create_string_buffer(b"x" * payload)
    output = ctypes.create_string_buffer(payload)
    try:
        pending = [store.submit_io(True, [index], [ctypes.addressof(source)]) for index in range(16)]
        with pytest.raises(RuntimeError, match="full"):
            store.submit_io(True, [16], [ctypes.addressof(source)])
        for operation in pending:
            operation.result()
            operation.result()
        assert store.stats()["native_queue_peak"] == 16
        for index in range(16):
            store.io(False, index, ctypes.addressof(output))
            assert output.raw == b"x" * payload
        store.submit_io(True, [16], [ctypes.addressof(source)]).result()
        store.io(False, 16, ctypes.addressof(output))
        assert output.raw == b"x" * payload
        result = store.stats()
        assert result["record_writes"] == 17
        assert result["source_crc_bytes"] == (0 if verification == "none" else 17 * payload)
    finally:
        store.close()
