"""Compare the Python budget contract with actual CPU/Pinned allocations."""
import gc
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/vllm"))
from host_budget import host_cache_budget, kv_slot_bytes, pool_allocation_bytes, pool_usable_budget
from pool_runtime import Pools


@pytest.mark.parametrize("kind", ["cpu", "pinned"])
def test_budget_rounding_matches_real_pool(kind):
    pools = Pools(gpu_limit=16 << 20, cpu_limit=32 << 20, pinned_limit=32 << 20)
    try:
        for size in [1, 64, 65, 193, 1024, 1025, 4096, 4097, 5003, 5003 * 16,
                     (1 << 20) - 1, 1 << 20, (1 << 20) + 1, 7 << 20, (7 << 20) + 1,
                     28 << 20, (28 << 20) + 1]:
            before = pools.stats()[kind]["active_bytes"]
            tensor = pools.tensor(kind, (size,), torch.uint8)
            assert pools.stats()[kind]["active_bytes"] - before == pool_allocation_bytes(size)
            assert pools.owns(kind, tensor.data_ptr())
            del tensor
            gc.collect()
            assert pools.stats()[kind]["active_bytes"] == before
    finally:
        pools.close()


@pytest.mark.parametrize("payload", [5003, 9000, (1792 << 10) + 1])
def test_compact_slot_matches_real_page_aligned_pool_allocation(payload):
    pools = Pools(cpu_limit=32 << 20)
    try:
        before = pools.stats()["cpu"]["active_bytes"]
        tensor = pools.tensor("cpu", ((payload + 4095) // 4096 * 4096,), torch.uint8)
        assert pools.stats()["cpu"]["active_bytes"] - before == kv_slot_bytes(payload, compact=True)
        del tensor
        gc.collect()
        assert pools.stats()["cpu"]["active_bytes"] == before
    finally:
        pools.close()


def test_rounding_overflow_and_floor_boundaries():
    maximum = (1 << 64) - 64
    assert pool_allocation_bytes(4097) == 4160
    assert pool_allocation_bytes(5003 * 16) == 80896
    assert pool_allocation_bytes(maximum) == maximum
    assert pool_usable_budget(maximum) == maximum
    assert pool_usable_budget(383 << 20) == 380 << 20
    assert pool_usable_budget(4095) == 4032
    for invalid in [0, -1, True, maximum + 1]:
        with pytest.raises(ValueError):
            pool_allocation_bytes(invalid)


@pytest.mark.parametrize("payload", [5003, 1792 << 10])
def test_staging_budget_equals_real_two_buffer_charge(payload):
    pools = Pools(pinned_limit=64 << 20)
    try:
        before = pools.stats()["pinned"]["active_bytes"]
        first = pools.tensor("pinned", (16, payload), torch.uint8)
        second = pools.tensor("pinned", first.shape, first.dtype)
        charged = pools.stats()["pinned"]["active_bytes"] - before
        assert host_cache_budget(512 << 20, payload, 16, True) + charged == 512 << 20
        del first, second
        gc.collect()
        assert pools.stats()["pinned"]["active_bytes"] == before
    finally:
        pools.close()
