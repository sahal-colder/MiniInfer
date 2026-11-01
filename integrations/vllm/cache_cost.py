"""Opt-in cost gate using matched, completed-operation calibration samples.

No timing extrapolation: unknown sizes and stale profiles retain the base policy.
The identity must include model, hardware, storage tier, budgets and build hashes.
"""

import math


class CacheCostModel:
    def __init__(self, profile, identity):
        self.matched = bool(identity) and profile.get("identity") == identity
        self.cost_rejections = self.cost_fallbacks = 0
        self.samples = {}
        rows = profile.get("samples", [])
        if not isinstance(rows, list) or len(rows) > 256:
            raise ValueError("Cost calibration requires at most 256 size buckets")
        for row in rows:
            for key in ("tokens", "blocks", "count"):
                if type(row.get(key)) is not int or row[key] <= 0:
                    raise ValueError("Invalid calibration size/sample count")
            for key in ("recompute_ns", "read_ns", "write_ns", "eviction_ns"):
                value = row.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid completed-operation timing")
            key = (row["tokens"], row["blocks"])
            if key in self.samples:
                raise ValueError("Duplicate cost bucket")
            self.samples[key] = dict(row)

    def _sample(self, tokens, blocks):
        row = self.samples.get((tokens, blocks)) if self.matched else None
        return row if row and row["count"] >= 3 else None

    def restore(self, tokens, blocks):
        row = self._sample(tokens, blocks)
        if row is None:
            return None
        profitable = row["read_ns"] < row["recompute_ns"]
        return None if profitable and row.get("rejection_only", False) else profitable

    def store(self, tokens, blocks):
        row = self._sample(tokens, blocks)
        # Conservative: charge saving and eviction to one subsequent reuse.
        if row is None:
            return None
        profitable = row["read_ns"] + row["write_ns"] + row["eviction_ns"] < row["recompute_ns"]
        return None if profitable and row.get("rejection_only", False) else profitable


    def can_store(self, tokens, blocks):
        decision = self.store(tokens, blocks)
        self.cost_fallbacks += int(decision is None)
        self.cost_rejections += int(decision is False)
        return decision is not False

    def stats(self):
        return dict(
            cost_matched=self.matched,
            cost_rejections=self.cost_rejections,
            cost_fallbacks=self.cost_fallbacks,
        )


def release_pending_hit(manager, request_id):
    pending = manager._pending_cpu_hits.pop(request_id, None)
    if pending is not None:
        manager._free_pending_cpu_hit(pending)
