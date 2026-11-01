"""Bounded repeat-request admission; independent of vLLM GPU allocations."""

from collections import OrderedDict
import types


class FrequencyAdmission:
    prompt_only = False

    def __init__(self, min_reuse=2, history_capacity=16384):
        if (
            type(min_reuse) is not int
            or type(history_capacity) is not int
            or min_reuse < 1
            or history_capacity < 1
        ):
            raise ValueError("Admission threshold and history capacity must be positive integers")
        self.min_reuse, self.history_capacity = min_reuse, history_capacity
        self.history = OrderedDict()
        self.active = {}
        self.cost_model = None
        self.tokens_per_block = 1
        self.decay_interval = 0
        self.frequency_limit = min_reuse
        self.observed_requests = self.admitted_requests = 0

    def observe(self, request_id, key):
        if request_id in self.active:
            previous, allowed = self.active[request_id]
            if previous != key:
                raise ValueError("An active request changed its prompt admission key")
            return allowed
        self.observed_requests += 1
        if self.decay_interval and self.observed_requests % self.decay_interval == 0:
            for item in self.history:
                self.history[item] //= 2
        allowed = False
        if key is not None:
            count = min(self.frequency_limit, self.history.pop(key, 0) + 1)
            self.history[key] = count
            if len(self.history) > self.history_capacity:
                self.history.popitem(last=False)
            allowed = self._admit(key, count)
        self.active[request_id] = (key, allowed)
        self.admitted_requests += int(allowed)
        return allowed

    def can_admit(self, key):
        return self.cost_model is None or self.cost_model.can_store(
            key[0] * self.tokens_per_block, key[0]
        )

    def _admit(self, key, count):
        return count >= self.min_reuse and self.can_admit(key)

    def allows(self, request_id):
        return self.active.get(request_id, (None, False))[1]

    def finish(self, request_id):
        self.active.pop(request_id, None)

    def reset(self):
        self.history.clear()
        self.active.clear()

    def stats(self):
        return dict(
            min_reuse=self.min_reuse,
            history_capacity=self.history_capacity,
            history_entries=len(self.history),
            active_requests=len(self.active),
            observed_requests=self.observed_requests,
            admitted_requests=self.admitted_requests,
            rejected_requests=self.observed_requests - self.admitted_requests,
            **(self.cost_model.stats() if self.cost_model is not None else {}),
        )


class CapacityAdmission(FrequencyAdmission):
    """Bounded, decaying prefix admission for a CPU-resident latency profile.

    The key's first field is the number of reusable hash blocks. Admission
    compares recent frequency before native vLLM allocates any destination;
    vLLM still owns actual eviction, pinning and completion.
    """

    prompt_only = True

    def __init__(self, capacity_blocks, min_reuse=2, history_capacity=16384, decay_interval=128):
        super().__init__(min_reuse, history_capacity)
        if type(capacity_blocks) is not int or capacity_blocks < 1 or decay_interval < 2:
            raise ValueError("Invalid prefix admission capacity/decay")
        self.capacity_blocks, self.decay_interval = capacity_blocks, decay_interval
        self.frequency_limit = 1 << 30
        self.residents = {}

    def _admit(self, key, frequency):
        cost = key[0]
        if type(cost) is not int or cost < 1:
            raise ValueError("Invalid prefix block cost")
        if not self.can_admit(key):
            return False
        if key in self.residents:
            return True
        if frequency < self.min_reuse or cost > self.capacity_blocks:
            return False
        free = self.capacity_blocks - sum(self.residents.values())
        pinned = {k for k, accepted in self.active.values() if accepted}
        victims = sorted(
            (k for k in self.residents if k not in pinned),
            key=lambda k: self.history.get(k, 0),
        )
        selected = []
        for victim in victims:
            if free >= cost or self.history.get(victim, 0) >= frequency:
                break
            selected.append(victim)
            free += self.residents[victim]
        if free < cost:
            return False
        for victim in selected:
            del self.residents[victim]
        self.residents[key] = cost
        return True

    def reset(self):
        super().reset()
        self.residents.clear()

    def stats(self):
        return dict(
            super().stats(),
            capacity_blocks=self.capacity_blocks,
            admitted_blocks=sum(self.residents.values()),
            resident_prefixes=len(self.residents),
        )


def install_admission(manager, policy):
    """Filter both eager selection paths before the parent allocates/pins slots.

    Only installed for a single FullAttention group without boundary handoffs.
    Retains the original methods and all of vLLM's completion/refcount machinery.
    """
    select = manager._select_eager_blocks_to_store
    tail = manager._find_fa_partial_tail_source

    def select_admitted(self, state, groups):
        if not policy.allows(state.request.request_id):
            return (
                [],
                [0] * len(self.cpu_kv_cache_config.kv_cache_groups),
                ([] if self.enable_kv_cache_events else None),
            )
        if policy.prompt_only:
            # 只保存下次同前缀请求能复用的完整块，最后一个 prompt token 仍需重算 logits。
            limit = max(0, (state.request.num_prompt_tokens - 1) // self.block_size)
            groups = tuple(ids[:limit] for ids in groups)
        return select(state, groups)

    def admitted_tail(self, request, groups):
        if not policy.allows(request.request_id) or policy.prompt_only:
            return None
        return tail(request, groups)

    manager._select_eager_blocks_to_store = types.MethodType(select_admitted, manager)
    manager._find_fa_partial_tail_source = types.MethodType(admitted_tail, manager)
