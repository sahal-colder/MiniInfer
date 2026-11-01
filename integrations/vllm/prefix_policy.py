"""Bounded prompt-only offload using vLLM's existing pins and completions."""

from collections import OrderedDict
import types

from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id


class PromptOffloadPolicy:
    def __init__(self, manager, max_batched_tokens, history_capacity):
        self.manager = manager
        self.history_capacity = history_capacity
        self.hashes = OrderedDict()
        self.watermark = 2 * ((max_batched_tokens + manager.block_size - 1) // manager.block_size)
        self.skipped_eager_blocks = self.eviction_blocks = self.tail_blocks = 0
        self.full_prefix_hits = 0
        self.deduplicated_blocks = 0

    def observe(self, request, allowed):
        if not allowed or request.skip_reading_prefix_cache:
            return
        count = request.num_prompt_tokens // self.manager.hash_block_size
        for block_hash in request.block_hashes[:count]:
            key = make_block_hash_with_group_id(block_hash, 0)
            self.hashes[key] = None
            self.hashes.move_to_end(key)
        while len(self.hashes) > self.history_capacity:
            self.hashes.popitem(last=False)

    def reset(self):
        self.hashes.clear()

    def stats(self):
        return dict(
            prompt_hashes=len(self.hashes),
            skipped_eager_blocks=self.skipped_eager_blocks,
            eviction_blocks=self.eviction_blocks,
            tail_blocks=self.tail_blocks,
            full_prefix_hits=self.full_prefix_hits,
            deduplicated_blocks=self.deduplicated_blocks,
        )

    def install(self):
        manager = self.manager
        select = manager._select_eager_blocks_to_store
        prepare = manager.prepare_store_specs
        match = manager.get_num_new_matched_tokens
        policy = self

        def select_tail(self, state, groups):
            gpu, advanced, metadata = select(state, groups)
            count = state.request.num_prompt_tokens // self.block_size
            tail = groups[0][count - 1] if count and count <= len(groups[0]) else None
            keep = [i for i, block in enumerate(gpu) if block == tail]
            pool = self._gpu_block_pool
            pending = {pool.blocks[index].block_hash for index in self._in_flight_store_gpu_blocks}
            if keep and pool.blocks[tail].block_hash in pending:
                policy.deduplicated_blocks += len(keep)
                keep = []
            policy.skipped_eager_blocks += len(gpu) - len(keep)
            policy.tail_blocks += len(keep)
            return (
                [gpu[i] for i in keep],
                advanced,
                [metadata[i] for i in keep] if metadata is not None else None,
            )

        def prepare_prompt(self, output):
            gpu, cpu, requests, metadata = prepare(output)
            pool = self._gpu_block_pool
            if pool is None:
                return gpu, cpu, requests, metadata
            free = self.cpu_block_pool.get_num_free_blocks()
            selected = []
            keys = {pool.blocks[index].block_hash for index in self._in_flight_store_gpu_blocks}
            # Always inspect the oldest bounded window. Advancing a cursor
            # through the entire free queue would also copy recent GPU data.
            for visited, block in enumerate(pool.free_block_queue.iter_blocks_after(None)):
                if visited >= policy.watermark or len(selected) >= free:
                    break
                key = block.block_hash
                if (
                    block.is_null
                    or key not in policy.hashes
                    or key in keys
                    or block.block_id in self._in_flight_store_gpu_blocks
                    or self.cpu_block_pool.cached_block_hash_to_block.get_one_block(key) is not None
                ):
                    continue
                selected.append(block.block_id)
                keys.add(key)
            if selected:
                destinations = self.cpu_block_pool.get_new_blocks(len(selected))
                pool.touch([pool.blocks[index] for index in selected])
                self._in_flight_store_gpu_blocks.update(selected)
                gpu.extend(selected)
                cpu.extend(block.block_id for block in destinations)
                if metadata is not None:
                    metadata.extend({} for _ in selected)
                policy.eviction_blocks += len(selected)
            return gpu, cpu, requests, metadata

        def match_full_prompt(self, request, computed):
            # The scheduler already replays the final token after an async
            # full-prompt load. Preserve the base path for all other cases.
            if (
                request.skip_reading_prefix_cache
                or request.num_tokens != request.num_prompt_tokens
                or not request.num_tokens
                or request.num_tokens % self.block_size
                or computed % self.block_size
                or computed >= request.num_tokens
            ):
                return match(request, computed)
            stale = self._pending_cpu_hits.pop(request.request_id, None)
            if stale is not None:
                self._free_pending_cpu_hit(stale)
            hashes = request.block_hashes[computed // self.hash_block_size :]
            blocks, length, _ = self.cpu_coordinator.find_longest_cache_hit(
                hashes, request.num_tokens - computed
            )
            if not length:
                return 0, False
            self.cpu_block_pool.touch(
                [block for group in blocks for block in group if not block.is_null]
            )
            self._pending_cpu_hits[request.request_id] = (blocks, length, computed)
            policy.full_prefix_hits += int(length + computed == request.num_tokens)
            return length, True

        manager._select_eager_blocks_to_store = types.MethodType(select_tail, manager)
        manager.prepare_store_specs = types.MethodType(prepare_prompt, manager)
        manager.get_num_new_matched_tokens = types.MethodType(match_full_prompt, manager)
