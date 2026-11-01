"""Optional bounded CUDA-event spans and host wall times; spans can overlap."""

from collections import deque
import threading
import time

import torch


class CUDASpanMetrics:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._pending = deque()
        self._totals = {}
        self._lock = threading.Lock()

    def start(self, label, stream):
        if not self.enabled:
            return None
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        return label, begin, end, time.perf_counter_ns()

    def finish(self, marker, stream):
        if marker is None:
            return
        label, begin, end, started = marker
        end.record(stream)
        with self._lock:
            self._add(label + "_host_ns", time.perf_counter_ns() - started)
            self._pending.append((label, begin, end))
            # Bound event ownership even for a very large offload operation.
            self._collect(len(self._pending) >= 256)

    def add_host(self, label, elapsed):
        if self.enabled:
            with self._lock:
                self._add(label, elapsed)

    def add_poll(self, elapsed, cpu):
        if self.enabled:
            with self._lock:
                self._add("kv_poll_host_ns", elapsed)
                self._add("kv_poll_thread_cpu_ns", cpu)
                self._add("kv_poll_calls", 1)

    def _add(self, label, value):
        self._totals[label] = self._totals.get(label, 0) + value

    def _collect(self, wait):
        while self._pending:
            label, begin, end = self._pending[0]
            if not wait and not end.query():
                break
            if wait:
                end.synchronize()
            self._add(label + "_gpu_span_ns", round(begin.elapsed_time(end) * 1e6))
            self._add(label + "_calls", 1)
            self._pending.popleft()

    def collect(self, wait=False):
        if self.enabled:
            with self._lock:
                self._collect(wait)

    def snapshot(self):
        with self._lock:
            return dict(self._totals)


def install_model_timing():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    metrics = CUDASpanMetrics()

    def wrap(original, label):
        def timed(runner, *args, **kwargs):
            output = args[0] if args else kwargs.get("scheduler_output")
            if (
                label == "model_forward"
                and output is not None
                and not output.total_num_scheduled_tokens
            ):
                started = time.perf_counter_ns()
                cpu_started = time.thread_time_ns()
                try:
                    return original(runner, *args, **kwargs)
                finally:
                    cpu = time.thread_time_ns() - cpu_started
                    metrics.add_poll(
                        time.perf_counter_ns() - started,
                        cpu,
                    )
            if label == "sampling" and runner.execute_model_state is None:
                return original(runner, *args, **kwargs)
            stream = torch.cuda.current_stream()
            marker = metrics.start(label, stream)
            try:
                return original(runner, *args, **kwargs)
            finally:
                metrics.finish(marker, stream)

        return timed

    for method, label in (("execute_model", "model_forward"), ("sample_tokens", "sampling")):
        setattr(GPUModelRunner, method, wrap(getattr(GPUModelRunner, method), label))
    return metrics
