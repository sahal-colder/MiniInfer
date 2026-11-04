#!/usr/bin/env python3
"""Local vLLM engine, one request at a time, timestamps from incremental engine.step()."""
import argparse
import json
from pathlib import Path
import time


def main(args):
    import torch
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 77
    torch.cuda.set_device(0)
    ballast = torch.empty(args.ballast_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
    ballast.zero_()
    torch.cuda.synchronize()
    prompt = [int(token) for token in args.prompt_file.read_text().split()]
    start = time.perf_counter()
    options = dict(kv_cache_memory_bytes=args.kv_cache_bytes) if args.kv_cache_bytes else {}
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(args.model), tokenizer=str(args.model), skip_tokenizer_init=True,
        dtype="float32", kv_cache_dtype="auto", tensor_parallel_size=1, max_num_seqs=1,
        max_model_len=args.context, gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True, enforce_eager=args.eager, seed=args.seed,
        enable_chunked_prefill=False, max_num_batched_tokens=args.context,
        **options,
    ))
    init_ms = (time.perf_counter() - start) * 1000

    def request(tokens, count, request_id):
        params = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False)
        started = time.perf_counter()
        engine.add_request(request_id, {"prompt_token_ids": tokens}, params)
        timestamps = []
        seen = 0
        while engine.has_unfinished_requests():
            for output in engine.step():
                now = time.perf_counter()
                if output.request_id != request_id or not output.outputs:
                    continue
                current = output.outputs[0].token_ids
                if len(current) > seen:
                    # A coalesced engine output is recorded explicitly, not presented as per-token ITL.
                    timestamps.extend([now] * (len(current) - seen))
                    seen = len(current)
                final = output
        if seen != count:
            raise RuntimeError(f"Expected {count} tokens, received {seen}")
        intervals = [(right - left) * 1000 for left, right in zip(timestamps, timestamps[1:])]
        return dict(ttft_ms=(timestamps[0] - started) * 1000,
                    tpot_ms=(timestamps[-1] - timestamps[0]) * 1000 / (seen - 1),
                    request_ms=(timestamps[-1] - started) * 1000, prompt_tokens=len(tokens),
                    output_tokens=seen, tokens=list(final.outputs[0].token_ids),
                    prefix_hit_tokens=getattr(final, "num_cached_tokens", None),
                    decode_samples_ms=intervals, coalesced_tokens=sum(value == 0 for value in intervals),
                    ssd_read_bytes=None, prefill_ssd_read_bytes=None,
                    ballast_bytes=ballast.numel())

    warm_prompt = list(prompt)
    warm_prompt[0] = (warm_prompt[0] + 1) % args.vocab
    for index in range(args.warmup):
        request(warm_prompt, 2, f"warmup-{index}")
    results = []
    for phase in ["miss", "repeat"]:
        row = request(prompt, args.generate, phase)
        results.append(dict(phase=phase, **row))
    if results[0]["tokens"] != results[1]["tokens"]:
        raise RuntimeError("vLLM repeated prefix changed generated tokens")
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        model_init_ms=init_ms, vllm_version=vllm.__version__, torch_version=torch.__version__,
        torch_cuda=torch.version.cuda, requests=results), indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ballast-mib", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-bytes", type=int, default=0)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.generate < 2:
        parser.error("--generate must be at least 2")
    try:
        raise SystemExit(main(args))
    except ModuleNotFoundError as error:
        print(f"Dependency unavailable: {error}")
        raise SystemExit(77)
