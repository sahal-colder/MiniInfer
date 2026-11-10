#!/usr/bin/env python3
"""Create paired FP32 Llama checkpoints; no downloads or implicit weight conversion."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct

FIELDS = ("dim", "hidden_dim", "layer_num", "head_num", "kv_head_num", "vocab_size", "seq_len")


def layout(config):
    dim, hidden, layers, heads, kv_heads, vocab, context = (config[key] for key in FIELDS)
    if min(dim, hidden, layers, heads, kv_heads, context) <= 0 or not vocab or dim % heads or heads % kv_heads:
        raise ValueError("Invalid legacy FP32 model geometry")
    head_dim = dim // heads
    if head_dim % 2:
        raise ValueError("Invalid legacy FP32 model geometry")
    entries = []
    cursor = 28

    def add(name, shape):
        nonlocal cursor
        size = math.prod(shape) * 4
        entries.append(dict(name=name, shape=shape, offset=cursor, bytes=size))
        cursor += size

    add("tok_embeddings.weight", [abs(vocab), dim])
    for layer in range(layers):
        add(f"layers.{layer}.attention_norm.weight", [dim])
    for name, rows in (("wq", dim), ("wk", kv_heads * head_dim), ("wv", kv_heads * head_dim), ("wo", dim)):
        for layer in range(layers):
            add(f"layers.{layer}.attention.{name}.weight", [rows, dim])
    for layer in range(layers):
        add(f"layers.{layer}.ffn_norm.weight", [dim])
    for name, shape in (("w1", [hidden, dim]), ("w2", [dim, hidden]), ("w3", [hidden, dim])):
        for layer in range(layers):
            add(f"layers.{layer}.feed_forward.{name}.weight", shape)
    add("norm.weight", [dim])
    cursor += context * head_dim * 4  # Two legacy RoPE tables, unused by native forward.
    if vocab < 0:
        add("output.weight", [abs(vocab), dim])
    return entries, cursor


def generate_legacy(path, config, seed):
    entries, size = layout(config)
    rng = random.Random(seed)
    with path.open("xb") as stream:
        stream.write(struct.pack("<7i", *(config[key] for key in FIELDS)))
        for entry in entries:
            if stream.tell() != entry["offset"]:
                stream.write(b"\0" * (entry["offset"] - stream.tell()))
            count = entry["bytes"] // 4
            norm = "norm" in entry["name"]
            scale = 0.4 if entry["name"] == "tok_embeddings.weight" else 0.25 / math.sqrt(config["dim"])
            for start in range(0, count, 65536):
                amount = min(65536, count - start)
                values = [1 + rng.uniform(-0.05, 0.05) if norm else rng.uniform(-scale, scale)
                          for _ in range(amount)]
                stream.write(struct.pack(f"<{amount}f", *values))
        if stream.tell() < size:
            stream.write(b"\0" * (size - stream.tell()))
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_name(name):
    if name == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    return "model." + name.replace("attention_norm", "input_layernorm").replace(
        "ffn_norm", "post_attention_layernorm").replace("attention.wq", "self_attn.q_proj").replace(
        "attention.wk", "self_attn.k_proj").replace("attention.wv", "self_attn.v_proj").replace(
        "attention.wo", "self_attn.o_proj").replace("feed_forward.w1", "mlp.gate_proj").replace(
        "feed_forward.w2", "mlp.down_proj").replace("feed_forward.w3", "mlp.up_proj")


def prepare(destination, mock, context, hf):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    native = destination / "model.bin"
    if mock:
        config = dict(dim=128, hidden_dim=256, layer_num=4, head_num=4,
                      kv_head_num=2, vocab_size=-256, seq_len=context)
        generate_legacy(native, config, seed=42)
        hf = destination / "hf"
        hf.mkdir()
        entries, _ = layout(config)
        tensors = []
        with native.open("rb") as stream:
            for entry in entries:
                stream.seek(entry["offset"])
                data = stream.read(entry["bytes"])
                # Native RoPE rotates adjacent pairs; HF uses the two head halves.
                if ".attention.wq." in entry["name"] or ".attention.wk." in entry["name"]:
                    width = config["dim"] * 4
                    head = config["dim"] // config["head_num"]
                    data = b"".join(data[(start + row) * width:(start + row + 1) * width]
                                    for start in range(0, entry["shape"][0], head)
                                    for row in list(range(0, head, 2)) + list(range(1, head, 2)))
                tensors.append((hf_name(entry["name"]), entry["shape"], data))
        header = {"__metadata__": {"format": "pt"}}
        offset = 0
        for name, shape, data in tensors:
            header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, offset + len(data)])
            offset += len(data)
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        with (hf / "model.safetensors").open("xb") as stream:
            stream.write(struct.pack("<Q", len(raw)))
            stream.write(raw)
            for _, _, data in tensors:
                stream.write(data)
        (hf / "config.json").write_text(json.dumps(dict(
            architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=config["dim"],
            intermediate_size=config["hidden_dim"], num_hidden_layers=config["layer_num"],
            num_attention_heads=config["head_num"], num_key_value_heads=config["kv_head_num"],
            vocab_size=abs(config["vocab_size"]), max_position_embeddings=context,
            rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu", tie_word_embeddings=False,
            attention_bias=False, mlp_bias=False, torch_dtype="float32", bos_token_id=1, eos_token_id=2,
        ), indent=2))
    if not mock:
        import torch
        from safetensors import safe_open
        hf = Path(hf).resolve()
        source = json.loads((hf / "config.json").read_text())
        if (source.get("model_type") != "llama" or source.get("rope_scaling")
                or source.get("rope_parameters") or source.get("attention_bias", False)
                or source.get("mlp_bias", False) or source.get("hidden_act", "silu") != "silu"
                or source.get("rms_norm_eps") != 1e-5 or source.get("rope_theta", 10000) != 10000
                or source.get("head_dim", source["hidden_size"] // source["num_attention_heads"])
                != source["hidden_size"] // source["num_attention_heads"]):
            raise ValueError("Only bias-free Llama with native head_dim, SiLU, eps=1e-5, theta=10000, no RoPE scaling is supported")
        config = dict(dim=source["hidden_size"], hidden_dim=source["intermediate_size"],
                      layer_num=source["num_hidden_layers"], head_num=source["num_attention_heads"],
                      kv_head_num=source.get("num_key_value_heads", source["num_attention_heads"]),
                      vocab_size=-source["vocab_size"], seq_len=source["max_position_embeddings"])
        entries, size = layout(config)
        files = sorted(hf.glob("*.safetensors"))
        index = {}
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as stream:
                index.update({key: file for key in stream.keys()})
        if not index:
            raise ValueError("Local HF safetensors required")
        with native.open("xb") as stream:
            stream.write(struct.pack("<7i", *(config[key] for key in FIELDS)))
            for entry in entries:
                if stream.tell() != entry["offset"]:
                    stream.write(b"\0" * (entry["offset"] - stream.tell()))
                name = hf_name(entry["name"])
                if name == "lm_head.weight" and source.get("tie_word_embeddings"):
                    name = "model.embed_tokens.weight"
                with safe_open(index[name], framework="pt", device="cpu") as tensor_file:
                    tensor = tensor_file.get_tensor(name).to(torch.float32)
                if list(tensor.shape) != entry["shape"]:
                    raise ValueError(f"Shape mismatch: {name}")
                if name.endswith(("q_proj.weight", "k_proj.weight")):
                    head = config["dim"] // config["head_num"]
                    tensor = tensor.reshape(-1, 2, head // 2, config["dim"]).transpose(1, 2).reshape(entry["shape"])
                stream.write(tensor.contiguous().numpy().tobytes())
            if stream.tell() < size:
                stream.write(b"\0" * (size - stream.tell()))
    weights = sorted(Path(hf).glob("*.safetensors"))
    manifest = dict(mock=mock, native=str(native), hf=str(hf), config=config,
                    dtype="float32", native_sha256=sha256(native),
                    hf_sha256={str(file): sha256(file) for file in [Path(hf) / "config.json", *weights]},
                    rope_conversion="native interleaved Q/K <-> HF half-split Q/K")
    (destination / "model_pair.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination / "model_pair.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mock", action="store_true")
    source.add_argument("--hf", type=Path, help="Local compatible HF Llama directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing weights are never overwritten")
    parser.add_argument("--context", type=int, default=4352)
    args = parser.parse_args()
    print(prepare(args.output, args.mock, args.context, args.hf))
