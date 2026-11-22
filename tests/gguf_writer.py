"""Writes minimal GGUF files for tests."""

import struct
from typing import Any

_TYPES = {"u32": (4, "<I"), "u64": (10, "<Q"), "f32": (6, "<f"), "bool": (7, "<?"), "i32": (5, "<i")}


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(kind: str, value: Any) -> bytes:
    if kind == "str":
        return struct.pack("<I", 8) + _string(value)
    if kind.startswith("arr:"):
        item_kind = kind[4:]
        if item_kind == "str":
            body = b"".join(_string(v) for v in value)
            return struct.pack("<IIQ", 9, 8, len(value)) + body
        type_id, fmt = _TYPES[item_kind]
        return struct.pack("<IIQ", 9, type_id, len(value)) + b"".join(struct.pack(fmt, v) for v in value)
    type_id, fmt = _TYPES[kind]
    return struct.pack("<I", type_id) + struct.pack(fmt, value)


def build_gguf(
    kvs: list[tuple[str, str, Any]],
    tensors: list[tuple[str, tuple[int, ...]]] = (),  # type: ignore[assignment]
    version: int = 3,
    magic: bytes = b"GGUF",
) -> bytes:
    """kvs: (key, kind, value) with kind in u32/u64/i32/f32/bool/str/arr:<kind>."""
    out = magic + struct.pack("<IQQ", version, len(tensors), len(kvs))
    for key, kind, value in kvs:
        out += _string(key) + _value(kind, value)
    for name, dims in tensors:
        out += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", 0, 0)
    return out


def llama_like(arch: str = "llama", *, vocab: int = 2000, extra: list[tuple[str, str, Any]] | None = None) -> bytes:
    """A llama-3.2-3B-shaped header (28 layers, 24 heads, 8 KV heads, 128 head dim) with a big tokenizer array."""
    kvs = [
        ("general.architecture", "str", arch),
        ("general.name", "str", "Test Model"),
        (f"{arch}.block_count", "u32", 28),
        (f"{arch}.context_length", "u32", 131072),
        (f"{arch}.embedding_length", "u32", 3072),
        (f"{arch}.attention.head_count", "u32", 24),
        (f"{arch}.attention.head_count_kv", "u32", 8),
        (f"{arch}.attention.layer_norm_rms_epsilon", "f32", 1e-5),
        ("general.file_type", "u32", 15),
        *(extra or []),
        ("tokenizer.ggml.tokens", "arr:str", [f"tok{i}" for i in range(vocab)]),
        ("tokenizer.ggml.scores", "arr:f32", [0.0] * vocab),
        ("general.quantization_version", "u32", 2),
    ]
    return build_gguf(kvs, tensors=[("token_embd.weight", (3072, 1000)), ("blk.0.attn_q.weight", (3072, 3072))])
