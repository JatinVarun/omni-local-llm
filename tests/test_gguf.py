from pathlib import Path

import pytest

from omnillm.core.gguf import GGUFError, read_gguf_metadata, read_gguf_metadata_from_fetch
from tests.gguf_writer import build_gguf, llama_like


def test_reads_scalars_strings_small_arrays_and_counts_tensor_parameters(tmp_path: Path):
    data = build_gguf(
        [
            ("general.architecture", "str", "llama"),
            ("llama.block_count", "u32", 32),
            ("general.parameter_count", "u64", 8_030_261_248),
            ("llama.rope.freq_base", "f32", 500000.0),
            ("some.flag", "bool", True),
            ("openelm.attention.head_count_kv", "arr:i32", [3, 3, 4, 4]),
            ("tokenizer.ggml.tokens", "arr:str", [f"t{i}" for i in range(500)]),
        ],
        tensors=[("a", (4096, 128)), ("b", (10,))],
    )
    path = tmp_path / "m.gguf"
    path.write_bytes(data)

    meta = read_gguf_metadata(path)

    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 32
    assert meta["general.parameter_count"] == 8_030_261_248
    assert meta["llama.rope.freq_base"] == 500000.0
    assert meta["some.flag"] is True
    assert meta["openelm.attention.head_count_kv"] == [3, 3, 4, 4]
    assert "tokenizer.ggml.tokens" not in meta  # large arrays are skipped, not decoded
    assert meta["gguf.tensor_parameter_count"] == 4096 * 128 + 10


def test_remote_read_stops_before_the_tokenizer_and_fetches_only_the_header():
    data = llama_like(vocab=50_000)
    requested: list[tuple[int, int]] = []

    def fetch(start: int, end: int) -> bytes:
        requested.append((start, end))
        return data[start : end + 1]

    meta = read_gguf_metadata_from_fetch(fetch)

    assert meta["llama.attention.head_count_kv"] == 8
    assert "general.quantization_version" not in meta  # stopped at the tokenizer section
    assert len(requested) == 1 and requested[0][1] < len(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (build_gguf([], magic=b"GGML"), "bad magic"),
        (build_gguf([], version=1), "Unsupported GGUF version"),
        (llama_like()[:60], "Unexpected end"),
    ],
)
def test_invalid_data_raises(tmp_path: Path, data: bytes, message: str):
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)

    with pytest.raises(GGUFError, match=message):
        read_gguf_metadata(path)
    with pytest.raises(GGUFError, match=message):
        read_gguf_metadata_from_fetch(lambda start, end: data[start : end + 1])
