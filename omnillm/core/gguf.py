"""Minimal reader for GGUF file headers (metadata key/values), for local files and remote HTTP range reads.

Only the header is read. Large arrays (e.g. the tokenizer vocabulary) are skipped without being decoded and,
for remote files, the reader stops at the tokenizer section so only the first few kilobytes are downloaded.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# value type id -> (struct format, byte size) for fixed-size scalars
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),  # UINT8
    1: ("<b", 1),  # INT8
    2: ("<H", 2),  # UINT16
    3: ("<h", 2),  # INT16
    4: ("<I", 4),  # UINT32
    5: ("<i", 4),  # INT32
    6: ("<f", 4),  # FLOAT32
    7: ("<?", 1),  # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}
_STRING = 8
_ARRAY = 9
_MAX_ARRAY_ITEMS_DECODED = 64  # small arrays (e.g. per-layer head counts) are decoded; big ones skipped
_MAX_NESTING = 8  # arrays of arrays; real files use at most 2 levels


class GGUFError(ValueError):
    """Raised when data is not a readable GGUF header."""


class _Source:
    """Sequential reader over a seekable file or a range-fetch callback, with skip support."""

    def __init__(
        self,
        handle: BinaryIO | None = None,
        fetch: Callable[[int, int], bytes] | None = None,
        block_size: int = 256 * 1024,
    ):
        self._handle = handle
        self._fetch = fetch
        self._block_size = block_size
        self._buffer = b""
        self._buffer_start = 0
        self.position = 0

    def read(self, size: int) -> bytes:
        if self._handle is not None:
            data = self._handle.read(size)
            self.position += len(data)
            if len(data) != size:
                raise GGUFError("Unexpected end of GGUF data.")
            return data
        assert self._fetch is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            offset = self.position - self._buffer_start
            if offset < 0 or offset >= len(self._buffer):
                self._buffer_start = self.position
                self._buffer = self._fetch(self.position, self.position + max(self._block_size, remaining) - 1)
                offset = 0
                if not self._buffer:
                    raise GGUFError("Unexpected end of GGUF data.")
            piece = self._buffer[offset : offset + remaining]
            chunks.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(chunks)

    def skip(self, size: int) -> None:
        if self._handle is not None:
            self._handle.seek(size, 1)
        self.position += size

    def unpack(self, fmt: str, size: int) -> Any:
        return struct.unpack(fmt, self.read(size))[0]


def _read_string(source: _Source) -> str:
    length = source.unpack("<Q", 8)
    if length > 1 << 24:
        raise GGUFError("Implausibly long GGUF string; the file is corrupt or not GGUF.")
    return source.read(length).decode("utf-8", errors="replace")


def _read_value(source: _Source, value_type: int, depth: int = 0) -> Any:
    if value_type in _SCALARS:
        fmt, size = _SCALARS[value_type]
        return source.unpack(fmt, size)
    if value_type == _STRING:
        return _read_string(source)
    if value_type == _ARRAY:
        if depth >= _MAX_NESTING:
            raise GGUFError("GGUF arrays are nested too deeply; the file is corrupt or malicious.")
        item_type = source.unpack("<I", 4)
        count = source.unpack("<Q", 8)
        if count <= _MAX_ARRAY_ITEMS_DECODED:
            return [_read_value(source, item_type, depth + 1) for _ in range(count)]
        if item_type in _SCALARS:
            source.skip(count * _SCALARS[item_type][1])
        elif item_type == _STRING:
            for _ in range(count):
                source.skip(source.unpack("<Q", 8))
        else:
            for _ in range(count):
                _read_value(source, item_type, depth + 1)
        return None  # large array: skipped
    raise GGUFError(f"Unknown GGUF value type {value_type}.")


def _parse(source: _Source, *, stop_at_tokenizer: bool, read_tensors: bool = False) -> dict[str, Any]:
    if source.read(4) != GGUF_MAGIC:
        raise GGUFError("Not a GGUF file (bad magic).")
    version = source.unpack("<I", 4)
    if version not in (2, 3):
        raise GGUFError(f"Unsupported GGUF version {version}.")
    tensor_count = source.unpack("<Q", 8)
    kv_count = source.unpack("<Q", 8)
    metadata: dict[str, Any] = {"gguf.version": version, "gguf.tensor_count": tensor_count}
    for _ in range(kv_count):
        key = _read_string(source)
        if stop_at_tokenizer and key.startswith("tokenizer.") and "general.architecture" in metadata:
            break
        value_type = source.unpack("<I", 4)
        value = _read_value(source, value_type)
        if value is not None:
            metadata[key] = value
    if read_tensors:
        # Tensor infos follow the metadata: name, n_dims, dims[n_dims], ggml type, data offset.
        total = 0
        for _ in range(tensor_count):
            _read_string(source)
            n_dims = source.unpack("<I", 4)
            elements = 1
            for _ in range(n_dims):
                elements *= source.unpack("<Q", 8)
            source.skip(4 + 8)
            total += elements
        metadata["gguf.tensor_parameter_count"] = total
    return metadata


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Reads the metadata key/values of a local GGUF file."""
    with open(path, "rb") as handle:
        return _parse(_Source(handle=handle), stop_at_tokenizer=False, read_tensors=True)


def read_gguf_metadata_from_fetch(fetch: Callable[[int, int], bytes]) -> dict[str, Any]:
    """Reads GGUF metadata through `fetch(start, end_inclusive) -> bytes`, e.g. HTTP Range requests."""
    return _parse(_Source(fetch=fetch), stop_at_tokenizer=True)
