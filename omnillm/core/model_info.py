"""Model architecture and quantization metadata from real sources, used by the hardware fit analysis.

Sources, most to least exact:

* Hugging Face GGUF repo: real per-quant file sizes (tree API) + architecture from a GGUF header (HTTP range read)
* Hugging Face safetensors repo: architecture from ``config.json`` + exact parameter count
* Installed Ollama model: GGUF metadata via ``ollama show`` + installed size
* Local ``.gguf`` file: header + file size
* Name heuristics ("8b", "30b-a3b"): typical architecture for the size
"""

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omnillm.core.errors import BackendUnavailableError, InvalidRequestError
from omnillm.core.gguf import GGUFError, read_gguf_metadata, read_gguf_metadata_from_fetch

GIB = 1024**3

#: Approximate bits per weight of llama.cpp quantization types (incl. per-block scales).
QUANT_BPW: dict[str, float] = {
    "F32": 32.0,
    "BF16": 16.0,
    "F16": 16.0,
    "FP16": 16.0,
    "Q8_0": 8.5,
    "Q8_K": 8.5,
    "Q6_K": 6.56,
    "Q5_1": 6.0,
    "Q5_K_M": 5.69,
    "Q5_K_S": 5.54,
    "Q5_0": 5.5,
    "Q4_1": 5.0,
    "Q4_K_M": 4.85,
    "Q4_K_S": 4.58,
    "Q4_0": 4.55,
    "IQ4_NL": 4.5,
    "Q3_K_L": 4.27,
    "IQ4_XS": 4.25,
    "MXFP4": 4.25,
    "Q3_K_M": 3.91,
    "IQ3_M": 3.66,
    "Q3_K_S": 3.5,
    "IQ3_S": 3.44,
    "IQ3_XS": 3.3,
    "Q2_K": 3.2,
    "IQ3_XXS": 3.06,
    "Q2_K_S": 2.9,
    "IQ2_M": 2.7,
    "IQ2_S": 2.5,
    "IQ2_XS": 2.31,
    "IQ2_XXS": 2.06,
    "TQ2_0": 2.06,
    "IQ1_M": 1.75,
    "TQ1_0": 1.69,
    "IQ1_S": 1.56,
}

#: llama.cpp `general.file_type` (LLAMA_FTYPE_*) -> quantization label.
GGUF_FILE_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4",
}

_QUANT_IN_NAME = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ[1-4]_(?:XXS|XS|NL|S|M)|Q[2-8]_K(?:_(?:XXL|XL|L|M|S))?|Q[4-8]_[01](?:_[48]_[48])?|"
    r"BF16|F16|F32|MXFP4|TQ[12]_0))(?=[-_.]|$)"
)
_SPLIT_SUFFIX = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_MOE_SHARED_FRACTION = 0.05  # rough share of always-active (non-expert) weights in MoE models


def quant_bpw(label: str | None) -> float | None:
    """Bits per weight for a quant label; 'UD-'/dynamic variants map to their base type."""
    if not label:
        return None
    base = label.upper().removeprefix("UD-")
    base = re.sub(r"^(Q4_0)_[48]_[48]$", r"\1", base)  # ARM-repacked Q4_0 variants
    if base in QUANT_BPW:
        return QUANT_BPW[base]
    # "_L"/"_XL" variants keep embeddings/output at higher precision: close to the base type's medium/plain size.
    stem = re.sub(r"_(XXL|XL|L)$", "", base)
    return QUANT_BPW.get(f"{stem}_M") or QUANT_BPW.get(stem)


def quant_quality(label: str) -> str:
    bpw = quant_bpw(label) or 0.0
    if bpw >= 15:
        return "Full precision"
    if bpw >= 8:
        return "Near lossless"
    if bpw >= 6:
        return "Excellent"
    if bpw >= 5.4:
        return "Very high"
    if bpw >= 4.4:
        return "High (sweet spot)"
    if bpw >= 3.6:
        return "Good"
    if bpw >= 2.9:
        return "Noticeable loss"
    return "Heavy loss"


def quant_label_from_filename(filename: str) -> str | None:
    name = Path(filename).name
    matches = _QUANT_IN_NAME.findall(_SPLIT_SUFFIX.sub(".gguf", name).removesuffix(".gguf"))
    return matches[-1].upper() if matches else None


# --------------------------------------------------------------------------- architecture


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """The facts that determine memory use: layer count and per-token KV-cache size."""

    n_layers: int
    n_kv_heads: int
    key_length: int
    value_length: int
    n_embd: int | None = None
    n_heads: int | None = None
    context_length: int | None = None
    param_count: int | None = None
    active_param_count: int | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    name: str | None = None
    exact: bool = True

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_count and self.expert_count > 1) or (
            self.active_param_count is not None
            and self.param_count is not None
            and self.active_param_count < self.param_count * 0.8
        )

    def kv_bytes_per_token(self, bytes_per_element: float = 2.0) -> float:
        return self.n_layers * self.n_kv_heads * (self.key_length + self.value_length) * bytes_per_element

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "is_moe": self.is_moe}

    def describe(self) -> str:
        parts = []
        if self.param_count:
            parts.append(f"{self.param_count / 1e9:.1f}B params")
        if self.is_moe and self.active_param_count:
            parts.append(f"MoE ~{self.active_param_count / 1e9:.1f}B active")
        parts.append(f"{self.n_layers} layers")
        if self.n_heads and self.n_kv_heads and self.n_kv_heads != self.n_heads:
            parts.append(f"GQA {self.n_heads}/{self.n_kv_heads} heads")
        if self.context_length:
            parts.append(f"trained ctx {self.context_length:,}")
        if not self.exact:
            parts.append("architecture estimated")
        return ", ".join(parts)


def _first_int(value: Any) -> int | None:
    if isinstance(value, list):  # per-layer arrays (e.g. OpenELM): size for the worst layer
        numbers = [v for v in value if isinstance(v, int)]
        return max(numbers) if numbers else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _active_params(total: int | None, experts: int | None, used: int | None) -> int | None:
    if not total or not experts or experts <= 1 or not used:
        return None
    return int(total * (_MOE_SHARED_FRACTION + (1 - _MOE_SHARED_FRACTION) * used / experts))


def architecture_from_gguf(metadata: Mapping[str, Any]) -> ModelArchitecture | None:
    """Builds an architecture from GGUF metadata keys (also what `ollama show` returns as model_info)."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str):
        return None

    def get(key: str) -> int | None:
        return _first_int(metadata.get(f"{arch}.{key}"))

    n_layers = get("block_count")
    n_embd = get("embedding_length")
    n_heads = get("attention.head_count")
    if not n_layers or not n_heads:
        return None
    n_kv_heads = get("attention.head_count_kv") or n_heads
    head_dim = get("attention.key_length") or (n_embd // n_heads if n_embd else 128)
    value_length = get("attention.value_length") or head_dim
    kv_lora_rank = get("attention.kv_lora_rank")
    if kv_lora_rank:  # MLA (DeepSeek-V2/V3): compressed latent + rope part cached per layer
        n_kv_heads, head_dim, value_length = 1, kv_lora_rank + (get("rope.dimension_count") or 64), 0
    params = _first_int(metadata.get("general.parameter_count")) or _first_int(
        metadata.get("gguf.tensor_parameter_count")
    )
    experts, used = get("expert_count"), get("expert_used_count")
    return ModelArchitecture(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        key_length=head_dim,
        value_length=value_length,
        n_embd=n_embd,
        n_heads=n_heads,
        context_length=get("context_length"),
        param_count=params,
        active_param_count=_active_params(params, experts, used),
        expert_count=experts,
        expert_used_count=used,
        name=arch,
    )


def architecture_from_hf_config(config: Mapping[str, Any], param_count: int | None = None) -> ModelArchitecture | None:
    """Builds an architecture from a transformers ``config.json``."""
    text_config = config.get("text_config")
    cfg: Mapping[str, Any] = text_config if isinstance(text_config, dict) else config
    n_layers = _first_int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers"))
    n_embd = _first_int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model"))
    n_heads = _first_int(cfg.get("num_attention_heads") or cfg.get("n_head"))
    if not n_layers or not n_heads:
        return None
    n_kv_heads = _first_int(cfg.get("num_key_value_heads")) or n_heads
    head_dim = _first_int(cfg.get("head_dim")) or (n_embd // n_heads if n_embd else 128)
    value_length = head_dim
    kv_lora_rank = _first_int(cfg.get("kv_lora_rank"))
    if kv_lora_rank:
        n_kv_heads, head_dim, value_length = 1, kv_lora_rank + (_first_int(cfg.get("qk_rope_head_dim")) or 64), 0
    experts = _first_int(cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts"))
    used = _first_int(cfg.get("num_experts_per_tok"))
    model_type = cfg.get("model_type") or config.get("model_type")
    return ModelArchitecture(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        key_length=head_dim,
        value_length=value_length,
        n_embd=n_embd,
        n_heads=n_heads,
        context_length=_first_int(cfg.get("max_position_embeddings")),
        param_count=param_count,
        active_param_count=_active_params(param_count, experts, used),
        expert_count=experts,
        expert_used_count=used,
        name=model_type if isinstance(model_type, str) else None,
    )


# Typical dense decoder shapes by size (layers, hidden, kv heads, head dim) for name-only estimates.
_TYPICAL_SHAPES: tuple[tuple[float, int, int, int, int], ...] = (
    (0.5, 24, 1024, 4, 64),
    (1.5, 16, 2048, 8, 64),
    (4.5, 28, 3072, 8, 128),
    (9.5, 32, 4096, 8, 128),
    (16.0, 40, 5120, 8, 128),
    (40.0, 64, 5120, 8, 128),
    (80.0, 80, 8192, 8, 128),
    (float("inf"), 126, 16384, 8, 128),
)


def estimated_architecture(param_count_b: float, active_param_count_b: float | None = None) -> ModelArchitecture:
    _, layers, hidden, kv_heads, head_dim = next(shape for shape in _TYPICAL_SHAPES if param_count_b <= shape[0])
    return ModelArchitecture(
        n_layers=layers,
        n_kv_heads=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        n_embd=hidden,
        n_heads=hidden // head_dim,
        param_count=int(param_count_b * 1e9),
        active_param_count=int(active_param_count_b * 1e9) if active_param_count_b else None,
        exact=False,
    )


def parse_active_parameter_count(model_name: str) -> float | None:
    """Active params of MoE names like 'qwen3-30b-a3b' -> 3.0."""
    match = re.search(r"(?:^|[-_:])a(\d+(?:\.\d+)?)b(?:[-_:]|$)", model_name.lower())
    return float(match.group(1)) if match else None


# ------------------------------------------------------------------------------ model info


@dataclass(slots=True)
class QuantOption:
    """One downloadable/installed quantization of a model."""

    label: str
    size_bytes: int | None = None
    files: tuple[str, ...] = ()
    installed: bool = False

    @property
    def bpw(self) -> float | None:
        return quant_bpw(self.label)


@dataclass(slots=True)
class ModelInfo:
    name: str
    architecture: ModelArchitecture
    source: str
    quants: list[QuantOption] = field(default_factory=list)
    repo: str | None = None
    revision: str = "main"
    notes: list[str] = field(default_factory=list)


SOURCE_DESCRIPTIONS = {
    "huggingface-gguf": "Hugging Face GGUF repo (exact file sizes + GGUF header)",
    "huggingface": "Hugging Face model metadata (no GGUF files; quant sizes estimated)",
    "ollama": "installed Ollama model (GGUF metadata; other quant sizes estimated)",
    "gguf-file": "local GGUF file (exact; * marks the file itself)",
    "estimated": "model name heuristics (estimated)",
}


def gguf_file_info(path: str | Path) -> ModelInfo:
    file_path = Path(path).expanduser()
    try:
        metadata = read_gguf_metadata(file_path)
    except GGUFError as error:
        raise InvalidRequestError(f"{file_path}: {error}") from error
    arch = architecture_from_gguf(metadata)
    if arch is None:
        raise InvalidRequestError(f"{file_path}: GGUF header lacks the architecture keys needed for a fit analysis.")
    label = quant_label_from_filename(file_path.name) or GGUF_FILE_TYPES.get(
        _first_int(metadata.get("general.file_type")) or -1, "GGUF"
    )
    return ModelInfo(
        name=file_path.name,
        architecture=arch,
        source="gguf-file",
        quants=[QuantOption(label, file_path.stat().st_size, (str(file_path),), installed=True)],
    )


def ollama_model_info(model: str, client: Any) -> ModelInfo | None:
    """Reads an installed Ollama model's GGUF metadata; returns None when it isn't installed."""
    try:
        shown = client.show(model)
    except Exception as error:
        if isinstance(error, ConnectionError) or getattr(error, "status_code", None) == 404:
            return None
        if type(error).__name__ in {"ResponseError", "ConnectError"}:
            return None
        raise
    model_info = getattr(shown, "modelinfo", None) or getattr(shown, "model_info", None) or {}
    if isinstance(shown, Mapping):
        model_info = shown.get("model_info") or shown.get("modelinfo") or {}
    arch = architecture_from_gguf(model_info)
    if arch is None:
        return None
    details = getattr(shown, "details", None)
    label = getattr(details, "quantization_level", None) or GGUF_FILE_TYPES.get(
        _first_int(model_info.get("general.file_type")) or -1, "unknown"
    )
    size: int | None = None
    try:
        listed = client.list()
        for entry in getattr(listed, "models", None) or []:
            if getattr(entry, "model", None) in (model, f"{model}:latest"):
                size = getattr(entry, "size", None)
    except Exception:
        size = None
    return ModelInfo(
        name=model,
        architecture=arch,
        source="ollama",
        quants=[QuantOption(str(label).upper(), size, installed=True)],
    )


def estimated_model_info(name: str, param_count_b: float | None = None) -> ModelInfo:
    from omnillm.core.hardware import parse_model_parameter_count

    notes: list[str] = []
    total = param_count_b or parse_model_parameter_count(name)
    if total is None:
        total = 7.0
        notes.append("Could not infer the parameter count from the name; assuming 7B. Pass --params to override.")
    notes.append(
        "Estimated from the model name. For exact numbers pass a Hugging Face repo (e.g. 'unsloth/Qwen3-8B-GGUF'), "
        "an installed Ollama model, or a .gguf file."
    )
    return ModelInfo(
        name=name,
        architecture=estimated_architecture(total, parse_active_parameter_count(name)),
        source="estimated",
        notes=notes,
    )


# ----------------------------------------------------------------------------- Hugging Face


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def make_hf_client(token: str | None = None, timeout: float = 20.0) -> Any:
    try:
        import httpx
    except ImportError as error:
        raise BackendUnavailableError("Hugging Face lookups need httpx: pip install httpx") from error
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=hf_endpoint(), headers=headers, timeout=timeout, follow_redirects=True)


def _hf_get_json(client: Any, path: str, what: str | None = None) -> Any:
    try:
        response = client.get(path)
    except Exception as error:
        raise BackendUnavailableError(f"Cannot reach Hugging Face ({hf_endpoint()}).") from error
    target = what or path
    if response.status_code in (401, 403, 404):
        # The Hub answers 401 for repos that don't exist as well as for private/gated ones.
        raise InvalidRequestError(
            f"Hugging Face: {target} was not found or is private/gated (check the name; set HF_TOKEN if you have access)."
        )
    if response.status_code >= 400:
        raise BackendUnavailableError(f"Hugging Face returned HTTP {response.status_code} for {path}.")
    return response.json()


def list_hf_gguf_quants(repo: str, *, revision: str = "main", client: Any) -> list[QuantOption]:
    """Lists a repo's GGUF quantizations with real sizes; split files are grouped and summed."""
    entries = _hf_get_json(client, f"/api/models/{repo}/tree/{revision}?recursive=true", f"'{repo}'")
    groups: dict[str, QuantOption] = {}
    for entry in entries if isinstance(entries, list) else []:
        path = entry.get("path", "")
        if entry.get("type") != "file" or not path.lower().endswith(".gguf") or "mmproj" in path.lower():
            continue
        size = (entry.get("lfs") or {}).get("size") or entry.get("size")
        label = quant_label_from_filename(path) or Path(path).stem
        key = _SPLIT_SUFFIX.sub("", path)
        option = groups.get(key)
        if option is None:
            groups[key] = QuantOption(label, size, (path,))
        else:
            option.files = tuple(sorted((*option.files, path)))
            option.size_bytes = (option.size_bytes or 0) + (size or 0)
    return sorted(groups.values(), key=lambda q: q.size_bytes or 0, reverse=True)


def read_hf_gguf_metadata(repo: str, path: str, *, revision: str = "main", client: Any) -> dict[str, Any]:
    """Reads a remote GGUF header with HTTP range requests (only the first few KB-MB are downloaded)."""
    url = f"/{repo}/resolve/{revision}/{path}"

    def fetch(start: int, end: int) -> bytes:
        wanted = end - start + 1
        try:
            with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
                if response.status_code >= 400:
                    raise InvalidRequestError(f"Cannot read {repo}/{path}: HTTP {response.status_code}")
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) >= wanted:  # a server ignoring Range (HTTP 200) won't make us download GBs
                        break
                return bytes(data[:wanted])
        except InvalidRequestError:
            raise
        except Exception as error:
            raise BackendUnavailableError(f"Cannot reach Hugging Face ({hf_endpoint()}).") from error

    try:
        return read_gguf_metadata_from_fetch(fetch)
    except GGUFError as error:
        raise InvalidRequestError(f"{repo}/{path}: {error}") from error


def hf_model_info(repo: str, *, revision: str = "main", client: Any | None = None) -> ModelInfo:
    """Fit metadata for a Hugging Face repo: GGUF repos give exact sizes, others give exact architecture."""
    owns_client = client is None
    client = client or make_hf_client()
    try:
        quants = list_hf_gguf_quants(repo, revision=revision, client=client)
        if quants:
            smallest = min(quants, key=lambda q: q.size_bytes or 0)
            metadata = read_hf_gguf_metadata(repo, smallest.files[0], revision=revision, client=client)
            arch = architecture_from_gguf(metadata)
            if arch is None:
                raise InvalidRequestError(f"{repo}: GGUF header lacks architecture keys.")
            if arch.param_count is None:
                total = _hf_gguf_param_count(repo, client)
                if total:
                    arch = _with_params(arch, total)
            if arch.param_count is None:
                reference = next((q for label in ("BF16", "F16", "Q8_0") for q in quants if q.label == label), smallest)
                bpw = reference.bpw
                smallest = reference
                if bpw and smallest.size_bytes:
                    estimated_params = int(smallest.size_bytes * 8 / bpw)
                    arch = _with_params(arch, estimated_params)
            return ModelInfo(
                name=repo, architecture=arch, source="huggingface-gguf", quants=quants, repo=repo, revision=revision
            )

        info = _hf_get_json(client, f"/api/models/{repo}?expand%5B%5D=safetensors")
        total = ((info or {}).get("safetensors") or {}).get("total")
        params: int | None = int(total) if isinstance(total, (int, float)) else None
        notes: list[str] = []
        try:
            config = _hf_get_json(client, f"/{repo}/resolve/{revision}/config.json", f"{repo}/config.json")
            arch = architecture_from_hf_config(config, params)
        except InvalidRequestError:
            arch = None
            notes.append("config.json is not accessible (gated repo?); architecture estimated from the size.")
        if arch is None:
            if not params:
                raise InvalidRequestError(f"{repo}: no GGUF files and no parameter count found.")
            arch = estimated_architecture(params / 1e9, parse_active_parameter_count(repo))
        notes.append(
            f"{repo} has no GGUF files; sizes are estimates. Search Hugging Face for '{repo.split('/')[-1]} GGUF' to download one."
        )
        return ModelInfo(name=repo, architecture=arch, source="huggingface", repo=repo, revision=revision, notes=notes)
    finally:
        if owns_client:
            client.close()


def _hf_gguf_param_count(repo: str, client: Any) -> int | None:
    try:
        info = _hf_get_json(client, f"/api/models/{repo}?expand%5B%5D=gguf")
    except (InvalidRequestError, BackendUnavailableError):
        return None
    total = ((info or {}).get("gguf") or {}).get("total")
    return int(total) if isinstance(total, (int, float)) and total > 0 else None


def _with_params(arch: ModelArchitecture, params: int) -> ModelArchitecture:
    from dataclasses import replace

    return replace(
        arch, param_count=params, active_param_count=_active_params(params, arch.expert_count, arch.expert_used_count)
    )


def download_hf_files(
    repo: str, files: tuple[str, ...], *, revision: str = "main", progress: Callable[[str], None] | None = None
) -> Path:
    """Downloads GGUF file(s) into the Hugging Face cache (shared with the llama.cpp backend); returns the first."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise BackendUnavailableError(
            "Downloading needs huggingface-hub: pip install 'omni-local-llm[llama-cpp]'"
        ) from error
    paths = []
    for name in files:
        if progress:
            progress(name)
        paths.append(Path(hf_hub_download(repo_id=repo, filename=name, revision=revision)))
    return paths[0]
