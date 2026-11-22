import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from omnillm.core.errors import InvalidRequestError
from omnillm.core.hardware import estimate_kv_cache_gb
from omnillm.core.model_info import (
    architecture_from_gguf,
    architecture_from_hf_config,
    estimated_model_info,
    gguf_file_info,
    hf_model_info,
    list_hf_gguf_quants,
    ollama_model_info,
    quant_bpw,
    quant_label_from_filename,
)
from tests.gguf_writer import llama_like

LLAMA_32_3B = {  # as returned by `ollama show llama3.2:3b`
    "general.architecture": "llama",
    "general.parameter_count": 3212749888,
    "llama.attention.head_count": 24,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
    "llama.block_count": 28,
    "llama.context_length": 131072,
    "llama.embedding_length": 3072,
}

QWEN3_30B_A3B_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}


@pytest.mark.parametrize(
    ("filename", "label"),
    [
        ("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
        ("Qwen3-30B-A3B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf", "BF16"),
        ("Llama-3.2-3B-Instruct-Q4_0_4_8.gguf", "Q4_0_4_8"),
        ("model.IQ3_XXS.gguf", "IQ3_XXS"),
        ("qwen2.5-7b-instruct-q5_k_m.gguf", "Q5_K_M"),
        ("model.gguf", None),
    ],
)
def test_quant_label_from_filename(filename: str, label: str | None):
    assert quant_label_from_filename(filename) == label


def test_quant_bpw_maps_variants_to_base_types():
    assert quant_bpw("UD-Q4_K_XL") == quant_bpw("Q4_K_M") == 4.85
    assert quant_bpw("Q6_K_L") == quant_bpw("Q6_K")
    assert quant_bpw("Q4_0_4_4") == quant_bpw("Q4_0")
    assert quant_bpw("UD-Q8_K_XL") == 8.5
    assert quant_bpw("WEIRD") is None


def test_gguf_architecture_gives_exact_gqa_kv_cache():
    arch = architecture_from_gguf(LLAMA_32_3B)

    assert arch is not None
    assert (arch.n_layers, arch.n_heads, arch.n_kv_heads, arch.key_length) == (28, 24, 8, 128)
    # 2 (K+V) x 28 layers x 8 KV heads x 128 dims x 2 bytes x 8192 tokens = 0.875 GiB
    assert estimate_kv_cache_gb(arch, 8192, "f16") == pytest.approx(0.875)
    assert estimate_kv_cache_gb(arch, 8192, "q8_0") == pytest.approx(0.875 * 1.0625 / 2)
    assert not arch.is_moe


def test_gguf_architecture_defaults_mla_and_per_layer_arrays():
    qwen2 = architecture_from_gguf(
        {
            "general.architecture": "qwen2",
            "qwen2.block_count": 28,
            "qwen2.embedding_length": 3584,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
        }
    )
    assert qwen2 is not None and qwen2.key_length == 128 and qwen2.value_length == 128

    deepseek = architecture_from_gguf(
        {
            "general.architecture": "deepseek2",
            "deepseek2.block_count": 61,
            "deepseek2.embedding_length": 7168,
            "deepseek2.attention.head_count": 128,
            "deepseek2.attention.kv_lora_rank": 512,
            "deepseek2.rope.dimension_count": 64,
            "deepseek2.expert_count": 256,
            "deepseek2.expert_used_count": 8,
            "general.parameter_count": 671_000_000_000,
        }
    )
    assert deepseek is not None
    assert deepseek.kv_bytes_per_token(1) == 61 * (512 + 64)  # compressed latent, not 128 full heads
    assert deepseek.is_moe and deepseek.active_param_count is not None
    assert 20e9 < deepseek.active_param_count < 60e9

    openelm = architecture_from_gguf(
        {
            "general.architecture": "openelm",
            "openelm.block_count": 4,
            "openelm.embedding_length": 1280,
            "openelm.attention.head_count": [12, 12, 16, 20],
            "openelm.attention.head_count_kv": [3, 3, 4, 5],
        }
    )
    assert openelm is not None and openelm.n_kv_heads == 5

    assert architecture_from_gguf({"general.name": "no arch"}) is None


def test_hf_config_architecture_detects_moe_and_nested_text_config():
    arch = architecture_from_hf_config(QWEN3_30B_A3B_CONFIG, param_count=30_532_122_624)

    assert arch is not None
    assert (arch.n_layers, arch.n_kv_heads, arch.key_length, arch.context_length) == (48, 4, 128, 40960)
    assert arch.is_moe
    assert arch.active_param_count == pytest.approx(3.3e9, rel=0.1)

    nested = architecture_from_hf_config({"model_type": "gemma3", "text_config": {**QWEN3_30B_A3B_CONFIG}})
    assert nested is not None and nested.n_layers == 48


def test_estimated_info_reads_moe_active_params_from_name():
    info = estimated_model_info("qwen3-30b-a3b")

    assert info.architecture.param_count == 30_000_000_000
    assert info.architecture.active_param_count == 3_000_000_000
    assert not info.architecture.exact
    assert info.source == "estimated"


def test_local_gguf_file_info(tmp_path: Path):
    path = tmp_path / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    path.write_bytes(llama_like())

    info = gguf_file_info(path)

    assert info.source == "gguf-file"
    assert info.architecture.n_layers == 28
    assert info.architecture.param_count == 3072 * 1000 + 3072 * 3072  # from tensor infos
    assert info.quants[0].label == "Q4_K_M"
    assert info.quants[0].size_bytes == path.stat().st_size


class FakeHub:
    """Mock Hugging Face Hub: tree API, model API, config.json, and GGUF files honouring Range requests."""

    def __init__(
        self,
        files: dict[str, bytes | int],
        *,
        model_api: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        config_status: int = 200,
        tree_status: int = 200,
        honour_range: bool = True,
    ):
        self.files = files
        self.model_api = model_api or {}
        self.config = config
        self.config_status = config_status
        self.tree_status = tree_status
        self.honour_range = honour_range
        self.range_requests: list[str] = []
        self.bytes_served = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/tree/" in path:
            tree = [
                {"type": "file", "path": name, "size": len(v) if isinstance(v, bytes) else v}
                for name, v in self.files.items()
            ]
            return httpx.Response(self.tree_status, json=tree if self.tree_status == 200 else {"error": "x"})
        if path.startswith("/api/models/"):
            return httpx.Response(200, json=self.model_api)
        if path.endswith("/config.json"):
            return httpx.Response(self.config_status, json=self.config or {})
        name = path.split("/resolve/main/", 1)[1]
        data = self.files[name]
        assert isinstance(data, bytes)
        range_header = request.headers.get("range")
        if range_header and self.honour_range:
            self.range_requests.append(range_header)
            start, end = (int(x) for x in range_header.removeprefix("bytes=").split("-"))
            body = data[start : end + 1]
            self.bytes_served += len(body)
            return httpx.Response(206, content=body)

        def stream():
            for i in range(0, len(data), 4096):
                self.bytes_served += len(data[i : i + 4096])
                yield data[i : i + 4096]

        return httpx.Response(200, content=stream())


def hub_client(hub: FakeHub) -> httpx.Client:
    return httpx.Client(base_url="https://hub.test", transport=httpx.MockTransport(hub))


def test_list_hf_gguf_quants_groups_splits_and_skips_mmproj():
    hub = FakeHub(
        {
            "README.md": 100,
            "mmproj-model-f16.gguf": 600,
            "M-Q4_K_M.gguf": 4_000,
            "M-Q8_0.gguf": 8_000,
            "BF16/M-BF16-00001-of-00002.gguf": 10_000,
            "BF16/M-BF16-00002-of-00002.gguf": 6_000,
        }
    )

    quants = list_hf_gguf_quants("org/M-GGUF", client=hub_client(hub))

    assert [(q.label, q.size_bytes, q.files) for q in quants] == [
        ("BF16", 16_000, ("BF16/M-BF16-00001-of-00002.gguf", "BF16/M-BF16-00002-of-00002.gguf")),
        ("Q8_0", 8_000, ("M-Q8_0.gguf",)),
        ("Q4_K_M", 4_000, ("M-Q4_K_M.gguf",)),
    ]


def test_hf_gguf_repo_info_reads_header_by_range_and_exact_param_count():
    header = llama_like(vocab=40_000)
    hub = FakeHub(
        {"M-Q8_0.gguf": 3_400_000_000, "M-Q4_K_M.gguf": header},
        model_api={"gguf": {"total": 3212749888}},
    )

    info = hf_model_info("org/M-GGUF", client=hub_client(hub))

    assert info.source == "huggingface-gguf"
    assert info.architecture.n_kv_heads == 8
    assert info.architecture.param_count == 3212749888
    assert [q.label for q in info.quants] == ["Q8_0", "Q4_K_M"]
    assert hub.range_requests and hub.bytes_served < len(header)


def test_hf_header_read_never_downloads_whole_file_when_range_is_ignored():
    header = llama_like(vocab=40_000) + b"\0" * 2_000_000
    hub = FakeHub({"M-Q4_K_M.gguf": header}, model_api={"gguf": {"total": 1}}, honour_range=False)

    info = hf_model_info("org/M-GGUF", client=hub_client(hub))

    assert info.architecture.n_layers == 28
    assert hub.bytes_served < 600_000


def test_hf_safetensors_repo_uses_config_and_param_count():
    hub = FakeHub(
        {"model.safetensors": 100}, model_api={"safetensors": {"total": 30_532_122_624}}, config=QWEN3_30B_A3B_CONFIG
    )

    info = hf_model_info("Qwen/Qwen3-30B-A3B", client=hub_client(hub))

    assert info.source == "huggingface"
    assert info.architecture.exact and info.architecture.is_moe
    assert info.architecture.param_count == 30_532_122_624
    assert any("no GGUF files" in note for note in info.notes)


def test_gated_config_falls_back_to_estimated_architecture():
    hub = FakeHub({"model.safetensors": 100}, model_api={"safetensors": {"total": 8_030_261_248}}, config_status=401)

    info = hf_model_info("meta-llama/Llama-3.1-8B-Instruct", client=hub_client(hub))

    assert not info.architecture.exact
    assert info.architecture.param_count == 8_030_261_248
    assert any("gated" in note for note in info.notes)


def test_missing_repo_error_is_actionable():
    hub = FakeHub({}, tree_status=401)

    with pytest.raises(InvalidRequestError, match="'nobody/nothing' was not found or is private/gated"):
        hf_model_info("nobody/nothing", client=hub_client(hub))


def test_ollama_model_info_uses_show_and_list():
    class Client:
        def show(self, model: str) -> Any:
            if model != "llama3.2:3b":
                error = Exception("not found")
                error.status_code = 404  # type: ignore[attr-defined]
                raise error
            return SimpleNamespace(modelinfo=LLAMA_32_3B, details=SimpleNamespace(quantization_level="Q4_K_M"))

        def list(self) -> Any:
            return SimpleNamespace(models=[SimpleNamespace(model="llama3.2:3b", size=2_019_393_189)])

    info = ollama_model_info("llama3.2:3b", Client())

    assert info is not None and info.source == "ollama"
    assert info.quants[0].label == "Q4_K_M" and info.quants[0].installed
    assert info.quants[0].size_bytes == 2_019_393_189
    assert ollama_model_info("missing", Client()) is None
    assert json.dumps(info.architecture.to_dict())
