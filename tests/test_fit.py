from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from omnillm import LocalLLMManager, OpenAICompatibleAdapter
from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter
from omnillm.adapters.ollama_adapter import OllamaAdapter
from omnillm.core import fit as fit_module
from omnillm.core.errors import InvalidRequestError
from omnillm.core.fit import alias_snippet, analyze_model_fit, pull_quant, resolve_fit_target
from omnillm.core.hardware import HardwareProfile
from tests.gguf_writer import llama_like
from tests.test_model_info import LLAMA_32_3B, FakeHub, hub_client

HW = HardwareProfile(
    "Linux",
    "6.8",
    "x86_64",
    8,
    32.0,
    28.0,
    gpu_name="GPU",
    gpu_count=1,
    total_vram_gb=8.0,
    free_vram_gb=7.5,
    has_cuda=True,
)


class FakeOllama:
    def __init__(self, installed: dict[str, dict[str, Any]]):
        self.installed = installed

    def show(self, model: str) -> Any:
        if model not in self.installed:
            error = Exception("missing")
            error.status_code = 404  # type: ignore[attr-defined]
            raise error
        return SimpleNamespace(modelinfo=self.installed[model], details=SimpleNamespace(quantization_level="Q4_K_M"))

    def list(self) -> Any:
        return SimpleNamespace(models=[SimpleNamespace(model=m, size=2_000_000_000) for m in self.installed])


def gguf_hub() -> FakeHub:
    return FakeHub(
        {"Llama-Q8_0.gguf": 3_400_000_000, "Llama-Q4_K_M.gguf": llama_like()},
        model_api={"gguf": {"total": 3_212_749_888}},
    )


def manager_with_aliases() -> LocalLLMManager:
    manager = LocalLLMManager()
    manager.register_backend("llama.cpp", lambda: LlamaCPPAdapter(llama_factory=object, hub_download=lambda **_: ""))
    manager.register_backend("box", lambda: OllamaAdapter(client=FakeOllama({"llama3.2:3b": LLAMA_32_3B})))
    manager.register_backend("remote", lambda: OpenAICompatibleAdapter("http://remote:1234"))
    manager.register_alias(
        "coder", backend="llama.cpp", model="org/Llama-GGUF", filename="Llama-Q4_K_M.gguf", n_ctx=8192
    )
    manager.register_alias("boxed", backend="box", model="llama3.2:3b", n_ctx=2048)
    manager.register_alias("hosted", backend="remote", model="qwen3-8b")
    return manager


def test_local_gguf_file(tmp_path: Path):
    path = tmp_path / "x-Q4_K_M.gguf"
    path.write_bytes(llama_like())

    assert resolve_fit_target(str(path)).info.source == "gguf-file"


def test_alias_to_llamacpp_reads_the_hf_repo_and_remembers_file_and_context():
    target = resolve_fit_target("coder", manager=manager_with_aliases(), hf_client=hub_client(gguf_hub()))

    assert target.info.source == "huggingface-gguf"
    assert target.preferred_quant_file == "Llama-Q4_K_M.gguf"
    assert target.context_length == 8192


def test_alias_to_ollama_backend_uses_that_backends_client():
    target = resolve_fit_target("boxed", manager=manager_with_aliases())

    assert target.info.source == "ollama"
    assert target.context_length == 2048


def test_alias_to_openai_backend_is_estimated_with_explanation():
    target = resolve_fit_target("hosted", manager=manager_with_aliases())

    assert target.info.source == "estimated"
    assert "served by backend 'remote'" in target.info.notes[0]


def test_bare_names_use_installed_ollama_models_then_estimates():
    client = FakeOllama({"llama3.2:3b": LLAMA_32_3B})

    assert resolve_fit_target("llama3.2:3b", ollama_client=client).info.source == "ollama"
    assert resolve_fit_target("mistral:7b", ollama_client=client).info.source == "estimated"


def test_offline_never_touches_the_network():
    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network used")

    client = httpx.Client(transport=httpx.MockTransport(explode), base_url="https://hub.test")

    assert resolve_fit_target("org/Model-8B-GGUF", offline=True, hf_client=client).info.source == "estimated"


def test_unreachable_hub_falls_back_to_estimates_with_a_note():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    client = httpx.Client(transport=httpx.MockTransport(refuse), base_url="https://hub.test")

    info = resolve_fit_target("org/Model-8B-GGUF", hf_client=client).info
    assert info.source == "estimated" and "Cannot reach Hugging Face" in info.notes[0]


def test_analyze_alias_notes_the_aliases_own_quant():
    rec = analyze_model_fit("coder", manager=manager_with_aliases(), hardware=HW, hf_client=hub_client(gguf_hub()))

    assert rec.context_length == 8192
    assert rec.notes[0].startswith("Alias 'coder' uses Q4_K_M:")


def test_pull_downloads_the_recommended_or_chosen_quant(monkeypatch: pytest.MonkeyPatch):
    downloads: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        fit_module,
        "download_hf_files",
        lambda repo, files, revision="main", progress=None: (
            downloads.append((repo, files)) or Path("/cache") / files[0]
        ),
    )
    rec = analyze_model_fit("org/Llama-GGUF", hardware=HW, hf_client=hub_client(gguf_hub()))

    path, result = pull_quant(rec)
    chosen_path, chosen = pull_quant(rec, "q4_k_m")

    assert result.quant == rec.recommended_quant == "Q8_0"
    assert downloads == [("org/Llama-GGUF", ("Llama-Q8_0.gguf",)), ("org/Llama-GGUF", ("Llama-Q4_K_M.gguf",))]
    assert chosen.quant == "Q4_K_M" and chosen_path == Path("/cache/Llama-Q4_K_M.gguf")
    with pytest.raises(InvalidRequestError, match="Choose from: Q8_0, Q4_K_M"):
        pull_quant(rec, "IQ1_S")

    snippet = alias_snippet(rec, result)
    assert snippet.splitlines()[0] == "[models.llama]"
    assert 'filename = "Llama-Q8_0.gguf"' in snippet and "n_gpu_layers = -1" in snippet


def test_pull_rejects_non_hf_sources():
    rec = analyze_model_fit("mistral:7b", hardware=HW, offline=True)

    with pytest.raises(InvalidRequestError, match="--pull needs a Hugging Face GGUF repo"):
        pull_quant(rec)
