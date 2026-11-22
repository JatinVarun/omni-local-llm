"""Model resolution, aliases, and fallbacks in LocalLLMManager."""

from typing import Any

import pytest

from omnillm import ChatResponse, EmbeddingData, EmbeddingResponse, LocalLLMManager
from omnillm.core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from omnillm.core.types import BackendCapabilities
from tests.fakes import RecordingBackend


def make_manager(**backends: RecordingBackend) -> LocalLLMManager:
    return LocalLLMManager(dict(backends), default_backend="main")


def test_resolution_rules():
    manager = make_manager(main=RecordingBackend(), remote=RecordingBackend())
    manager.register_alias("coder", backend="remote", model="qwen-coder", temperature=0.2)
    manager.register_alias("short", model="remote/llama3")

    assert manager.resolve("coder").ref == "remote/qwen-coder"
    assert manager.resolve("coder").options == {"temperature": 0.2}
    assert manager.resolve("short").ref == "remote/llama3"
    assert manager.resolve("remote/x").ref == "remote/x"
    # Unknown prefix is part of the model name (e.g. namespaced Ollama or HF models).
    assert manager.resolve("huihui_ai/qwen3").ref == "main/huihui_ai/qwen3"
    assert manager.resolve("plain").ref == "main/plain"
    # An explicit backend bypasses alias and prefix handling.
    assert manager.resolve("coder", backend="main").ref == "main/coder"


def test_alias_options_are_defaults_and_explicit_values_win():
    remote = RecordingBackend()
    manager = make_manager(main=RecordingBackend(), remote=remote)
    manager.register_alias(
        "coder", backend="remote", model="m", temperature=0.2, n_ctx=8192, filename="m.gguf", local_files_only=True
    )

    manager.chat(model="coder", messages=[{"role": "user", "content": "hi"}])
    manager.chat(model="coder", messages=[], temperature=0.9, local_files_only=False)

    first, second = remote.requests
    assert (first.model, first.temperature, first.runtime.n_ctx) == ("m", 0.2, 8192)
    assert first.model_source.filename == "m.gguf" and first.model_source.local_files_only is True
    assert second.temperature == 0.9 and second.runtime.n_ctx == 8192
    assert second.model_source.local_files_only is False


def test_register_alias_validation():
    manager = make_manager(main=RecordingBackend())
    with pytest.raises(ValueError, match="unsupported options"):
        manager.register_alias("a", model="x", temprature=0.1)
    with pytest.raises(ValueError, match="must not contain '/'"):
        manager.register_alias("a/b", model="x")
    with pytest.raises(ValueError, match="needs a model"):
        manager.register_alias("a", model="")


def test_fallback_on_unavailable_backend_uses_each_targets_own_options():
    down = RecordingBackend(error=BackendUnavailableError("down"))
    up = RecordingBackend(response=ChatResponse(content="from fallback"))
    manager = make_manager(main=down, backup=up)
    manager.register_alias("smart", backend="main", model="big", temperature=0.1, fallbacks=["small"])
    manager.register_alias("small", backend="backup", model="tiny", max_tokens=50)

    response = manager.chat(model="smart", messages=[])

    assert isinstance(response, ChatResponse) and response.content == "from fallback"
    assert down.requests[0].model == "big"
    assert (up.requests[0].model, up.requests[0].temperature, up.requests[0].max_tokens) == ("tiny", None, 50)


def test_fallback_on_unsupported_feature():
    no_vision = RecordingBackend()
    no_vision.capabilities = BackendCapabilities(vision=False)
    vision = RecordingBackend(response=ChatResponse(content="I see"))
    manager = make_manager(main=no_vision, vision=vision)
    manager.register_alias("eyes", backend="main", model="text-only", fallbacks=["vision/llava"])

    response = manager.chat(model="eyes", messages=[{"role": "user", "content": "x", "images": [b"img"]}])

    assert isinstance(response, ChatResponse) and response.content == "I see"
    assert no_vision.requests == []


def test_invalid_requests_do_not_fall_back():
    bad = RecordingBackend(error=InvalidRequestError("context too long"))
    backup = RecordingBackend()
    manager = make_manager(main=bad, backup=backup)
    manager.register_alias("a", model="x", fallbacks=["backup/y"])

    with pytest.raises(InvalidRequestError, match="context too long"):
        manager.chat(model="a", messages=[])
    assert backup.requests == []


def test_all_targets_failing_raises_primary_error_with_notes():
    manager = make_manager(
        main=RecordingBackend(error=BackendUnavailableError("main down")),
        backup=RecordingBackend(error=BackendUnavailableError("backup down")),
    )
    manager.register_alias("a", model="x", fallbacks=["backup/y"])

    with pytest.raises(BackendUnavailableError, match="main down") as info:
        manager.chat(model="a", messages=[])

    notes = "\n".join(info.value.__notes__)
    assert "main/x" in notes and "backup/y: BackendUnavailableError: backup down" in notes


def test_single_target_errors_are_unchanged():
    error = BackendUnavailableError("down")
    manager = make_manager(main=RecordingBackend(error=error))

    with pytest.raises(BackendUnavailableError) as info:
        manager.chat(model="x", messages=[])

    assert info.value is error and not getattr(error, "__notes__", None)


def test_fallback_cycles_are_skipped_and_unknown_backends_raise():
    manager = make_manager(main=RecordingBackend(), other=RecordingBackend())
    manager.register_alias("a", model="other/x", fallbacks=["b"])
    manager.register_alias("b", model="main/y", fallbacks=["a", "b"])
    manager.register_alias("broken", backend="ghost", model="x")

    assert [t.ref for t in manager.candidates("a")] == ["other/x", "main/y"]
    with pytest.raises(BackendNotFoundError):
        manager.chat(model="broken", messages=[])


async def test_async_chat_and_embedding_fallbacks():
    down = RecordingBackend(error=BackendUnavailableError("down"))
    up = RecordingBackend(
        response=ChatResponse(content="ok"),
        embedding_response=EmbeddingResponse(data=(EmbeddingData(index=0, embedding=(1.0,)),)),
    )
    manager = make_manager(main=down, backup=up)
    manager.register_alias("a", model="x", fallbacks=["backup/y"], n_ctx=2048, temperature=0.5)

    response = await manager.achat(model="a", messages=[])
    embedding = await manager.aembed(model="a", input="text")
    sync_embedding = manager.embed(model="a", input=["text"])

    assert isinstance(response, ChatResponse) and response.content == "ok"
    assert embedding.embeddings == sync_embedding.embeddings == [[1.0]]
    # Only embedding-relevant alias options reach embedding requests.
    assert down.embedding_requests[0].runtime.n_ctx == 2048


def test_sessions_resolve_aliases_on_every_turn():
    primary = RecordingBackend(response=ChatResponse(content="primary"))
    backup = RecordingBackend(response=ChatResponse(content="backup"))
    manager = make_manager(main=primary, backup=backup)
    manager.register_alias("assistant", model="main/x", temperature=0.3, fallbacks=["backup/y"])
    session = manager.create_session(model="assistant", system_prompt="sys")

    assert session.send("one").content == "primary"
    primary.error = BackendUnavailableError("down")
    assert session.send("two").content == "backup"

    assert primary.requests[0].temperature == 0.3
    assert [m["content"] for m in backup.requests[0].messages] == ["sys", "one", "primary", "two"]
    assert session.turn_count == 2


def test_positional_backend_model_messages_still_work():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    manager = LocalLLMManager({"fake": backend})

    response: Any = manager.chat("fake", "m", [{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    with pytest.raises(ValueError, match="messages are required"):
        manager.chat("fake", "m")


def test_capability_checks_still_apply_without_fallbacks():
    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(tools=False)
    manager = LocalLLMManager({"fake": backend})

    with pytest.raises(UnsupportedFeatureError):
        manager.chat("fake", "m", [], tools=[{"type": "function"}])
