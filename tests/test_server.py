import json

from fastapi.testclient import TestClient

from omnillm import ChatChunk, ChatResponse, LocalLLMManager, Usage
from omnillm.server import create_app
from tests.fakes import RecordingBackend


def make_client(backend: RecordingBackend) -> TestClient:
    return TestClient(create_app(LocalLLMManager({"fake": backend})))


def test_chat_completions_forwards_parameters_and_returns_usage():
    backend = RecordingBackend(response=ChatResponse(content="Mock response", usage=Usage(3, 2)))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "max_tokens": 20,
            "top_p": 0.8,
            "stop": ["END"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"].startswith("chatcmpl-")
    assert payload["choices"][0]["message"]["content"] == "Mock response"
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    request = backend.requests[0]
    assert request.temperature == 0.3
    assert request.max_tokens == 20
    assert request.top_p == 0.8
    assert request.stop == ["END"]


def test_chat_completions_json_schema_response_format():
    backend = RecordingBackend(response=ChatResponse(content='{"result": 123}'))
    client = make_client(backend)

    schema = {"type": "object", "properties": {"result": {"type": "integer"}}}
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/test",
            "messages": [{"role": "user", "content": "compute"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Result", "schema": schema},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"result": 123}'
    request = backend.requests[0]
    assert request.json_mode is True
    assert request.response_format is not None
    assert request.response_format["type"] == "json_schema"
    assert request.json_schema == schema


def test_models_endpoint_lists_models_from_available_backends():
    client = make_client(RecordingBackend(models=["one", "two"]))

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["fake/one", "fake/two"]


def test_streaming_response_has_role_chunks_finish_reason_and_done_marker():
    backend = RecordingBackend(chunks=(ChatChunk(content="Hello"), ChatChunk(content="!", finish_reason="stop")))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake/test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert json.loads(events[0])["choices"][0]["delta"] == {"role": "assistant"}
    assert json.loads(events[1])["choices"][0]["delta"] == {"content": "Hello"}
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"


def test_request_errors_are_openai_shaped():
    client = make_client(RecordingBackend())

    unknown_backend = client.post(
        "/v1/chat/completions",
        json={"model": "unknown/test", "messages": [{"role": "user", "content": "hi"}]},
    )
    unsupported_n = client.post(
        "/v1/chat/completions",
        json={"model": "fake/test", "messages": [{"role": "user", "content": "hi"}], "n": 2},
    )

    assert unknown_backend.status_code == 404
    assert unknown_backend.json()["error"]["type"] == "invalid_request_error"
    assert unsupported_n.status_code == 400
    assert unsupported_n.json()["error"]["message"] == "Only n=1 is supported."


def test_embeddings_endpoint_returns_openai_format():
    from omnillm.core.types import EmbeddingData, EmbeddingResponse

    backend = RecordingBackend(
        embedding_response=EmbeddingResponse(
            data=(
                EmbeddingData(index=0, embedding=(0.1, 0.2)),
                EmbeddingData(index=1, embedding=(0.3, 0.4)),
            ),
            usage=Usage(prompt_tokens=6),
        )
    )
    client = make_client(backend)

    response = client.post(
        "/v1/embeddings",
        json={"model": "fake/nomic-embed", "input": ["first", "second"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["model"] == "fake/nomic-embed"
    assert len(payload["data"]) == 2
    assert payload["data"][0] == {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
    assert payload["data"][1] == {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
    assert payload["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
    assert backend.embedding_requests[0].input == ("first", "second")


def test_embeddings_endpoint_validates_input_and_encoding():
    client = make_client(RecordingBackend())

    bad_encoding = client.post(
        "/v1/embeddings",
        json={"model": "fake/test", "input": "text", "encoding_format": "base64"},
    )
    empty_input = client.post(
        "/v1/embeddings",
        json={"model": "fake/test", "input": []},
    )

    assert bad_encoding.status_code == 400
    assert "Only float encoding_format is supported" in bad_encoding.json()["error"]["message"]
    assert empty_input.status_code == 400
    assert "Embedding input must contain non-empty strings" in empty_input.json()["error"]["message"]


def test_multimodal_chat_completions():
    backend = RecordingBackend(response=ChatResponse(content="I see a cat."))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/llava",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "I see a cat."
    assert len(backend.requests[0].messages) == 1
    assert isinstance(backend.requests[0].messages[0]["content"], list)


def test_server_never_reads_local_files_named_in_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "abcd").write_bytes(b"top secret")
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    client = make_client(backend)

    rejected = client.post(
        "/v1/chat/completions",
        json={"model": "fake/m", "messages": [{"role": "user", "content": "x", "images": [str(tmp_path / "abcd")]}]},
    )
    accepted = client.post(
        "/v1/chat/completions",
        json={"model": "fake/m", "messages": [{"role": "user", "content": "x", "images": ["abcd"]}]},
    )
    remote = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/m",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://169.254.169.254/"}}]}
            ],
        },
    )

    assert rejected.status_code == 400
    assert remote.status_code == 400
    assert accepted.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].messages[0]["images"] == ["data:image/jpeg;base64,abcd"]


def test_server_forwards_llamacpp_model_source_and_seed():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/owner/repo-GGUF",
            "messages": [{"role": "user", "content": "hi"}],
            "filename": "model-Q4_K_M.gguf",
            "mmproj_filename": "mmproj.gguf",
            "seed": 42,
            "max_completion_tokens": 9,
        },
    )

    assert response.status_code == 200
    request = backend.requests[0]
    assert request.model == "owner/repo-GGUF"
    assert request.model_source.filename == "model-Q4_K_M.gguf"
    assert request.model_source.mmproj_filename == "mmproj.gguf"
    assert request.seed == 42
    assert request.max_tokens == 9


def test_server_tool_choice_none_drops_tools_and_forced_choice_is_rejected():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    client = make_client(backend)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    base = {"model": "fake/m", "messages": [{"role": "user", "content": "hi"}], "tools": tools}

    none = client.post("/v1/chat/completions", json={**base, "tool_choice": "none"})
    auto = client.post("/v1/chat/completions", json={**base, "tool_choice": "auto"})
    required = client.post("/v1/chat/completions", json={**base, "tool_choice": "required"})

    assert none.status_code == 200 and backend.requests[0].tools is None
    assert auto.status_code == 200 and backend.requests[1].tools == tools
    assert required.status_code == 400
    assert "tool_choice" in required.json()["error"]["message"]


def test_streaming_tool_calls_have_indexes_finish_reason_and_optional_usage():
    call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    backend = RecordingBackend(
        chunks=(
            ChatChunk(tool_calls=(call,)),
            ChatChunk(finish_reason="stop", usage=Usage(5, 3)),
        )
    )
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_delta = events[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["index"] == 0
    assert tool_delta["id"] == "call_1"
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_server_routes_aliases_and_lists_them():
    backend = RecordingBackend(response=ChatResponse(content="ok"), models=["llama3"])
    manager = LocalLLMManager({"fake": backend}, default_backend="fake")
    manager.register_alias("coder", backend="fake", model="qwen-coder", temperature=0.2)
    client = TestClient(create_app(manager))

    chat = client.post("/v1/chat/completions", json={"model": "coder", "messages": [{"role": "user", "content": "x"}]})
    namespaced = client.post(
        "/v1/chat/completions", json={"model": "huihui_ai/qwen3", "messages": [{"role": "user", "content": "x"}]}
    )
    models = client.get("/v1/models").json()["data"]

    assert chat.status_code == 200 and chat.json()["model"] == "coder"
    assert (backend.requests[0].model, backend.requests[0].temperature) == ("qwen-coder", 0.2)
    assert namespaced.status_code == 200 and backend.requests[1].model == "huihui_ai/qwen3"
    assert {"id": "coder", "object": "model", "created": 0, "owned_by": "alias:fake/qwen-coder"} in models
    assert any(m["id"] == "fake/llama3" for m in models)


def test_server_module_app_is_built_lazily(monkeypatch, tmp_path):
    import omnillm.server as server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OMNILLM_CONFIG", raising=False)
    monkeypatch.setattr(server, "_default_app", None)

    assert server.app is server.app
