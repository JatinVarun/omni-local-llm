import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from omnillm import ChatChunk, ChatResponse, LocalLLMManager, Usage
from omnillm.adapters.openai_adapter import OpenAICompatibleAdapter, normalize_base_url
from omnillm.core.errors import BackendUnavailableError, InvalidRequestError
from omnillm.core.types import ChatRequest, EmbeddingRequest
from omnillm.server import create_app
from tests.fakes import RecordingBackend


class Recorder:
    """httpx MockTransport handler that records requests and replays canned responses."""

    def __init__(self, *responses: httpx.Response | Exception):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


class TrackingStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def sse(*events: dict[str, Any] | str) -> list[bytes]:
    lines = [b": keep-alive comment\n\n"]
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {data}\n\n".encode())
    return lines


def adapter_for(handler: Recorder, **kwargs: Any) -> OpenAICompatibleAdapter:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleAdapter("http://server:1234/v1", transport=transport, name="test", **kwargs)


STREAM_EVENTS = (
    {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
    {
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{"}}]},
                "finish_reason": None,
            }
        ]
    },
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    "[DONE]",
)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://localhost:1234", "http://localhost:1234/v1"),
        ("http://localhost:1234/", "http://localhost:1234/v1"),
        ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
        ("https://host/api/openai/v1", "https://host/api/openai/v1"),
    ],
)
def test_base_url_normalization(given: str, expected: str):
    assert normalize_base_url(given) == expected


def test_chat_builds_openai_payload_and_parses_response():
    handler = Recorder(
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )
    )
    adapter = adapter_for(handler, api_key="secret")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]

    response = adapter.chat(
        ChatRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi", "images": [b"img"]}],
            temperature=0.1,
            max_tokens=50,
            top_p=0.9,
            stop=("END",),
            seed=3,
            tools=tools,
            response_format={"type": "json_schema", "json_schema": {"name": "Out", "schema": {"type": "object"}}},
        )
    )

    request = handler.requests[0]
    assert str(request.url) == "http://server:1234/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = handler.body()
    assert body["model"] == "qwen3"
    assert body["temperature"] == 0.1 and body["max_tokens"] == 50 and body["top_p"] == 0.9
    assert body["stop"] == ["END"] and body["seed"] == 3 and body["tools"] == tools
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Out", "schema": {"type": "object"}},
    }
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "stream_options" not in body

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "f", "arguments": '{"a": 1}'},
    }
    assert response.usage == Usage(12, 5)


def test_json_mode_and_raw_schema_formats():
    handler = Recorder(
        httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}),
    )
    adapter = adapter_for(handler)

    adapter.chat(ChatRequest(model="m", messages=[], json_mode=True))
    response = adapter.chat(ChatRequest(model="m", messages=[], response_format={"properties": {"a": {}}}))

    assert handler.body(0)["response_format"] == {"type": "json_object"}
    assert handler.body(1)["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": {"properties": {"a": {}}}},
    }
    assert response.finish_reason == "length"


def test_sync_stream_parses_sse_and_closes_response():
    stream = TrackingStream(sse(*STREAM_EVENTS))
    handler = Recorder(httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"}))
    adapter = adapter_for(handler)

    result = adapter.chat(ChatRequest(model="m", messages=[], stream=True))
    assert isinstance(result, Iterator)
    chunks = list(result)

    assert handler.body()["stream_options"] == {"include_usage": True}
    assert "".join(c.content for c in chunks) == "Hello"
    assert chunks[3].tool_calls[0]["index"] == 0
    assert chunks[4].finish_reason == "tool_calls"
    assert chunks[-1].usage == Usage(7, 3)
    assert stream.closed


def test_abandoned_sync_stream_closes_response():
    stream = TrackingStream(sse(*STREAM_EVENTS))
    adapter = adapter_for(Recorder(httpx.Response(200, stream=stream)))

    result = adapter.chat(ChatRequest(model="m", messages=[], stream=True))
    assert isinstance(result, Iterator)
    next(result)
    result.close()  # type: ignore[attr-defined]

    assert stream.closed


async def test_async_stream_parses_sse_and_closes_response():
    stream = TrackingStream(sse(*STREAM_EVENTS))
    adapter = adapter_for(Recorder(httpx.Response(200, stream=stream)), stream_usage=False)

    result = await adapter.achat(ChatRequest(model="m", messages=[], stream=True))
    assert isinstance(result, AsyncIterator)
    chunks = [chunk async for chunk in result]

    assert "".join(c.content for c in chunks) == "Hello"
    assert chunks[-1].usage == Usage(7, 3)
    assert stream.closed


async def test_async_non_stream_chat():
    handler = Recorder(httpx.Response(200, json={"choices": [{"message": {"content": "hey"}}]}))
    adapter = adapter_for(handler)

    response = await adapter.achat(ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}]))

    assert isinstance(response, ChatResponse) and response.content == "hey"
    assert handler.body()["stream"] is False


@pytest.mark.parametrize(
    ("response", "error_type", "fragment"),
    [
        (httpx.ConnectError("refused"), BackendUnavailableError, "Cannot reach backend 'test'"),
        (httpx.ReadTimeout("slow"), BackendUnavailableError, "timed out"),
        (httpx.Response(404, json={"error": {"message": "model 'x' not found"}}), InvalidRequestError, "not found"),
        (httpx.Response(400, text="bad things"), InvalidRequestError, "bad things"),
        (httpx.Response(401, json={"error": "nope"}), BackendUnavailableError, "API key"),
        (httpx.Response(503, json={"detail": "loading"}), BackendUnavailableError, "loading"),
    ],
)
def test_errors_are_translated(response: Any, error_type: type[Exception], fragment: str):
    adapter = adapter_for(Recorder(response))

    with pytest.raises(error_type, match=fragment):
        adapter.chat(ChatRequest(model="x", messages=[]))


def test_stream_http_errors_surface_at_call_time():
    adapter = adapter_for(Recorder(httpx.Response(400, json={"error": {"message": "context too long"}})))

    with pytest.raises(InvalidRequestError, match="context too long"):
        adapter.chat(ChatRequest(model="x", messages=[], stream=True))


def test_mid_stream_error_event_raises():
    stream = TrackingStream(sse({"choices": [{"delta": {"content": "a"}}]}, {"error": {"message": "OOM"}}))
    adapter = adapter_for(Recorder(httpx.Response(200, stream=stream)))

    result = adapter.chat(ChatRequest(model="x", messages=[], stream=True))
    assert isinstance(result, Iterator)
    assert next(result).content == "a"
    with pytest.raises(InvalidRequestError, match="OOM"):
        next(result)
    assert stream.closed


def test_embeddings_are_sorted_by_index_and_models_are_listed():
    handler = Recorder(
        httpx.Response(
            200,
            json={
                "data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        ),
        httpx.Response(200, json={"data": [{"id": "qwen3"}, {"id": "nomic"}]}),
    )
    adapter = adapter_for(handler)

    response = adapter.embed(EmbeddingRequest(model="nomic", input=("a", "b")))
    models = adapter.list_models()

    assert handler.body(0) == {"model": "nomic", "input": ["a", "b"], "encoding_format": "float"}
    assert response.embeddings == [[0.1], [0.2]]
    assert response.usage is not None and response.usage.prompt_tokens == 4
    assert models == ["qwen3", "nomic"]


async def test_async_embeddings():
    handler = Recorder(httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]}))
    adapter = adapter_for(handler)

    response = await adapter.aembed(EmbeddingRequest(model="e", input=("x",)))

    assert response.embeddings == [[1.0, 2.0]]


async def test_round_trip_against_omnillm_server():
    """Our adapter talking to our own OpenAI-compatible server checks both sides of the wire format."""
    call = {"id": "call_9", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    backend = RecordingBackend(
        response=ChatResponse(content="pong", usage=Usage(2, 1)),
        chunks=(ChatChunk(content="po"), ChatChunk(content="ng"), ChatChunk(tool_calls=(call,), usage=Usage(2, 2))),
    )
    app = create_app(LocalLLMManager({"fake": backend}))
    adapter = OpenAICompatibleAdapter("http://omnillm/v1", async_transport=httpx.ASGITransport(app=app))

    response = await adapter.achat(ChatRequest(model="fake/echo", messages=[{"role": "user", "content": "ping"}]))
    stream = await adapter.achat(
        ChatRequest(model="fake/echo", messages=[{"role": "user", "content": "ping"}], stream=True)
    )
    assert isinstance(stream, AsyncIterator)
    chunks = [chunk async for chunk in stream]

    assert isinstance(response, ChatResponse)
    assert (response.content, response.usage) == ("pong", Usage(2, 1))
    assert backend.requests[0].model == "echo"
    assert "".join(c.content for c in chunks) == "pong"
    assert chunks[-2].finish_reason == "tool_calls"
    assert chunks[-1].usage == Usage(2, 2)
    assert any(c.tool_calls and c.tool_calls[0]["id"] == "call_9" for c in chunks)
