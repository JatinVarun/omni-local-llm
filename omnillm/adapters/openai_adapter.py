"""Adapter for any server exposing the OpenAI HTTP API.

Works with vLLM, llama.cpp's llama-server, LM Studio, SGLang, KoboldCpp, MLX-LM, LocalAI, TGI, Jan, Ollama's
own ``/v1`` endpoint, and hosted OpenAI-compatible APIs.
"""

import asyncio
import json
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from omnillm.core.base import LLMBackend
from omnillm.core.errors import BackendUnavailableError, InvalidRequestError, ModelNotFoundError, OmniLLMError
from omnillm.core.media import normalize_messages_for_openai
from omnillm.core.tool_calls import to_openai_tool_call
from omnillm.core.types import (
    BackendCapabilities,
    ChatChunk,
    ChatRequest,
    ChatResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    ModelSource,
    Usage,
)

httpx: Any = None
try:
    import httpx as httpx_module
except ImportError:  # Optional dependency.
    pass
else:
    httpx = httpx_module

_MAX_ERROR_CHARS = 500


def normalize_base_url(base_url: str) -> str:
    """Strips trailing slashes and appends ``/v1`` when the URL has no path (``http://host:1234``)."""
    url = base_url.strip().rstrip("/")
    if not url:
        raise ValueError("base_url must not be empty.")
    if urlsplit(url).path in ("", "/"):
        url += "/v1"
    return url


def _finish_reason(value: Any) -> FinishReason:
    if value in {"stop", "length", "tool_calls"}:
        return value
    if value == "function_call":
        return "tool_calls"
    return "stop"


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, Mapping):
        return None
    prompt_tokens = raw.get("prompt_tokens")
    completion_tokens = raw.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:_MAX_ERROR_CHARS]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:_MAX_ERROR_CHARS]
    if isinstance(error, str):
        return error[:_MAX_ERROR_CHARS]
    if isinstance(data, dict) and data.get("detail"):
        return str(data["detail"])[:_MAX_ERROR_CHARS]
    return body.strip()[:_MAX_ERROR_CHARS]


class _ClosingIterator(Iterator[ChatChunk]):
    """Iterates chunks and closes the HTTP response once exhausted, failed, closed, or garbage-collected."""

    def __init__(self, chunks: Iterator[ChatChunk], close: Callable[[], None]):
        self._chunks = chunks
        self._close = close
        self._closed = False

    def __next__(self) -> ChatChunk:
        try:
            return next(self._chunks)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close()

    def __del__(self) -> None:
        self.close()


class OpenAICompatibleAdapter(LLMBackend):
    """Talks to an OpenAI-compatible ``/v1`` API over HTTP. Models are managed by the server, not pulled here."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
        connect_timeout: float = 5.0,
        stream_usage: bool = True,
        capabilities: BackendCapabilities | None = None,
        name: str = "openai",
        transport: Any = None,
        async_transport: Any = None,
    ):
        if httpx is None:
            raise BackendUnavailableError("The OpenAI-compatible backend requires httpx: pip install httpx")
        self.base_url = normalize_base_url(base_url)
        self.name = name
        self.capabilities = capabilities or BackendCapabilities()
        self._stream_usage = stream_usage
        self._headers = {"Accept": "application/json", **dict(headers or {})}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._transport = transport
        self._async_transport = async_transport if async_transport is not None else transport
        self._client: Any = None
        self._async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------ clients

    def _sync_client(self) -> Any:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url, headers=self._headers, timeout=self._timeout, transport=self._transport
            )
        return self._client

    def _async_client(self) -> Any:
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url, headers=self._headers, timeout=self._timeout, transport=self._async_transport
            )
            self._async_clients[loop] = client
        return client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------- errors

    def _unavailable(self, error: Exception) -> BackendUnavailableError:
        if isinstance(error, httpx.TimeoutException):
            return BackendUnavailableError(f"Backend '{self.name}' at {self.base_url} timed out.")
        return BackendUnavailableError(f"Cannot reach backend '{self.name}' at {self.base_url}.")

    def _status_error(self, status: int, body: str) -> OmniLLMError:
        message = f"Backend '{self.name}' returned HTTP {status}: {_error_message(body)}"
        if status in (401, 403):
            return BackendUnavailableError(f"{message} (check the API key)")
        if status in (408, 429) or status >= 500:
            return BackendUnavailableError(message)
        if status == 404:
            return ModelNotFoundError(message)
        return InvalidRequestError(message)

    # ----------------------------------------------------------------- payloads

    def _response_format(self, request: ChatRequest) -> dict[str, Any] | None:
        schema = request.json_schema
        if schema is not None:
            spec = (request.response_format or {}).get("json_schema")
            spec = spec if isinstance(spec, dict) else {}
            json_schema: dict[str, Any] = {"name": spec.get("name") or "response", "schema": schema}
            if "strict" in spec:
                json_schema["strict"] = spec["strict"]
            return {"type": "json_schema", "json_schema": json_schema}
        if request.is_json:
            return {"type": "json_object"}
        return None

    def _chat_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": normalize_messages_for_openai(request.messages),
            "stream": request.stream,
        }
        for key, value in {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "seed": request.seed,
        }.items():
            if value is not None:
                payload[key] = value
        if request.stop is not None:
            payload["stop"] = request.stop if isinstance(request.stop, str) else list(request.stop)
        if request.tools:
            payload["tools"] = list(request.tools)
        response_format = self._response_format(request)
        if response_format is not None:
            payload["response_format"] = response_format
        if request.stream and self._stream_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _response(data: Any) -> ChatResponse:
        if not isinstance(data, dict):
            raise InvalidRequestError("Backend returned a malformed chat completion.")
        choices = data.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        tool_calls = tuple(to_openai_tool_call(call) for call in message.get("tool_calls") or [])
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else _finish_reason(choice.get("finish_reason")),
            usage=_usage(data.get("usage")),
        )

    def _chunk(self, data_line: str) -> ChatChunk | None:
        try:
            data = json.loads(data_line)
        except ValueError as error:
            raise InvalidRequestError(f"Backend '{self.name}' sent a malformed stream event.") from error
        if not isinstance(data, dict):
            return None
        if data.get("error"):
            raise InvalidRequestError(f"Backend '{self.name}' stream error: {_error_message(json.dumps(data))}")
        choices = data.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")
        return ChatChunk(
            content=delta.get("content") or "",
            # Streaming tool-call deltas are partial (index + fragments) and are passed through as-is.
            tool_calls=tuple(dict(call) for call in delta.get("tool_calls") or []),
            finish_reason=_finish_reason(finish_reason) if finish_reason is not None else None,
            usage=_usage(data.get("usage")),
        )

    @staticmethod
    def _data_lines(lines: Iterator[str]) -> Iterator[str]:
        for line in lines:
            if not line.startswith("data:"):
                continue  # comments (":"), event/id fields, blank separators
            data = line[5:].strip()
            if data == "[DONE]":
                return
            if data:
                yield data

    def _chunks(self, lines: Iterator[str]) -> Iterator[ChatChunk]:
        try:
            for data in self._data_lines(lines):
                chunk = self._chunk(data)
                if chunk is not None:
                    yield chunk
        except httpx.TransportError as error:
            raise self._unavailable(error) from error

    # --------------------------------------------------------------------- chat

    def pull_model(self, model_name: str, source: ModelSource) -> None:
        """Models live on the remote server; nothing to download."""
        del model_name, source

    def chat(self, request: ChatRequest) -> ChatResponse | Iterator[ChatChunk]:
        client = self._sync_client()
        payload = self._chat_payload(request)
        try:
            if not request.stream:
                response = client.post("/chat/completions", json=payload)
                if response.status_code >= 400:
                    raise self._status_error(response.status_code, response.text)
                return self._response(response.json())

            response = client.send(client.build_request("POST", "/chat/completions", json=payload), stream=True)
        except httpx.TransportError as error:
            raise self._unavailable(error) from error
        if response.status_code >= 400:
            try:
                body = response.read().decode(errors="replace")
            finally:
                response.close()
            raise self._status_error(response.status_code, body)
        return _ClosingIterator(self._chunks(response.iter_lines()), response.close)

    async def achat(self, request: ChatRequest) -> ChatResponse | AsyncIterator[ChatChunk]:
        client = self._async_client()
        payload = self._chat_payload(request)
        try:
            if not request.stream:
                response = await client.post("/chat/completions", json=payload)
                if response.status_code >= 400:
                    raise self._status_error(response.status_code, response.text)
                return self._response(response.json())

            response = await client.send(client.build_request("POST", "/chat/completions", json=payload), stream=True)
        except httpx.TransportError as error:
            raise self._unavailable(error) from error
        if response.status_code >= 400:
            try:
                body = (await response.aread()).decode(errors="replace")
            finally:
                await response.aclose()
            raise self._status_error(response.status_code, body)

        async def stream() -> AsyncIterator[ChatChunk]:
            try:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if data and (chunk := self._chunk(data)) is not None:
                        yield chunk
            except httpx.TransportError as error:
                raise self._unavailable(error) from error
            finally:
                await response.aclose()

        return stream()

    # --------------------------------------------------------------- embeddings

    def _embedding_payload(self, request: EmbeddingRequest) -> dict[str, Any]:
        return {"model": request.model, "input": list(request.input), "encoding_format": "float"}

    @staticmethod
    def _embedding_response(request: EmbeddingRequest, data: Any) -> EmbeddingResponse:
        if not isinstance(data, dict):
            raise InvalidRequestError("Backend returned a malformed embeddings response.")
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        embeddings = [
            EmbeddingData(index=item.get("index", position), embedding=tuple(item.get("embedding") or ()))
            for position, item in enumerate(items)
        ]
        usage = _usage(data.get("usage"))
        return EmbeddingResponse(
            data=embeddings,
            model=request.model,
            usage=Usage(prompt_tokens=usage.prompt_tokens) if usage else None,
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            response = self._sync_client().post("/embeddings", json=self._embedding_payload(request))
        except httpx.TransportError as error:
            raise self._unavailable(error) from error
        if response.status_code >= 400:
            raise self._status_error(response.status_code, response.text)
        return self._embedding_response(request, response.json())

    async def aembed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            response = await self._async_client().post("/embeddings", json=self._embedding_payload(request))
        except httpx.TransportError as error:
            raise self._unavailable(error) from error
        if response.status_code >= 400:
            raise self._status_error(response.status_code, response.text)
        return self._embedding_response(request, response.json())

    # ------------------------------------------------------------------- models

    def list_models(self) -> list[str]:
        try:
            response = self._sync_client().get("/models")
        except httpx.TransportError as error:
            raise self._unavailable(error) from error
        if response.status_code >= 400:
            raise self._status_error(response.status_code, response.text)
        data = response.json()
        items = data.get("data") if isinstance(data, dict) else None
        return [str(item["id"]) for item in items or [] if isinstance(item, dict) and item.get("id")]
