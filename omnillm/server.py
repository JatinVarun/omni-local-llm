import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from omnillm.core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from omnillm.core.manager import LocalLLMManager
from omnillm.core.media import sanitize_untrusted_messages
from omnillm.core.types import ChatChunk, ChatResponse, Usage


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    images: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    n: int = Field(default=1, ge=1)
    # Non-OpenAI extensions (send via `extra_body` in the OpenAI SDK) needed to address llama.cpp GGUF models.
    filename: str | None = None
    mmproj_filename: str | None = None


def _resolve_tools(request: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    """Applies `tool_choice`. Local backends can't force a call, so only "auto" and "none" are honoured."""
    choice = request.tool_choice
    if choice is None or choice == "auto":
        return request.tools
    if choice == "none":
        return None
    raise UnsupportedFeatureError("Only tool_choice 'auto' and 'none' are supported by local backends.")


class EmbeddingInputRequest(BaseModel):
    model: str = Field(min_length=1)
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


def _error_response(error: Exception) -> JSONResponse:
    status_code = 500
    error_type = "server_error"
    message = "An internal error has occurred."
    if isinstance(error, BackendNotFoundError):
        status_code, error_type = 404, "invalid_request_error"
        message = str(error)
    elif isinstance(error, BackendUnavailableError):
        status_code, error_type = 503, "server_error"
        message = "The requested backend is currently unavailable."
    elif isinstance(error, (InvalidRequestError, UnsupportedFeatureError, ValueError)):
        status_code, error_type = 400, "invalid_request_error"
        message = str(error)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def create_app(manager: LocalLLMManager | None = None) -> FastAPI:
    app = FastAPI(title="Omni-Local-LLM OpenAI API", version="0.3.0")
    llm_manager = manager if manager is not None else LocalLLMManager.from_config()

    @app.get("/v1/models", response_model=None)
    async def list_models() -> JSONResponse | dict[str, Any]:
        try:
            # Backends may be remote HTTP servers: query them off the event loop.
            backend_models = await asyncio.to_thread(llm_manager.list_models)
            aliases = llm_manager.aliases
        except Exception as error:
            return _error_response(error)
        models = [
            {"id": name, "object": "model", "created": 0, "owned_by": f"alias:{target.ref}"}
            for name, target in aliases.items()
        ]
        models.extend(
            {"id": f"{backend}/{model}", "object": "model", "created": 0, "owned_by": backend}
            for backend, model in backend_models
        )
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse | dict[str, Any]:
        if request.n != 1:
            return _error_response(InvalidRequestError("Only n=1 is supported."))

        try:
            messages = sanitize_untrusted_messages(
                [message.model_dump(exclude_none=True) for message in request.messages]
            )
            tools = _resolve_tools(request)
            response_format_dict: dict[str, Any] | None = None
            json_mode = False
            if request.response_format is not None and request.response_format.type != "text":
                json_mode = True
                response_format_dict = request.response_format.model_dump(by_alias=True, exclude_none=True)

            response = await llm_manager.achat(
                model=request.model,
                messages=messages,
                stream=request.stream,
                json_mode=json_mode,
                response_format=response_format_dict,
                tools=tools,
                temperature=request.temperature,
                max_tokens=request.max_completion_tokens or request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
                seed=request.seed,
                filename=request.filename,
                mmproj_filename=request.mmproj_filename,
            )
        except Exception as error:
            return _error_response(error)

        chat_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        if request.stream:
            if not isinstance(response, AsyncIterator):
                return _error_response(InvalidRequestError("Backend did not return a stream."))

            include_usage = request.stream_options is not None and request.stream_options.include_usage

            async def stream() -> AsyncIterator[str]:
                finish_reason = "stop"
                saw_tool_calls = False
                next_tool_index = 0
                usage: Usage | None = None
                yield _sse(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                try:
                    async for chunk in response:
                        if not isinstance(chunk, ChatChunk):
                            raise InvalidRequestError("Backend returned an invalid stream chunk.")
                        delta: dict[str, Any] = {}
                        if chunk.content:
                            delta["content"] = chunk.content
                        if chunk.tool_calls:
                            calls = []
                            for call in chunk.tool_calls:
                                call = dict(call)
                                if "index" not in call:
                                    call["index"] = next_tool_index
                                next_tool_index = max(next_tool_index, call["index"] + 1)
                                calls.append(call)
                            delta["tool_calls"] = calls
                            saw_tool_calls = True
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if delta:
                            yield _sse(
                                {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                                }
                            )
                    yield _sse(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "tool_calls" if saw_tool_calls else finish_reason,
                                }
                            ],
                        }
                    )
                    if include_usage:
                        yield _sse(
                            {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [],
                                "usage": usage.as_openai() if usage is not None else None,
                            }
                        )
                except Exception as error:
                    payload = bytes(_error_response(error).body).decode()
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        if not isinstance(response, ChatResponse):
            return _error_response(InvalidRequestError("Backend returned a stream."))

        message: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            message["tool_calls"] = list(response.tool_calls)
        payload: dict[str, Any] = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response.finish_reason,
                }
            ],
        }
        if response.usage is not None and (usage := response.usage.as_openai()) is not None:
            payload["usage"] = usage
        return payload

    @app.post("/v1/embeddings", response_model=None)
    async def embeddings(request: EmbeddingInputRequest) -> JSONResponse | dict[str, Any]:
        if request.encoding_format != "float":
            return _error_response(InvalidRequestError("Only float encoding_format is supported."))

        inputs = [request.input] if isinstance(request.input, str) else request.input
        if not inputs or any(not isinstance(text, str) or not text for text in inputs):
            return _error_response(InvalidRequestError("Embedding input must contain non-empty strings."))

        try:
            response = await llm_manager.aembed(
                model=request.model,
                input=inputs,
            )
        except Exception as error:
            return _error_response(error)

        data = [
            {
                "object": "embedding",
                "index": item.index,
                "embedding": list(item.embedding),
            }
            for item in response.data
        ]
        payload: dict[str, Any] = {
            "object": "list",
            "data": data,
            "model": request.model,
        }
        if response.usage is not None and response.usage.prompt_tokens is not None:
            payload["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.prompt_tokens,
            }
        return payload

    return app


_default_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Builds the module-level ``app`` (for ``uvicorn omnillm.server:app``) on first access, so importing this
    module never reads config files."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
