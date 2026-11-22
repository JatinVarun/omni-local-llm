from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from omnillm.core.session import ChatSession

from omnillm.core.base import LLMBackend
from omnillm.core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from omnillm.core.media import has_images
from omnillm.core.types import (
    AsyncChatResult,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelSource,
    ModelTarget,
    RuntimeOptions,
)

BackendFactory = Callable[[], LLMBackend]

#: Request options an alias may preset; explicit per-call values always win.
ALIAS_OPTION_KEYS: tuple[str, ...] = (
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "seed",
    "filename",
    "revision",
    "cache_dir",
    "local_files_only",
    "n_gpu_layers",
    "n_ctx",
    "mmproj_filename",
    "clip_model_path",
)
_EMBED_OPTION_KEYS = frozenset({"filename", "revision", "cache_dir", "local_files_only", "n_gpu_layers", "n_ctx"})


def _create_ollama() -> LLMBackend:
    from omnillm.adapters.ollama_adapter import OllamaAdapter

    return OllamaAdapter()


def _create_llamacpp() -> LLMBackend:
    from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter

    return LlamaCPPAdapter()


def _resolve_response_format(
    response_model: type[Any] | None,
    response_format: dict[str, Any] | None,
    json_mode: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if response_model is not None:
        if hasattr(response_model, "model_json_schema"):
            schema = response_model.model_json_schema()
        elif hasattr(response_model, "schema"):
            schema = response_model.schema()
        else:
            raise ValueError(f"response_model '{response_model}' must be a Pydantic model with model_json_schema().")
        return True, {
            "type": "json_schema",
            "json_schema": {
                "name": getattr(response_model, "__name__", "ResponseModel"),
                "schema": schema,
                "strict": True,
            },
        }
    if response_format is not None:
        fmt_type = response_format.get("type", "")
        is_json = (
            json_mode
            or fmt_type in {"json_object", "json_schema"}
            or "schema" in response_format
            or "properties" in response_format
        )
        return is_json, response_format
    return json_mode, None


def _attach_parsed_model[R: (ChatResult, AsyncChatResult)](result: R, response_model: type[Any] | None) -> R:
    if response_model is None or not isinstance(result, ChatResponse):
        return result
    try:
        parsed = result.parse_as(response_model)
        return cast(
            R,
            ChatResponse(
                content=result.content,
                tool_calls=result.tool_calls,
                finish_reason=result.finish_reason,
                usage=result.usage,
                parsed=parsed,
            ),
        )
    except Exception as error:
        name = getattr(response_model, "__name__", str(response_model))
        raise InvalidRequestError(f"Failed to parse response into {name}: {error}") from error


class _AliasSpec:
    __slots__ = ("backend", "fallbacks", "model", "options")

    def __init__(self, backend: str | None, model: str, options: dict[str, Any], fallbacks: tuple[str, ...]):
        self.backend = backend
        self.model = model
        self.options = options
        self.fallbacks = fallbacks


def _merge_options(defaults: Mapping[str, Any], explicit: Mapping[str, Any]) -> dict[str, Any]:
    return {**defaults, **{key: value for key, value in explicit.items() if value is not None}}


class LocalLLMManager:
    """The external seam for backend registration, model resolution, routing, and request validation.

    Models can be addressed three ways when `backend` is omitted:

    * an alias registered with `register_alias` (or declared in a config file),
    * a ``"backend/model"`` reference whose prefix is a registered backend,
    * a bare model name, which goes to `default_backend`.
    """

    def __init__(self, backends: Mapping[str, LLMBackend] | None = None, *, default_backend: str = "ollama"):
        self._backend_factories: dict[str, BackendFactory] = (
            {}
            if backends is not None
            else {
                "ollama": _create_ollama,
                "llama.cpp": _create_llamacpp,
            }
        )
        self._backends = dict(backends or {})
        self._aliases: dict[str, _AliasSpec] = {}
        self.default_backend = default_backend

    @classmethod
    def from_config(cls, path: str | Path | None = None, *, search: bool = True) -> "LocalLLMManager":
        """Creates a manager configured from a TOML file.

        With no `path`, looks at ``$OMNILLM_CONFIG``, ``./omnillm.toml``, then the user config directory
        (``~/.config/omnillm/config.toml``). If nothing is found, returns a default manager.
        """
        from omnillm.core.config import apply_config, find_config_file, load_config

        manager = cls()
        config_path = find_config_file(path, search=search)
        if config_path is not None:
            apply_config(manager, load_config(config_path))
        return manager

    # ------------------------------------------------------------------ registry

    @property
    def backend_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self._backend_factories, *self._backends)))

    def register_backend(self, name: str, factory: BackendFactory) -> None:
        if not name or "/" in name:
            raise ValueError("Backend names must be non-empty and must not contain '/'.")
        self._backend_factories[name] = factory
        self._backends.pop(name, None)

    def register_alias(
        self,
        name: str,
        *,
        model: str,
        backend: str | None = None,
        fallbacks: Sequence[str] = (),
        **options: Any,
    ) -> None:
        """Registers a short model name with preset request options and optional fallback models.

        `model` may itself be a ``"backend/model"`` reference when `backend` is omitted. `fallbacks` are
        aliases or model references tried in order when a target's backend is unavailable.
        """
        if not name or "/" in name:
            raise ValueError("Alias names must be non-empty and must not contain '/'.")
        if not model:
            raise ValueError(f"Alias '{name}' needs a model.")
        unknown = sorted(set(options) - set(ALIAS_OPTION_KEYS))
        if unknown:
            raise ValueError(f"Alias '{name}' has unsupported options {unknown}. Allowed: {list(ALIAS_OPTION_KEYS)}")
        if isinstance(fallbacks, str):
            fallbacks = (fallbacks,)
        self._aliases[name] = _AliasSpec(backend, model, dict(options), tuple(fallbacks))

    @property
    def alias_names(self) -> tuple[str, ...]:
        return tuple(self._aliases)

    @property
    def aliases(self) -> dict[str, ModelTarget]:
        return {name: self.resolve(name) for name in self._aliases}

    def _get_backend(self, name: str) -> LLMBackend:
        if name in self._backends:
            return self._backends[name]
        if name not in self._backend_factories:
            raise BackendNotFoundError(f"Backend '{name}' is not supported. Choose from {list(self.backend_names)}")
        self._backends[name] = self._backend_factories[name]()
        return self._backends[name]

    # ---------------------------------------------------------------- resolution

    def _resolve_ref(self, ref: str) -> tuple[str, str]:
        prefix, sep, rest = ref.partition("/")
        if sep and rest and prefix in self.backend_names:
            return prefix, rest
        return self.default_backend, ref

    def resolve(self, model: str, backend: str | None = None) -> ModelTarget:
        """Resolves a model reference (alias, ``backend/model``, or bare name) to a concrete target."""
        if not model:
            raise ValueError("A model name is required.")
        if backend is not None:
            return ModelTarget(backend=backend, model=model)
        spec = self._aliases.get(model)
        if spec is None:
            backend_name, model_name = self._resolve_ref(model)
            return ModelTarget(backend=backend_name, model=model_name)
        if spec.backend is not None:
            backend_name, model_name = spec.backend, spec.model
        else:
            backend_name, model_name = self._resolve_ref(spec.model)
        return ModelTarget(
            backend=backend_name,
            model=model_name,
            options=dict(spec.options),
            fallbacks=spec.fallbacks,
            alias=model,
        )

    def candidates(self, model: str, backend: str | None = None) -> list[ModelTarget]:
        """Returns the target for `model` followed by its fallbacks (expanded depth-first, cycles skipped)."""
        ordered: list[ModelTarget] = []
        seen: set[str] = set()

        def visit(ref: str, explicit_backend: str | None) -> None:
            target = self.resolve(ref, explicit_backend)
            key = target.alias or target.ref
            if key in seen:
                return
            seen.add(key)
            ordered.append(target)
            for fallback in target.fallbacks:
                visit(fallback, None)

        visit(model, backend)
        return ordered

    def _run_with_fallbacks[T](self, targets: Sequence[ModelTarget], attempt: Callable[[ModelTarget], T]) -> T:
        errors: list[Exception] = []
        for target in targets:
            try:
                return attempt(target)
            except (BackendUnavailableError, UnsupportedFeatureError) as error:
                if len(targets) == 1:
                    raise
                errors.append(error)
        raise self._fallback_failure(targets, errors)

    async def _arun_with_fallbacks[T](
        self, targets: Sequence[ModelTarget], attempt: Callable[[ModelTarget], Awaitable[T]]
    ) -> T:
        errors: list[Exception] = []
        for target in targets:
            try:
                return await attempt(target)
            except (BackendUnavailableError, UnsupportedFeatureError) as error:
                if len(targets) == 1:
                    raise
                errors.append(error)
        raise self._fallback_failure(targets, errors)

    @staticmethod
    def _fallback_failure(targets: Sequence[ModelTarget], errors: Sequence[Exception]) -> Exception:
        primary = errors[0]
        for target, error in zip(targets, errors, strict=False):
            primary.add_note(f"{target.ref}: {type(error).__name__}: {error}")
        return primary

    # ------------------------------------------------------------------- requests

    def _make_request(
        self,
        target: ModelTarget,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        json_mode: bool,
        response_model: type[Any] | None,
        response_format: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]] | None,
        options: Mapping[str, Any],
    ) -> tuple[LLMBackend, ChatRequest]:
        backend = target.backend
        adapter = self._get_backend(backend)
        if stream and not adapter.capabilities.streaming:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support streaming.", feature="streaming")
        is_json, resolved_format = _resolve_response_format(response_model, response_format, json_mode)
        if is_json and not adapter.capabilities.json_mode:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support JSON mode.", feature="json_mode")
        if (
            response_model is not None or (resolved_format and resolved_format.get("type") == "json_schema")
        ) and not getattr(adapter.capabilities, "structured_outputs", True):
            raise UnsupportedFeatureError(
                f"Backend '{backend}' does not support structured outputs.", feature="structured_outputs"
            )
        if tools and not adapter.capabilities.tools:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support tool calling.", feature="tools")
        if has_images(messages) and not getattr(adapter.capabilities, "vision", True):
            raise UnsupportedFeatureError(
                f"Backend '{backend}' does not support vision / multimodal inputs.", feature="vision"
            )
        opts = _merge_options(target.options, options)
        return adapter, ChatRequest(
            model=target.model,
            messages=messages,
            stream=stream,
            json_mode=is_json,
            response_format=resolved_format,
            tools=tools,
            temperature=opts.get("temperature"),
            max_tokens=opts.get("max_tokens"),
            top_p=opts.get("top_p"),
            stop=opts.get("stop"),
            seed=opts.get("seed"),
            model_source=ModelSource(
                opts.get("filename"),
                opts.get("revision"),
                opts.get("cache_dir"),
                bool(opts.get("local_files_only", False)),
                opts.get("mmproj_filename"),
            ),
            runtime=RuntimeOptions(opts.get("n_gpu_layers"), opts.get("n_ctx"), opts.get("clip_model_path")),
        )

    def chat(
        self,
        backend: str | None = None,
        model: str | None = None,
        messages: Sequence[ChatMessage] | None = None,
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
        mmproj_filename: str | None = None,
        clip_model_path: str | None = None,
    ) -> ChatResult:
        """Sends a chat request.

        Pass `backend` and `model` explicitly, or omit `backend` and give `model` as an alias,
        ``"backend/model"`` reference, or bare name (see the class docstring). Alias fallbacks are tried in
        order when a backend is unavailable.
        """
        if messages is None:
            raise ValueError("messages are required.")
        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "seed": seed,
            "filename": filename,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "mmproj_filename": mmproj_filename,
            "clip_model_path": clip_model_path,
        }

        def attempt(target: ModelTarget) -> ChatResult:
            adapter, request = self._make_request(
                target,
                messages,
                stream=stream,
                json_mode=json_mode,
                response_model=response_model,
                response_format=response_format,
                tools=tools,
                options=options,
            )
            return _attach_parsed_model(adapter.chat(request), response_model)

        return self._run_with_fallbacks(self.candidates(model or "", backend), attempt)

    async def achat(
        self,
        backend: str | None = None,
        model: str | None = None,
        messages: Sequence[ChatMessage] | None = None,
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
        mmproj_filename: str | None = None,
        clip_model_path: str | None = None,
    ) -> AsyncChatResult:
        """Asynchronous version of `chat`."""
        if messages is None:
            raise ValueError("messages are required.")
        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "seed": seed,
            "filename": filename,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "mmproj_filename": mmproj_filename,
            "clip_model_path": clip_model_path,
        }

        async def attempt(target: ModelTarget) -> AsyncChatResult:
            adapter, request = self._make_request(
                target,
                messages,
                stream=stream,
                json_mode=json_mode,
                response_model=response_model,
                response_format=response_format,
                tools=tools,
                options=options,
            )
            return _attach_parsed_model(await adapter.achat(request), response_model)

        return await self._arun_with_fallbacks(self.candidates(model or "", backend), attempt)

    # ----------------------------------------------------------------- embeddings

    def _make_embedding_request(
        self, target: ModelTarget, input: str | Sequence[str], options: Mapping[str, Any]
    ) -> tuple[LLMBackend, EmbeddingRequest]:
        if isinstance(input, str):
            inputs: tuple[str, ...] = (input,)
        elif isinstance(input, Sequence):
            inputs = tuple(input)
        else:
            raise ValueError("Embedding input must be a string or a sequence of strings.")

        if not inputs or any(not isinstance(item, str) for item in inputs):
            raise ValueError("Embedding input must contain at least one string.")

        adapter = self._get_backend(target.backend)
        if not adapter.capabilities.embeddings:
            raise UnsupportedFeatureError(
                f"Backend '{target.backend}' does not support embeddings.", feature="embeddings"
            )

        defaults = {key: value for key, value in target.options.items() if key in _EMBED_OPTION_KEYS}
        opts = _merge_options(defaults, options)
        return adapter, EmbeddingRequest(
            model=target.model,
            input=inputs,
            model_source=ModelSource(
                opts.get("filename"),
                opts.get("revision"),
                opts.get("cache_dir"),
                bool(opts.get("local_files_only", False)),
            ),
            runtime=RuntimeOptions(opts.get("n_gpu_layers"), opts.get("n_ctx")),
        )

    def embed(
        self,
        backend: str | None = None,
        model: str | None = None,
        input: str | Sequence[str] | None = None,
        *,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> EmbeddingResponse:
        if input is None:
            raise ValueError("Embedding input is required.")
        options = {
            "filename": filename,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
        }

        def attempt(target: ModelTarget) -> EmbeddingResponse:
            adapter, request = self._make_embedding_request(target, input, options)
            return adapter.embed(request)

        return self._run_with_fallbacks(self.candidates(model or "", backend), attempt)

    async def aembed(
        self,
        backend: str | None = None,
        model: str | None = None,
        input: str | Sequence[str] | None = None,
        *,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> EmbeddingResponse:
        if input is None:
            raise ValueError("Embedding input is required.")
        options = {
            "filename": filename,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
        }

        async def attempt(target: ModelTarget) -> EmbeddingResponse:
            adapter, request = self._make_embedding_request(target, input, options)
            return await adapter.aembed(request)

        return await self._arun_with_fallbacks(self.candidates(model or "", backend), attempt)

    # ------------------------------------------------------------------- sessions

    def create_session(
        self,
        backend: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        *,
        max_turns: int | None = None,
        max_tokens_budget: int | None = None,
        strategy: Literal["full", "sliding_window"] = "full",
    ) -> "ChatSession":
        """Creates a conversation. `backend`/`model` are resolved on every request, so alias fallbacks apply."""
        from omnillm.core.session import ChatSession

        target = self.resolve(model or "", backend)
        self._get_backend(target.backend)
        return ChatSession(
            manager=self,
            backend=backend,
            model=model or "",
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_tokens_budget=max_tokens_budget,
            strategy=strategy,
        )

    def list_models(self, backend: str | None = None) -> list[tuple[str, str]]:
        """Lists models reported by backends; unreachable backends are skipped unless named explicitly."""
        names = (backend,) if backend else self.backend_names
        models: list[tuple[str, str]] = []
        for name in names:
            try:
                available_models = self._get_backend(name).list_models()
            except BackendUnavailableError:
                if backend is not None:
                    raise
                continue
            for model in available_models:
                models.append((name, model))
        return models
