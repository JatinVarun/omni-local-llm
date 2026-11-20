import asyncio
import contextlib
import enum
import inspect
import json
import re
import types
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

_SECTION_HEADER = re.compile(
    r"^\s*(Args|Arguments|Parameters|Params|Returns|Return|Yields|Raises|Examples?|Notes?)\s*:\s*$", re.IGNORECASE
)
_ARG_SECTION = {"args", "arguments", "parameters", "params"}
_GOOGLE_ARG = re.compile(r"^\s*\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SPHINX_PARAM = re.compile(r"^\s*:param\s+(?:[^:]*\s)?(\w+)\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Splits a docstring into a description and per-parameter descriptions (Google or Sphinx style)."""
    description_lines: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    arg_indent: int | None = None

    for line in doc.splitlines():
        header = _SECTION_HEADER.match(line)
        if header:
            section = header.group(1).lower()
            current, arg_indent = None, None
            continue

        sphinx = _SPHINX_PARAM.match(line)
        if sphinx:
            section, current = "sphinx", sphinx.group(1)
            params[current] = sphinx.group(2).strip()
            continue
        if line.strip().startswith(":"):
            section, current = "sphinx", None
            continue

        if section is None:
            description_lines.append(line)
        elif section in _ARG_SECTION:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            match = _GOOGLE_ARG.match(line)
            if match and (arg_indent is None or indent <= arg_indent):
                arg_indent = indent
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current is not None:
                params[current] = f"{params[current]} {line.strip()}".strip()
        elif section == "sphinx" and current is not None and line.strip():
            params[current] = f"{params[current]} {line.strip()}".strip()

    description = " ".join(" ".join(description_lines).split())
    return description, params


def _literal_type(values: Sequence[Any]) -> dict[str, Any]:
    kinds = {type(v) for v in values}
    if kinds == {bool}:
        return {"type": "boolean"}
    if kinds == {int}:
        return {"type": "integer"}
    if kinds <= {int, float}:
        return {"type": "number"}
    if kinds == {str}:
        return {"type": "string"}
    return {}


def _annotation_to_schema(annotation: Any, defs: dict[str, Any]) -> dict[str, Any]:
    """Maps a Python type annotation to a JSON Schema fragment; nested Pydantic $defs are collected in `defs`."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if annotation is Any:
        return {}
    if annotation is None or annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        schema: dict[str, Any] = _annotation_to_schema(args[0], defs)
        description = next((meta for meta in args[1:] if isinstance(meta, str)), None)
        if description:
            schema = {**schema, "description": description}
        return schema

    if origin is Union or origin is types.UnionType:
        members = [arg for arg in args if arg is not type(None)]
        if len(members) == 1:
            return _annotation_to_schema(members[0], defs)
        return {"anyOf": [_annotation_to_schema(member, defs) for member in members]}

    if origin is Literal:
        return {**_literal_type(args), "enum": list(args)}

    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        values = [member.value for member in annotation]
        return {**_literal_type(values), "enum": values}

    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is str:
        return {"type": "string"}

    if inspect.isclass(annotation) and hasattr(annotation, "model_json_schema"):
        schema = dict(annotation.model_json_schema(ref_template="#/$defs/{model}"))
        defs.update(schema.pop("$defs", {}))
        return schema

    container = origin or annotation
    if container in (list, set, frozenset, tuple, Sequence) or (
        inspect.isclass(container) and issubclass(container, Sequence) and container is not str
    ):
        schema = {"type": "array"}
        item_args = [arg for arg in args if arg is not Ellipsis]
        if len(item_args) == 1 or (container is tuple and item_args and len(set(item_args)) == 1):
            schema["items"] = _annotation_to_schema(item_args[0], defs)
        return schema
    if container in (dict, Mapping) or (inspect.isclass(container) and issubclass(container, Mapping)):
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _annotation_to_schema(args[1], defs)
        return schema

    return {"type": "string"}


def function_to_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generates an OpenAI-compatible function tool definition from a Python function.

    Uses type hints (including Optional, list[X], Literal, Enum, Annotated[T, "description"] and Pydantic
    models) and Google- or Sphinx-style docstrings for parameter descriptions.
    """
    if hasattr(func, "__tool_schema__"):
        return func.__tool_schema__

    name = getattr(func, "__name__", "unnamed_function")
    description, param_docs = _parse_docstring(inspect.getdoc(func) or "")

    type_hints: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        type_hints = get_type_hints(func, include_extras=True)

    properties: dict[str, Any] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}

    for param_name, param in inspect.signature(func).parameters.items():
        if param_name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _annotation_to_schema(type_hints.get(param_name, param.annotation), defs)
        if param_name in param_docs and "description" not in schema:
            schema = {**schema, "description": param_docs[param_name]}
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: dict[str, Any] = {"type": "object", "properties": properties, "required": required}
    if defs:
        parameters["$defs"] = defs

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Execute {name}",
            "parameters": parameters,
        },
    }


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that marks a Python function as an Omni-Local-LLM tool with auto-generated schema."""
    schema = function_to_tool_schema(func)
    func.__tool_schema__ = schema  # type: ignore[attr-defined]
    return func


def _run_awaitable_sync(awaitable: Awaitable[Any]) -> Any:
    """Runs an awaitable to completion from sync code, even when an event loop is already running."""

    async def runner() -> Any:
        return await awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, runner()).result()


def _format_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump_json"):
        return str(result.model_dump_json())
    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, default=str)
    return str(result)


class ToolRegistry:
    """Registry that manages tool definitions and executes function calls."""

    def __init__(self, tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None):
        self._functions: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

        if tools:
            for item in tools:
                if callable(item):
                    schema = function_to_tool_schema(item)
                    func_name = schema["function"]["name"]
                    self._functions[func_name] = item
                    self._schemas.append(schema)
                elif isinstance(item, dict):
                    func_info = item.get("function", {})
                    name = func_info.get("name") if isinstance(func_info, dict) else None
                    if name and callable(item.get("callable")):
                        self._functions[name] = item["callable"]
                    self._schemas.append({key: value for key, value in item.items() if key != "callable"})

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def has_tool(self, name: str) -> bool:
        return name in self._functions

    def _prepare(self, name: str, arguments: str | dict[str, Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
        if name not in self._functions:
            raise LookupError(f"Error: Tool '{name}' is not registered.")
        func = self._functions[name]

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                raise ValueError(f"Error: Failed to parse arguments JSON: {e}") from e
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            raise ValueError("Error: Tool arguments must be a JSON object.")

        try:
            return func, self._coerce(func, parsed)
        except Exception as e:
            raise ValueError(f"Error: Invalid arguments for tool '{name}': {e}") from e

    @staticmethod
    def _coerce(func: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
        """Converts JSON values into Pydantic models / Enums where the function's type hints ask for them."""
        try:
            hints = get_type_hints(func)
        except Exception:
            return arguments
        coerced = dict(arguments)
        for key, value in arguments.items():
            annotation = hints.get(key)
            if get_origin(annotation) in (Union, types.UnionType):
                members = [arg for arg in get_args(annotation) if arg is not type(None)]
                annotation = members[0] if len(members) == 1 else None
            if value is None or not inspect.isclass(annotation):
                continue
            if hasattr(annotation, "model_validate") and isinstance(value, dict):
                coerced[key] = annotation.model_validate(value)
            elif issubclass(annotation, enum.Enum) and not isinstance(value, annotation):
                coerced[key] = annotation(value)
        return coerced

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Executes a tool call synchronously and returns the result as a string."""
        try:
            func, kwargs = self._prepare(name, arguments)
        except (LookupError, ValueError) as e:
            return str(e)

        try:
            result = func(**kwargs)
            if inspect.isawaitable(result):
                result = _run_awaitable_sync(result)
            return _format_result(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    async def aexecute(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Executes a tool call asynchronously; sync tools run in a worker thread."""
        try:
            func, kwargs = self._prepare(name, arguments)
        except (LookupError, ValueError) as e:
            return str(e)

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = await asyncio.to_thread(func, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return _format_result(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
