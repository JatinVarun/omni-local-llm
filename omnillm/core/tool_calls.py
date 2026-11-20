"""Conversion between the OpenAI tool-call wire format (the library's canonical shape) and backend formats.

Canonical shape::

    {"id": "call_...", "type": "function", "function": {"name": "...", "arguments": "<JSON string>"}}
"""

import json
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from omnillm.core.types import ToolCall


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _arguments_to_json(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, Mapping):
        arguments = dict(arguments)
    return json.dumps(arguments)


def _arguments_to_dict(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def to_openai_tool_call(call: Any) -> ToolCall:
    """Normalizes a backend tool call (dict or SDK object) into the canonical OpenAI shape."""
    function = _get(call, "function") or {}
    normalized: ToolCall = {
        "id": _get(call, "id") or f"call_{uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": _get(function, "name") or "",
            "arguments": _arguments_to_json(_get(function, "arguments")),
        },
    }
    index = _get(call, "index")
    if index is not None:
        normalized["index"] = index
    return normalized


def to_ollama_tool_call(call: Any) -> dict[str, Any]:
    """Converts a canonical tool call into Ollama's shape, which requires `arguments` as an object."""
    function = _get(call, "function") or {}
    return {
        "function": {
            "name": _get(function, "name") or "",
            "arguments": _arguments_to_dict(_get(function, "arguments")),
        }
    }
