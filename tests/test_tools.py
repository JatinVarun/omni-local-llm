import asyncio
import enum
import json
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel

from omnillm import (
    ChatResponse,
    LocalLLMManager,
    ToolLoopLimitError,
    ToolRegistry,
    function_to_tool_schema,
    tool,
)
from tests.fakes import RecordingBackend


def add_numbers(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Gets the current weather for a city."""
    return f"Weather in {city}: 22 degrees {unit}"


def test_function_to_tool_schema():
    schema = function_to_tool_schema(add_numbers)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add_numbers"
    assert "Adds two integers" in fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"
    assert params["required"] == ["a", "b"]


def test_tool_decorator():
    schema = getattr(get_weather, "__tool_schema__", None)
    assert schema is not None
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["parameters"]["required"] == ["city"]


def test_tool_registry_execution():
    registry = ToolRegistry([add_numbers, get_weather])
    assert len(registry.schemas) == 2

    # String JSON args
    res = registry.execute("add_numbers", '{"a": 10, "b": 25}')
    assert res == "35"

    # Dict args
    res_weather = registry.execute("get_weather", {"city": "Rome"})
    assert "Rome" in res_weather

    # Unknown tool
    assert "Error: Tool 'unknown' is not registered" in registry.execute("unknown", {})


@pytest.mark.asyncio
async def test_tool_registry_async_execution():
    async def fetch_price(symbol: str) -> float:
        return 142.50

    registry = ToolRegistry([fetch_price])
    res = await registry.aexecute("fetch_price", '{"symbol": "NVDA"}')
    assert res == "142.5"


def test_session_act_loop():
    # Sequence of responses:
    # 1. Assistant requests tool call: add_numbers(a=5, b=15)
    # 2. After receiving tool output, assistant provides final answer
    backend = RecordingBackend()
    responses = [
        ChatResponse(
            content="",
            tool_calls=(
                {
                    "id": "call_1",
                    "function": {"name": "add_numbers", "arguments": '{"a": 5, "b": 15}'},
                },
            ),
        ),
        ChatResponse(content="The sum of 5 and 15 is 20."),
    ]

    # Dynamically return next response on chat()
    def chat_mock(request):
        backend.requests.append(request)
        return responses.pop(0)

    backend.chat = chat_mock

    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")
    final_response = session.act("Calculate 5 + 15", tools=[add_numbers])

    assert final_response.content == "The sum of 5 and 15 is 20."
    assert len(session.messages) == 4
    # User message
    assert session.messages[0]["role"] == "user"
    # Assistant tool call
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["tool_calls"][0]["function"]["name"] == "add_numbers"
    # Tool output
    assert session.messages[2]["role"] == "tool"
    assert session.messages[2]["content"] == "20"
    # Assistant final answer
    assert session.messages[3]["role"] == "assistant"
    assert session.messages[3]["content"] == "The sum of 5 and 15 is 20."


@pytest.mark.asyncio
async def test_session_aact_loop():
    backend = RecordingBackend()
    responses = [
        ChatResponse(
            content="",
            tool_calls=(
                {
                    "id": "call_10",
                    "function": {"name": "get_weather", "arguments": '{"city": "Milan"}'},
                },
            ),
        ),
        ChatResponse(content="The weather in Milan is pleasant."),
    ]

    async def achat_mock(request):
        backend.requests.append(request)
        return responses.pop(0)

    backend.achat = achat_mock

    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")
    final_response = await session.aact("How is Milan?", tools=[get_weather])

    assert final_response.content == "The weather in Milan is pleasant."
    assert len(session.messages) == 4
    assert session.messages[2]["role"] == "tool"
    assert "Milan" in session.messages[2]["content"]


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Address(BaseModel):
    city: str
    zip_code: str


def rich_tool(
    query: Annotated[str, "What to search for"],
    tags: list[str],
    mode: Literal["fast", "deep"] = "fast",
    color: Color | None = None,
    limit: int | None = None,
    address: Address | None = None,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search the catalogue.

    Longer explanation that spans
    two lines.

    Args:
        query: Ignored here because Annotated wins.
        tags: Tags to filter by,
            matched case-insensitively.
        limit: Maximum number of results.

    Returns:
        The matches.
    """
    return {"query": query, "color": color.value if color else None, "city": address.city if address else None}


def test_schema_supports_rich_annotations_and_docstrings():
    fn = function_to_tool_schema(rich_tool)["function"]
    props = fn["parameters"]["properties"]

    assert fn["description"] == "Search the catalogue. Longer explanation that spans two lines."
    assert props["query"] == {"type": "string", "description": "What to search for"}
    assert props["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "Tags to filter by, matched case-insensitively.",
    }
    assert props["mode"] == {"type": "string", "enum": ["fast", "deep"]}
    assert props["color"] == {"type": "string", "enum": ["red", "blue"]}
    assert props["limit"] == {"type": "integer", "description": "Maximum number of results."}
    assert props["address"]["properties"]["city"]["type"] == "string"
    assert "args" not in props and "kwargs" not in props
    assert fn["parameters"]["required"] == ["query", "tags"]


def test_schema_hoists_nested_pydantic_defs():
    class Order(BaseModel):
        shipping: Address

    def place(order: Order) -> str:
        """Place an order."""
        return "ok"

    params = function_to_tool_schema(place)["function"]["parameters"]

    assert params["properties"]["order"]["properties"]["shipping"] == {"$ref": "#/$defs/Address"}
    assert "Address" in params["$defs"]


def test_sphinx_docstrings_are_parsed():
    def lookup(key: str) -> str:
        """Look a key up.

        :param key: The key to find.
        :returns: The value.
        """
        return key

    fn = function_to_tool_schema(lookup)["function"]
    assert fn["description"] == "Look a key up."
    assert fn["parameters"]["properties"]["key"]["description"] == "The key to find."


def test_execute_coerces_pydantic_and_enum_arguments():
    registry = ToolRegistry([rich_tool])

    result = registry.execute(
        "rich_tool",
        {"query": "q", "tags": [], "color": "blue", "address": {"city": "Rome", "zip_code": "00100"}},
    )

    assert json.loads(result) == {"query": "q", "color": "blue", "city": "Rome"}
    assert registry.execute("rich_tool", {"query": "q", "tags": [], "color": "green"}).startswith("Error: Invalid")
    assert registry.execute("rich_tool", "[1, 2]") == "Error: Tool arguments must be a JSON object."


@pytest.mark.asyncio
async def test_sync_execute_runs_async_tools_inside_a_running_loop():
    async def fetch(symbol: str) -> dict[str, float]:
        await asyncio.sleep(0)
        return {"price": 1.5}

    registry = ToolRegistry([fetch])

    assert json.loads(registry.execute("fetch", {"symbol": "X"})) == {"price": 1.5}


def test_dict_tools_do_not_leak_callable_into_schema():
    schema = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}
    registry = ToolRegistry([{**schema, "callable": lambda: "pong"}])

    assert registry.schemas == [schema]
    assert registry.execute("ping", {}) == "pong"


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_act_raises_and_leaves_history_untouched_when_step_budget_runs_out():
    backend = RecordingBackend(response=ChatResponse(tool_calls=(_tool_call("add_numbers", '{"a": 1, "b": 2}'),)))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "model", system_prompt="sys")

    with pytest.raises(ToolLoopLimitError) as info:
        session.act("loop forever", tools=[add_numbers], max_steps=3)

    assert len(backend.requests) == 3
    assert session.messages == [{"role": "system", "content": "sys"}]
    assert info.value.response.tool_calls
    assert [m["role"] for m in info.value.messages] == ["assistant", "tool", "assistant", "tool", "assistant"]
    session.send("still usable")  # busy flag released


def test_act_holds_the_session_for_the_whole_loop_and_commits_atomically():
    responses = [ChatResponse(tool_calls=(_tool_call("peek", "{}"),)), ChatResponse(content="done")]
    backend = RecordingBackend()
    backend.chat = lambda request: backend.requests.append(request) or responses.pop(0)  # type: ignore[method-assign]
    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")

    def peek() -> str:
        """Tries to use the session re-entrantly."""
        try:
            session.send("sneaky")
        except RuntimeError as error:
            return str(error)
        return "not blocked"

    response = session.act("go", tools=[peek])

    assert response.content == "done"
    assert "active request" in session.messages[2]["content"]
    assert [m["role"] for m in session.messages] == ["user", "assistant", "tool", "assistant"]
    # The second model call saw the full in-turn transcript.
    assert [m["role"] for m in backend.requests[1].messages] == ["user", "assistant", "tool"]


def test_act_does_not_commit_when_backend_fails_mid_loop():
    calls = {"n": 0}
    backend = RecordingBackend()

    def chat(request: Any) -> ChatResponse:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("backend died")
        return ChatResponse(tool_calls=(_tool_call("add_numbers", '{"a": 1, "b": 2}'),))

    backend.chat = chat  # type: ignore[method-assign]
    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")

    with pytest.raises(ConnectionError):
        session.act("go", tools=[add_numbers])

    assert session.messages == []


@pytest.mark.asyncio
async def test_aact_raises_when_step_budget_runs_out():
    backend = RecordingBackend(response=ChatResponse(tool_calls=(_tool_call("get_weather", '{"city": "Rome"}'),)))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")

    with pytest.raises(ToolLoopLimitError):
        await session.aact("loop", tools=[get_weather], max_steps=2)

    assert session.messages == []
