# Python SDK

## Basic Chat

```python
from omnillm import LocalLLMManager

manager = LocalLLMManager()

response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)
```

## Chat Session (with Memory & Context Window Pruning)

`ChatSession` maintains conversation history, commits turns only on success, and supports proactive sliding-window context pruning to keep local models within their context limits:

```python
# Create a session with turn and token budget constraints
session = manager.create_session(
    backend="ollama",
    model="llama3",
    system_prompt="You are a helpful coding assistant.",
    max_turns=10,             # Keep the most recent 10 conversational turns
    max_tokens_budget=4096,   # Automatically evict oldest turns before exceeding context budget
    strategy="sliding_window",
)

# System prompt is ALWAYS preserved during pruning
response1 = session.send("Hello, I am Bob.")
response2 = session.send("What is my name?")  # Remembers "Bob"

# Inspection properties
print(f"Stored turns: {session.turn_count}")
print(f"Estimated token footprint: {session.estimated_tokens}")

# Reset / clear controls
session.clear()                # Clears history, keeping original system_prompt
session.reset("New prompt")    # Clears history and updates system prompt
```

## Async & Streaming

```python
import asyncio
from omnillm import LocalLLMManager

async def main():
    manager = LocalLLMManager()
    session = manager.create_session(backend="ollama", model="llama3")

    stream = await session.asend("Tell me a short story.", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Tool Calling & Autonomous Agent Loop (`@tool` & `session.act`)

Omni-Local-LLM provides an **autonomous tool-calling loop**. Decorate any Python function with `@tool` (or pass plain functions), and `session.act()` will automatically inspect type hints, invoke the function, feed results back to the model, and return the final answer:

```python
from omnillm import tool

# 1. Define tools with standard Python type hints and docstrings
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather conditions for a city."""
    return f"24 degrees {unit}, sunny"

@tool
def calculate_travel_time(distance_km: float, speed_kmh: float = 80.0) -> float:
    """Calculate the estimated travel time in hours."""
    return round(distance_km / speed_kmh, 2)

# 2. Run the autonomous agent loop
session = manager.create_session(backend="ollama", model="llama3")
response = session.act(
    "What's the weather in Rome, and how long does it take to drive 320 km?",
    tools=[get_weather, calculate_travel_time]
)

print(response.content)
# The model invokes get_weather("Rome"), then calculate_travel_time(320.0),
# and automatically returns the final answer with observations included!

# Asynchronous agent loop
response = await session.aact(
    "Check weather in Tokyo",
    tools=[get_weather]
)
```

Tool schemas are generated from type hints and docstrings: `Optional`, `list[X]`, `dict[str, X]`, `Literal`, `Enum`, Pydantic models (arguments are validated into model instances), `Annotated[str, "description"]`, and Google- or Sphinx-style `Args:` descriptions are all supported. Good schemas matter a lot for small local models.

`max_steps` bounds the number of model calls. The whole exchange (user message, tool calls, tool results, final answer) is committed to session history only once the model gives a final answer. If the budget runs out while the model is still requesting tools, `ToolLoopLimitError` is raised (with `.response` and `.messages` for inspection) and history is left unchanged.

Manual low-level OpenAI-format tool calling is also supported via `manager.chat(..., tools=tools)`. Tool calls are always returned in OpenAI format (`id`, `type`, JSON-string `arguments`) regardless of backend.

## Structured Outputs & Pydantic Validation

Guarantee JSON schema adherence at the sampler level (using Ollama schemas or llama.cpp GBNF grammars) and receive typed Pydantic models automatically:

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    skills: list[str]

response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Extract: Alice is a 28-year-old engineer skilled in Python and Rust."}],
    response_model=UserProfile,
)

# Access validated Pydantic object directly
user: UserProfile = response.parsed
print(user.name)    # Alice
print(user.skills)  # ['Python', 'Rust']
```

You can also pass raw JSON schemas or request basic JSON mode:

```python
# Raw JSON Schema
response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Generate a city record"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "City",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
                "required": ["city", "population"],
            },
        },
    },
)

# Basic JSON Mode
response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    json_mode=True,
)
```

## Embeddings (RAG & Semantic Search)

Generate dense vector embeddings for documents or search queries using either backend:

```python
# Synchronous embeddings with Ollama
response = manager.embed(
    backend="ollama",
    model="nomic-embed-text",
    input=["Retrieval-augmented generation with local LLMs", "Semantic similarity search"]
)
print(response.embeddings)  # [[0.021, -0.043, ...], [0.112, 0.009, ...]]
print(response.usage.prompt_tokens)

# Embeddings with llama.cpp (auto-downloads GGUF from Hugging Face Hub)
response = manager.embed(
    backend="llama.cpp",
    model="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    input="Single string input is also supported"
)

# Asynchronous embeddings
response = await manager.aembed(
    backend="ollama",
    model="nomic-embed-text",
    input=["Async batch embedding"]
)
```

## Vision & Multimodal (Image Understanding)

Run local multimodal models (such as `llava`, `llama3.2-vision`, or `qwen2-vl`) using local file paths, raw bytes, base64 data URIs, or OpenAI-standard image parts:

```python
# Direct image input via file path or bytes (Ollama)
response = manager.chat(
    backend="ollama",
    model="llava",
    messages=[{
        "role": "user",
        "content": "What is depicted in this photo?",
        "images": ["photo.jpg"],  # accepts file paths, Path objects, base64, or bytes
    }]
)
print(response.content)

# Using llama.cpp with Hugging Face GGUF + multimodal projector (mmproj)
response = manager.chat(
    backend="llama.cpp",
    model="myself/llava-1.5-7b-GGUF",
    filename="llava-1.5-7b-Q4_K.gguf",
    mmproj_filename="mmproj-model-f16.gguf",  # automatically downloaded & loaded
    messages=[{
        "role": "user",
        "content": "Analyze this chart.",
        "images": ["chart.png"],
    }]
)

# In ChatSession with conversational memory
session = manager.create_session(backend="ollama", model="llava")
response = session.send("What is in this image?", images=["invoice.png"])
response2 = session.send("Extract the total amount.")  # Remembers the image context!

# OpenAI-compatible multi-part format (also supported via FastAPI server)
response = manager.chat(
    backend="ollama",
    model="llava",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }]
)
```

---

[← Back to README](../README.md)
