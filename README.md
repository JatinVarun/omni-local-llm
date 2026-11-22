<p align="center">
  <img src="https://img.shields.io/badge/🧠-Omni--Local--LLM-blueviolet?style=for-the-badge&logoColor=white" alt="Omni-Local-LLM" height="50"/>
</p>

<h3 align="center">One interface. Every local LLM.</h3>

<p align="center">
  <em>A unified Python SDK and OpenAI-compatible API server for local large language models.</em>
</p>

<p align="center">
  <a href="https://github.com/matteoespo/omni-local-llm/actions/workflows/ci.yml"><img src="https://github.com/matteoespo/omni-local-llm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/actions/workflows/codeql.yml"><img src="https://github.com/matteoespo/omni-local-llm/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://pypi.org/project/omni-local-llm/"><img src="https://img.shields.io/pypi/v/omni-local-llm?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/omni-local-llm/"><img src="https://img.shields.io/pypi/pyversions/omni-local-llm" alt="Python"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/matteoespo/omni-local-llm?color=green" alt="License"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/stargazers"><img src="https://img.shields.io/github/stars/matteoespo/omni-local-llm?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-interactive-cli">CLI</a> •
  <a href="#-evaluation--benchmarking-harness">Harness</a> •
  <a href="#-api-server">API Server</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Features

| Feature | Description |
|---|---|
| **Multi-Backend** | [Ollama](https://ollama.com/), in-process [llama.cpp](https://github.com/ggerganov/llama.cpp), and **any OpenAI-compatible server** (vLLM, llama-server, LM Studio, SGLang, KoboldCpp, MLX-LM, LocalAI, …) behind one API |
| **Model Aliases & Fallbacks** | Name models once in `omnillm.toml` (`model="coder"`), preset their options, and fall back automatically when a server is down |
| **Vision & Multimodal** | Native image understanding across Ollama and llama.cpp (file paths, bytes, data URIs, and OpenAI format) |
| **Evaluation & Benchmarking** | Measure TTFT, tokens/sec, schema compliance, tool-calling precision, and needle-in-haystack context retrieval |
| **Unified CLI** | Dedicated binary `omnillm` with `chat`, `run`, `models`, `bench`, `eval`, `fit`, and `serve` commands |
| **Hardware Auto-Fit** | `omnillm fit <hf-repo \| ollama-model \| alias \| file.gguf>` reads the model's real architecture and file sizes, computes GQA-aware KV cache, recommends a quant and `n_gpu_layers` for your GPU/RAM, and `--pull` downloads it |
| **Chat Sessions & Pruning** | Resilient turn commits, sliding-window turn pruning, token budget enforcement, and thread-safe resets |
| **Streaming** | Real-time token-by-token streaming (sync & async) |
| **Tool Calling & Agents** | Autonomous function execution loop (`@tool` decorator & `session.act`) |
| **Structured Outputs** | Guaranteed JSON schema adherence and Pydantic validation via GBNF or Ollama schemas |
| **Vector Embeddings** | Generate vector embeddings for RAG and semantic search via SDK or API |
| **OpenAI-Compatible API** | Drop-in FastAPI server compatible with the OpenAI SDK |
| **Async-First** | Full `async`/`await` support for high-concurrency workloads |
| **Auto Model Pull** | Automatically downloads models from Ollama or Hugging Face Hub |

---

## Quick Start

```bash
# Install
pip install "omni-local-llm[ollama,server]"

# Or with uv (recommended)
uv pip install "omni-local-llm[ollama,server]"
```

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

**That's it.** Switch to llama.cpp by changing one word:

```python
response = manager.chat(
    backend="llama.cpp",
    model="unsloth/llama-3-8b-Instruct-GGUF",
    messages=[{"role": "user", "content": "Hello!"}],
    filename="llama-3-8b-Instruct-Q4_K_M.gguf"
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Your Application                   │
├─────────────┬─────────────────────┬─────────────────────┤
│   Python    │   Interactive CLI   │  OpenAI-Compatible  │
│     SDK     │  (python -m omnillm)│   FastAPI Server    │
├─────────────┴─────────────────────┴─────────────────────┤
│                    ChatSession                          │
│              (History · Streaming · Tools)              │
├─────────────────────────────────────────────────────────┤
│                  LocalLLMManager                        │
│     (Backend Registry · Aliases · Router · Fallbacks)   │
├──────────────────┬──────────────────┬───────────────────┤
│  OllamaAdapter   │ LlamaCPPAdapter  │ OpenAICompatible- │
│ (ollama client,  │ (llama-cpp-python│ Adapter (httpx:   │
│  any host)       │  + HF Hub)       │ vLLM, LM Studio…) │
└──────────────────┴──────────────────┴───────────────────┘
```

### Module Map

| Module | Purpose |
|---|---|
| [`omnillm/core/base.py`](omnillm/core/base.py) | `LLMBackend` — abstract base class for all adapters |
| [`omnillm/core/manager.py`](omnillm/core/manager.py) | `LocalLLMManager` — backend registry, model aliases, request routing & fallbacks |
| [`omnillm/core/config.py`](omnillm/core/config.py) | `omnillm.toml` loading and validation (backends, aliases) |
| [`omnillm/core/session.py`](omnillm/core/session.py) | `ChatSession` — conversation state, streaming, tool calls |
| [`omnillm/adapters/ollama_adapter.py`](omnillm/adapters/ollama_adapter.py) | Ollama integration via official Python client |
| [`omnillm/adapters/llamacpp_adapter.py`](omnillm/adapters/llamacpp_adapter.py) | llama.cpp integration + automatic GGUF model caching |
| [`omnillm/adapters/openai_adapter.py`](omnillm/adapters/openai_adapter.py) | Any OpenAI-compatible HTTP server (vLLM, llama-server, LM Studio, …) |
| [`omnillm/core/hardware.py`](omnillm/core/hardware.py) | Hardware detection, KV-cache/memory estimation, GPU/CPU placement, and quant recommendation |
| [`omnillm/core/model_info.py`](omnillm/core/model_info.py) · [`fit.py`](omnillm/core/fit.py) · [`gguf.py`](omnillm/core/gguf.py) | Model metadata from Hugging Face / Ollama / GGUF headers, `fit` orchestration and `--pull` |
| [`omnillm/server.py`](omnillm/server.py) | OpenAI-compatible FastAPI HTTP server |
| [`omnillm/__main__.py`](omnillm/__main__.py) | Interactive CLI chat, benchmark, and evaluation interface |

---

## Backends, Configuration & Model Aliases

### Any OpenAI-compatible server

Most local inference servers speak the OpenAI API. Point Omni-Local-LLM at one and use it like any other backend:

| Server | Default `base_url` |
|---|---|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1` |
| [llama.cpp `llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | `http://localhost:8080/v1` |
| [vLLM](https://docs.vllm.ai/) | `http://localhost:8000/v1` |
| [SGLang](https://docs.sglang.ai/) | `http://localhost:30000/v1` |
| [KoboldCpp](https://github.com/LostRuins/koboldcpp) | `http://localhost:5001/v1` |
| [Ollama](https://ollama.com/) (OpenAI endpoint) | `http://localhost:11434/v1` |

`omnillm serve` also defaults to port 8000, so pass `--port` if you run it next to vLLM.

```bash
# No config needed for a quick try
omnillm chat --base-url http://localhost:1234/v1 --model qwen3-8b
```

```python
from omnillm import LocalLLMManager, OpenAICompatibleAdapter

manager = LocalLLMManager()
manager.register_backend("lmstudio", lambda: OpenAICompatibleAdapter("http://localhost:1234/v1", name="lmstudio"))
response = manager.chat(model="lmstudio/qwen3-8b", messages=[{"role": "user", "content": "Hi!"}])
```

Streaming (with token usage), tool calling, structured outputs, vision, and embeddings all work through it. Unreachable servers, timeouts, auth failures and 5xx responses raise `BackendUnavailableError`; bad requests (unknown model, context too long) raise `InvalidRequestError` with the server's message.

### `omnillm.toml`: named backends and model aliases

Describe your servers and favourite models once. The CLI and `omnillm serve` load the first file found among `--config`, `$OMNILLM_CONFIG`, `./omnillm.toml`, and `~/.config/omnillm/config.toml`; in Python use `LocalLLMManager.from_config()`.

```toml
default_backend = "ollama"          # where bare model names go

[backends.lmstudio]
type = "openai"                     # any OpenAI-compatible server
base_url = "http://localhost:1234/v1"
# api_key_env = "LMSTUDIO_API_KEY"  # or api_key = "..."; also: headers, timeout, capabilities

[backends.gpu-box]
type = "ollama"                     # a second Ollama server
host = "http://gpu-box:11434"

[models.coder]
backend = "llama.cpp"
model = "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
n_ctx = 8192
temperature = 0.2
fallbacks = ["gpu-box/qwen2.5-coder:7b", "fast"]   # tried in order if a backend is unavailable

[models.fast]
model = "lmstudio/qwen3-4b"         # 'backend/model' works too
```

```python
manager = LocalLLMManager.from_config()
manager.chat(model="coder", messages=[...])                  # alias: backend, GGUF file, n_ctx, temperature preset
manager.chat(model="coder", messages=[...], temperature=1)   # explicit arguments override alias presets
manager.chat(model="gpu-box/llama3.2:3b", messages=[...])    # 'backend/model' reference
session = manager.create_session(model="coder")              # aliases resolve on every turn, so fallbacks apply
```

Model references resolve as: **alias** → **`backend/model`** (if the prefix is a known backend) → **bare name on `default_backend`**. So namespaced Ollama models such as `huihui_ai/qwen3` keep working. The config is validated strictly on load, so a typo like `temprature` fails with the file, table, and allowed keys instead of being silently ignored.

Fallbacks trigger only when a backend is unavailable or lacks a needed capability (e.g. vision), never on a bad request. If every target fails, the primary's error is raised with a note per attempted target. Aliases work everywhere: SDK, sessions, CLI (`--model coder`, `bench --targets coder,fast`), and the API server (`"model": "coder"`; aliases are listed in `/v1/models`).

---

## Installation

### Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or pip
- At least one backend:
  - [Ollama](https://ollama.com/) installed and running, **or**
  - A GGUF model on [Hugging Face Hub](https://huggingface.co/) (auto-downloaded)
  - **or** any OpenAI-compatible server (LM Studio, vLLM, llama-server, …)

### Install from source

```bash
git clone https://github.com/matteoespo/omni-local-llm.git
cd omni-local-llm
uv pip install -e ".[ollama,server]"
```

### Install from PyPI

```bash
pip install "omni-local-llm[ollama,server]"
# or
uv pip install "omni-local-llm[ollama,server]"
```

Extras: `ollama`, `llama-cpp`, `openai` (OpenAI-compatible servers; just `httpx`), `server`, or `all`.

---

## Usage

### Python SDK

#### Basic Chat

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

#### Chat Session (with Memory & Context Window Pruning)

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

#### Async & Streaming

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

#### Tool Calling & Autonomous Agent Loop (`@tool` & `session.act`)

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

#### Structured Outputs & Pydantic Validation

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

#### Embeddings (RAG & Semantic Search)

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

#### Vision & Multimodal (Image Understanding)

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

### Interactive CLI

Omni-Local-LLM provides a first-class command-line binary `omnillm` for interactive chat, one-shot prompt execution, model inspection, benchmarking, and serving:

```bash
# 1. Interactive multi-turn chat (with /clear, /reset, /tokens, /system, /exit)
omnillm chat --backend ollama --model llama3

# Chat with context window pruning constraints
omnillm chat --model llama3 --max-turns 10 --max-tokens 2048

# 2. One-shot prompt execution (ideal for bash scripts and UNIX pipes)
omnillm run --model llama3 "Explain quantum computing in one sentence."
cat document.txt | omnillm run --model llama3 "Summarize the key points:"

# 3. List aliases and the models every backend reports
omnillm models

# Use an alias from omnillm.toml, or any OpenAI-compatible server without a config
omnillm run --model coder "Write a Python function that reverses a string."
omnillm chat --base-url http://localhost:8080/v1 --model any
omnillm --config ./team.toml chat --model fast

# 4. Run performance benchmark (TTFT and throughput)
omnillm bench --backend ollama --model llama3 --runs 3
omnillm bench --targets coder,fast,ollama/llama3.2:3b   # compare aliases and refs side by side

# 5. Run model evaluations or Needle-In-A-Haystack retrieval test
omnillm eval --backend ollama --model llama3
omnillm eval --backend ollama --model llama3 --needle

# 6. Start the OpenAI-compatible FastAPI server
omnillm serve --host 127.0.0.1 --port 8000

# 7. Hardware auto-detection and smart fit quantization recommender
omnillm fit
omnillm fit unsloth/Qwen3-8B-GGUF --context 8192          # exact: real files + GGUF header
omnillm fit unsloth/Qwen3-8B-GGUF --pull                  # ...and download the recommended quant
omnillm fit llama3.2:3b --json                            # installed Ollama model
```

---

### Hardware Fit: Which Quant Should I Run? (`omnillm fit`)

`omnillm fit` answers "which file of this model should I download for *my* machine, and how do I run it?" It detects RAM, NVIDIA VRAM, and Apple Silicon unified memory, reads the model's **real architecture**, and evaluates every available quantization:

| You pass | Where the numbers come from |
|---|---|
| A Hugging Face GGUF repo (`unsloth/Qwen3-8B-GGUF`) | Real per-quant file sizes (incl. split files) + architecture from the GGUF header, read with HTTP range requests (a few hundred KB, not the model) |
| A Hugging Face model repo (`Qwen/Qwen3-30B-A3B`) | `config.json` + exact parameter count; quant sizes estimated |
| An installed Ollama model (`llama3.2:3b`) | GGUF metadata via `ollama show` + the installed size |
| A config alias or `backend/model` (`coder`) | Whatever the alias points to, using its `n_ctx` and noting the alias's own file |
| A `.gguf` file | The file's header and size |
| Just a name (`llama-3.1-8b`) or `--offline` | Estimated from the name (`8b`, MoE `30b-a3b`) |

What it computes:

- **KV cache from the real shape**: layers × KV heads × head dim × context, so grouped-query attention (and DeepSeek's MLA) are accounted for. `--kv-cache q8_0|q4_0` shows the effect of a quantized KV cache.
- **Placement**: fully on GPU, Metal unified memory, **split GPU + CPU with a concrete `n_gpu_layers`**, CPU-only, or doesn't fit. It warns when the RAM a plan needs is not free right now.
- **MoE awareness**: total vs. active parameters, plus tips for running experts on CPU.
- **A recommendation**: the best quant (up to Q8_0) that runs fully on the GPU. Otherwise, the best quant up to Q4_K_M split across GPU and CPU, with the faster GPU-only alternative noted.

```bash
omnillm fit unsloth/Qwen3-30B-A3B-GGUF --context 8192     # on a 4 GB GTX 1650 + 16 GB RAM laptop
```

```
🧠 Model Fit Analysis: unsloth/Qwen3-30B-A3B-GGUF
   30.5B params, MoE ~3.3B active, 48 layers, GQA 32/4 heads, trained ctx 40,960 | context 8,192 | KV cache f16
   Source: Hugging Face GGUF repo (exact file sizes + GGUF header)
----------------------------------------------------------------------------------------------------
   QUANT            BPW   WEIGHTS      KV     TOTAL  FIT                                QUALITY            SPEED
----------------------------------------------------------------------------------------------------
   Q8_0            8.50     30.25    0.75     31.53  ❌ Insufficient Memory             Near lossless      -
   ...
   IQ4_XS          4.25     15.25    0.75     16.53  ❌ Insufficient Memory             Good               -
→  Q3_K_M          3.91     13.70    0.75     14.98  ⚠️  GPU+CPU (9 layers on GPU)       Good               Good (MoE)
   ...
💡 Recommendation:
   Recommended: Q3_K_M → GPU + CPU with n_gpu_layers=9 of 48 (~14.98 GB total).
   ℹ️  MoE model: only a fraction of the weights is used per token ...
```

`--pull` downloads the recommended quant (or `--quant Q5_K_M`) into the Hugging Face cache that the llama.cpp backend uses, then prints a ready-to-paste alias:

```bash
omnillm fit bartowski/SmolLM2-135M-Instruct-GGUF --context 2048 --pull --quant Q4_K_M
```

```toml
[models.smollm2-135m-instruct]
backend = "llama.cpp"
model = "bartowski/SmolLM2-135M-Instruct-GGUF"
filename = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
n_ctx = 2048
n_gpu_layers = -1
```

Run `omnillm fit` without arguments for a compatibility matrix of popular models on your machine. Set `HF_TOKEN` for gated repos and `HF_ENDPOINT` for a mirror.

#### Python SDK

```python
from omnillm.core.fit import analyze_model_fit, pull_quant
from omnillm import detect_hardware, estimate_model_memory

rec = analyze_model_fit("unsloth/Qwen3-8B-GGUF", context_length=8192, kv_cache_type="q8_0")
print(rec.recommended_quant, rec.best_target, rec.recommended.n_gpu_layers)
print(rec.format_table())
path, chosen = pull_quant(rec)            # download the recommendation

print(detect_hardware().summary())
total_gb, weights_gb, kv_gb = estimate_model_memory(70.0, quant="Q4_K_M", context_length=16384)
```

---

### Evaluation & Benchmarking Harness

Omni-Local-LLM includes an evaluation and benchmarking harness to measure local model performance, schema reliability, and context retrieval:

#### 1. Performance Benchmarking (`BenchmarkHarness`)

Benchmark Time to First Token (TTFT), generation throughput (Tokens/sec), and latency across multiple models and backends:

```python
from omnillm.harness import BenchmarkHarness

harness = BenchmarkHarness()
suite = harness.run(
    targets=[
        ("ollama", "llama3"),
        ("llama.cpp", "unsloth/llama-3-8b-Instruct-GGUF", {"filename": "llama-3-8b-Instruct-Q4_K_M.gguf"}),
    ],
    prompt="Explain quantum physics in three sentences.",
    max_tokens=128,
    runs=3,
)

# Print markdown comparison table directly to terminal
suite.print_table()
```

Output:
```
| Backend   | Model                                | TTFT (ms) | Tokens/sec | Total (s) | Runs |
|-----------|--------------------------------------|-----------|------------|-----------|------|
| ollama    | llama3                               | 312.4     | 45.2       | 2.85      | 3    |
| llama.cpp | unsloth/llama-3-8b-Instruct-GGUF     | 240.1     | 48.6       | 2.63      | 3    |
```

#### 2. Capability & Schema Compliance (`EvalHarness`)

Systematically evaluate reasoning, JSON Schema adherence, and tool-calling precision:

```python
from pydantic import BaseModel
from omnillm.harness import EvalHarness, EvalTestCase

class Person(BaseModel):
    name: str
    age: int

harness = EvalHarness()
results = harness.run_suite(
    backend="ollama",
    model="llama3",
    test_cases=[
        EvalTestCase(name="Regex Check", prompt="Capital of France?", expected_pattern=r"Paris"),
        EvalTestCase(name="Schema Test", prompt="Extract: Bob, 34", expected_schema=Person),
    ],
)
results.print_table()
# Outputs pass rate %, individual test latency, and errors
```

#### 3. Context Retrieval ("Needle In A Haystack")

Stress-test context length retrieval by hiding a secret code at 0%, 25%, 50%, 75%, and 100% depths:

```python
from omnillm.harness import NeedleHarness

needle_harness = NeedleHarness()
result = needle_harness.run(
    backend="ollama",
    model="llama3",
    secret_code="ALPHA-777",
    depths=[0, 25, 50, 75, 100],
    target_word_count=1500,
)
result.print_table()
# Displays retrieval success and extraction latency per depth
```

---

### API Server

Launch an OpenAI-compatible local API server. It loads `omnillm.toml` like the CLI, so clients can request any alias (`"model": "coder"`), any `backend/model` reference, or a bare model name for the default backend. One endpoint fronts all your local servers, with fallbacks:

```bash
omnillm serve
# Or: python -m omnillm.server
# Server runs on http://localhost:8000
```

#### Use with the OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Chat completions
response = client.chat.completions.create(
    model="ollama/llama3",          # prefix with backend name
    messages=[{"role": "user", "content": "Explain relativity in one sentence."}]
)
print(response.choices[0].message.content)

# Structured Outputs (JSON Schema)
structured_res = client.chat.completions.create(
    model="ollama/llama3",
    messages=[{"role": "user", "content": "Extract: Alice, age 29"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "User",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    },
)
print(structured_res.choices[0].message.content)

# Embeddings
embedding_res = client.embeddings.create(
    model="ollama/nomic-embed-text",
    input=["Embed this chunk for vector search", "Another document"]
)
print(embedding_res.data[0].embedding)
```

#### Server notes

- **llama.cpp models** need a GGUF filename, passed as a non-standard field (`extra_body` in the OpenAI SDK):

  ```python
  client.chat.completions.create(
      model="llama.cpp/unsloth/llama-3-8b-Instruct-GGUF",
      messages=[{"role": "user", "content": "Hi"}],
      extra_body={"filename": "llama-3-8b-Instruct-Q4_K_M.gguf"},
  )
  ```

  An in-process llama.cpp model handles one request at a time; concurrent requests are queued.
- **Images** must be sent inline as base64 or `data:image/...;base64,...` URIs. Local file paths and remote URLs are rejected so API clients can never make the server read its own files or fetch arbitrary URLs.
- **`tool_choice`**: `"auto"` and `"none"` are supported; forcing a specific tool is not possible with local backends and returns a 400.
- `seed`, `max_completion_tokens`, and `stream_options: {"include_usage": true}` are supported.

#### Use with curl

```bash
# Chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Structured Outputs with JSON Schema
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3",
    "messages": [{"role": "user", "content": "Alice is 29 years old."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "User",
        "schema": {
          "type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"]
        }
      }
    }
  }'

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/nomic-embed-text",
    "input": ["Local vector embeddings with Omni-Local-LLM"]
  }'
```

---

## Running Tests

```bash
# With uv (recommended)
uv run pytest

# Or activate the venv first
source .venv/bin/activate
pytest

# With coverage
uv run pytest --cov=omnillm --cov-report=term-missing

# Type checking
uv run mypy omnillm
```

---

## Contributing

Contributions are welcome! Please see the [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up your development environment
- Adding new LLM backend adapters
- Submitting pull requests

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Roadmap

- [x] vLLM, KoboldCpp, LM Studio, llama-server, SGLang… (via the OpenAI-compatible adapter)
- [x] Model aliases with fallbacks (`omnillm.toml`)
- [x] Token usage reporting (including streams)
- [x] Vision/multimodal support
- [x] Hardware-aware `fit` using real GGUF / Hugging Face / Ollama metadata, with `--pull`
- [ ] AMD ROCm / Intel GPU detection in `fit`
- [ ] Prompt template management
- [ ] Docker image for the API server

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/matteoespo">matteoespo</a>
</p>
