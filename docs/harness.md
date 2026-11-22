# Evaluation & Benchmarking Harness

Omni-Local-LLM includes an evaluation and benchmarking harness to measure local model performance, schema reliability, and context retrieval:

## 1. Performance Benchmarking (`BenchmarkHarness`)

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

## 2. Capability & Schema Compliance (`EvalHarness`)

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

## 3. Context Retrieval ("Needle In A Haystack")

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

[← Back to README](../README.md)
