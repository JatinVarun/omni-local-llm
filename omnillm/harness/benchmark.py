import contextlib
import statistics
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from omnillm.core.manager import LocalLLMManager
from omnillm.core.types import ChatChunk


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    """A model endpoint to benchmark."""

    backend: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls, spec: "tuple[str, str] | tuple[str, str, dict[str, Any]] | BenchmarkTarget | str"
    ) -> "BenchmarkTarget":
        if isinstance(spec, BenchmarkTarget):
            return spec
        if isinstance(spec, str):
            if "/" in spec:
                backend, model = spec.split("/", 1)
                return cls(backend=backend, model=model)
            return cls(backend="ollama", model=spec)
        if len(spec) == 2:
            return cls(backend=spec[0], model=spec[1])
        if len(spec) == 3:
            return cls(backend=spec[0], model=spec[1], options=dict(spec[2]))
        raise ValueError(f"Invalid benchmark target specification: {spec}")


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Metrics recorded during a single benchmark run."""

    ttft_ms: float
    total_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    token_counts_exact: bool = False


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregated results across multiple runs for a single target."""

    target: BenchmarkTarget
    runs: list[BenchmarkMetrics] = field(default_factory=list)

    @property
    def mean_ttft_ms(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.ttft_ms for r in self.runs)

    @property
    def mean_tokens_per_second(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.tokens_per_second for r in self.runs)

    @property
    def mean_total_time_ms(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.total_time_ms for r in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.target.backend,
            "model": self.target.model,
            "mean_ttft_ms": round(self.mean_ttft_ms, 2),
            "mean_tokens_per_second": round(self.mean_tokens_per_second, 2),
            "mean_total_time_ms": round(self.mean_total_time_ms, 2),
            "runs_count": len(self.runs),
            "runs": [
                {
                    "ttft_ms": round(r.ttft_ms, 2),
                    "total_time_ms": round(r.total_time_ms, 2),
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "token_counts_exact": r.token_counts_exact,
                }
                for r in self.runs
            ],
        }


@dataclass(slots=True)
class BenchmarkSuite:
    """Collection of benchmark results comparing multiple targets."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"benchmarks": [r.to_dict() for r in self.results]}

    def format_table(self) -> str:
        """Formats the benchmark results into a clean markdown comparison table."""
        if not self.results:
            return "No benchmark results available."

        headers = ["Backend", "Model", "TTFT (ms)", "Tokens/sec", "Total (s)", "Runs"]
        rows = []
        for r in self.results:
            rows.append(
                [
                    r.target.backend,
                    r.target.model,
                    f"{r.mean_ttft_ms:.1f}",
                    f"{r.mean_tokens_per_second:.1f}",
                    f"{r.mean_total_time_ms / 1000:.2f}",
                    str(len(r.runs)),
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        def format_row(values: list[str]) -> str:
            return "| " + " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values)) + " |"

        separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
        lines = [format_row(headers), separator]
        for row in rows:
            lines.append(format_row(row))
        return "\n".join(lines)

    def print_table(self) -> None:
        print(self.format_table())


class _RunRecorder:
    """Times one streamed generation and counts tokens.

    Token counts come from the backend's reported usage when available (exact); otherwise each content
    chunk is counted as one token, which matches how Ollama and llama.cpp stream.
    """

    def __init__(self, prompt: str):
        self._prompt = prompt
        self._start = time.perf_counter()
        self._first_token: float | None = None
        self._content_chunks = 0
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None

    def observe(self, chunk: object) -> None:
        if getattr(chunk, "content", ""):
            if self._first_token is None:
                self._first_token = time.perf_counter()
            self._content_chunks += 1
        usage = chunk.usage if isinstance(chunk, ChatChunk) else None
        if usage is not None:
            if usage.prompt_tokens is not None:
                self._prompt_tokens = usage.prompt_tokens
            if usage.completion_tokens is not None:
                self._completion_tokens = usage.completion_tokens

    def finish(self) -> BenchmarkMetrics:
        end = time.perf_counter()
        first = self._first_token or end
        exact = self._completion_tokens is not None
        completion = self._completion_tokens if self._completion_tokens is not None else self._content_chunks
        # Decode throughput: tokens generated after the first one, over the time since the first one.
        decode_seconds = end - first
        tps = (completion - 1) / decode_seconds if completion > 1 and decode_seconds > 0.001 else 0.0
        return BenchmarkMetrics(
            ttft_ms=(first - self._start) * 1000,
            total_time_ms=(end - self._start) * 1000,
            prompt_tokens=self._prompt_tokens if self._prompt_tokens is not None else len(self._prompt) // 4,
            completion_tokens=completion,
            tokens_per_second=tps,
            token_counts_exact=exact,
        )


class BenchmarkHarness:
    """Harness for performance benchmarking, measuring TTFT, throughput (tokens/sec), and latency."""

    def __init__(self, manager: LocalLLMManager | None = None):
        self.manager = manager or LocalLLMManager()

    def run(
        self,
        targets: Sequence[tuple[str, str] | tuple[str, str, dict[str, Any]] | BenchmarkTarget | str],
        prompt: str = "Explain the theory of general relativity in three concise sentences.",
        *,
        max_tokens: int = 128,
        runs: int = 3,
        warmup: bool = True,
    ) -> BenchmarkSuite:
        """Runs the performance benchmark synchronously across all targets."""
        suite = BenchmarkSuite()
        normalized_targets = [BenchmarkTarget.from_spec(t) for t in targets]

        for target in normalized_targets:
            res = BenchmarkResult(target=target)
            messages = [{"role": "user", "content": prompt}]

            if warmup:
                with contextlib.suppress(Exception):
                    self.manager.chat(
                        backend=target.backend,
                        model=target.model,
                        messages=messages,
                        max_tokens=16,
                        **target.options,
                    )

            for _ in range(runs):
                recorder = _RunRecorder(prompt)
                stream = self.manager.chat(
                    backend=target.backend,
                    model=target.model,
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens,
                    **target.options,
                )
                if isinstance(stream, Iterator):
                    for chunk in stream:
                        recorder.observe(chunk)
                res.runs.append(recorder.finish())

            suite.results.append(res)

        return suite

    async def arun(
        self,
        targets: Sequence[tuple[str, str] | tuple[str, str, dict[str, Any]] | BenchmarkTarget | str],
        prompt: str = "Explain the theory of general relativity in three concise sentences.",
        *,
        max_tokens: int = 128,
        runs: int = 3,
        warmup: bool = True,
    ) -> BenchmarkSuite:
        """Runs the performance benchmark asynchronously across all targets."""
        suite = BenchmarkSuite()
        normalized_targets = [BenchmarkTarget.from_spec(t) for t in targets]

        for target in normalized_targets:
            res = BenchmarkResult(target=target)
            messages = [{"role": "user", "content": prompt}]

            if warmup:
                with contextlib.suppress(Exception):
                    await self.manager.achat(
                        backend=target.backend,
                        model=target.model,
                        messages=messages,
                        max_tokens=16,
                        **target.options,
                    )

            for _ in range(runs):
                recorder = _RunRecorder(prompt)
                stream = await self.manager.achat(
                    backend=target.backend,
                    model=target.model,
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens,
                    **target.options,
                )
                if isinstance(stream, AsyncIterator):
                    async for chunk in stream:
                        recorder.observe(chunk)
                res.runs.append(recorder.finish())

            suite.results.append(res)

        return suite
