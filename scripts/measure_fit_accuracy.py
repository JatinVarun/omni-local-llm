"""Measure how close `omnillm fit` is on your machine (NVIDIA GPU + Ollama).

For each installed Ollama model and context length, it compares fit's prediction with reality:
loads the model with fit's recommended GPU layer count, reads the VRAM actually used (nvidia-smi, per process),
and compares decode speed with Ollama's own automatic placement.

    uv run python scripts/measure_fit_accuracy.py llama3.2:3b qwen2.5:7b --contexts 4096,8192 --runs 3

Please share the printed table in a GitHub issue: it's the most useful feedback for improving fit.
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import httpx

from omnillm.core.fit import analyze_model_fit
from omnillm.core.hardware import QuantFitStatus, detect_hardware

GIB = 1024**3


class Ollama:
    def __init__(self, host: str):
        self.http = httpx.Client(base_url=host, timeout=900)

    def loaded(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = self.http.get("/api/ps").json().get("models", [])
        return models

    def unload_all(self) -> None:
        for model in self.loaded():
            self.http.post("/api/generate", json={"model": model["name"], "keep_alive": 0})
        for _ in range(30):
            if not self.loaded() and gpu_mib_used_by_ollama() == 0:
                return
            time.sleep(1)

    def run(self, model: str, ctx: int, num_gpu: int | None) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": ctx, "num_predict": 48, "temperature": 0, "seed": 1}
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        body = {"model": model, "prompt": "Write two sentences about the sea.", "options": options, "stream": False}
        response = self.http.post("/api/generate", json={**body, "keep_alive": "5m"})
        if response.status_code != 200:
            return {"ok": False, "error": response.text[:200]}
        data = response.json()
        time.sleep(1.5)
        ps = next((m for m in self.loaded() if m["name"] == model), {})
        return {
            "ok": True,
            "vram_gb": gpu_mib_used_by_ollama() / 1024,
            "tokens_per_second": data["eval_count"] / (data["eval_duration"] / 1e9),
            "ollama_size_gb": (ps.get("size") or 0) / GIB,
            "ollama_vram_gb": (ps.get("size_vram") or 0) / GIB,
        }


def gpu_mib_used_by_ollama() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"], text=True
    )
    return sum(
        int(used) for name, used in (line.rsplit(",", 1) for line in out.splitlines() if line) if "ollama" in name
    )


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in runs if r.get("ok")]
    if not ok:
        return {"ok": False, "error": runs[-1].get("error") if runs else "not run"}
    return {
        "ok": True,
        "vram_gb": max(r["vram_gb"] for r in ok),
        "tokens_per_second": statistics.median(r["tokens_per_second"] for r in ok),
        "ollama_size_gb": ok[-1]["ollama_size_gb"],
        "ollama_vram_gb": ok[-1]["ollama_vram_gb"],
    }


def measure(ollama: Ollama, model: str, ctx: int, runs: int) -> dict[str, Any]:
    import ollama as ollama_client

    ollama.unload_all()
    hardware = detect_hardware()
    rec = analyze_model_fit(model, context_length=ctx, hardware=hardware, ollama_client=ollama_client)
    row = next(r for r in rec.results if r.installed)
    arch = rec.architecture
    assert arch is not None
    full = row.status in (QuantFitStatus.GPU_FULL, QuantFitStatus.METAL_UNIFIED)
    layers = arch.n_layers if full else (row.n_gpu_layers or 0)
    per_layer = (row.weights_memory_gb + row.kv_cache_memory_gb) / arch.n_layers
    predicted_vram = row.overhead_memory_gb + layers * per_layer if layers else 0.0

    auto_runs, fit_runs = [], []
    for _ in range(runs):
        ollama.unload_all()
        auto_runs.append(ollama.run(model, ctx, None))
        if layers:
            ollama.unload_all()
            # Ollama counts the output layer as one more layer: offload it too when fit says "all on GPU".
            fit_runs.append(ollama.run(model, ctx, layers + 1 if full else layers))
    return {
        "model": model,
        "quant": row.quant,
        "context": ctx,
        "gpu": hardware.gpu_name,
        "free_vram_gb": hardware.free_vram_gb,
        "fit_status": row.status.value,
        "fit_gpu_layers": layers,
        "n_layers": arch.n_layers,
        "fit_total_gb": row.required_memory_gb,
        "fit_predicted_vram_gb": round(predicted_vram, 2),
        "at_fit_layers": summarize(fit_runs),
        "ollama_auto": summarize(auto_runs),
    }


def markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model (Ollama) | Quant | Context | fit says | Predicted VRAM | Measured VRAM | tok/s at fit's layers | tok/s Ollama auto |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        at, auto = r["at_fit_layers"], r["ollama_auto"]
        placement = (
            "all on GPU"
            if r["fit_gpu_layers"] == r["n_layers"]
            else f"{r['fit_gpu_layers']}/{r['n_layers']} layers on GPU"
        )
        measured = f"{at['vram_gb']:.2f} GB" if at.get("ok") else f"failed: {at.get('error', '')[:40]}"
        fit_tps = f"{at['tokens_per_second']:.1f}" if at.get("ok") else "-"
        auto_tps = f"{auto['tokens_per_second']:.1f}" if auto.get("ok") else "-"
        lines.append(
            f"| {r['model']} | {r['quant']} | {r['context']:,} | {placement} | {r['fit_predicted_vram_gb']:.2f} GB "
            f"| {measured} | {fit_tps} | {auto_tps} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Installed Ollama models, e.g. llama3.2:3b")
    parser.add_argument("--contexts", default="4096,8192", help="Comma-separated context lengths")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration (speeds are medians)")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama URL")
    parser.add_argument("--json", help="Also write raw results to this file")
    args = parser.parse_args()

    if not shutil.which("nvidia-smi"):
        sys.exit("This script needs an NVIDIA GPU (nvidia-smi).")
    ollama = Ollama(args.host)
    try:
        ollama.loaded()
    except httpx.HTTPError:
        sys.exit(f"Ollama is not reachable at {args.host}.")

    results = []
    for model in args.models:
        for ctx in (int(c) for c in args.contexts.split(",")):
            print(f"Measuring {model} @ {ctx:,} context ...", file=sys.stderr)
            results.append(measure(ollama, model, ctx, args.runs))
    ollama.unload_all()

    hw = detect_hardware()
    print(f"\n{hw.gpu_name} ({hw.total_vram_gb:.1f} GB VRAM), {hw.total_ram_gb:.0f} GB RAM\n")
    print(markdown(results))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
