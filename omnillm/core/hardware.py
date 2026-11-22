import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from omnillm.core.model_info import (
    GIB,
    SOURCE_DESCRIPTIONS,
    ModelArchitecture,
    ModelInfo,
    QuantOption,
    estimated_architecture,
    estimated_model_info,
    quant_bpw,
    quant_quality,
)


class QuantFitStatus(StrEnum):
    GPU_FULL = "GPU (Full Offload)"
    METAL_UNIFIED = "Metal Unified Memory"
    PARTIAL = "GPU + CPU (Partial Offload)"
    CPU_RAM = "CPU / System RAM"
    OOM = "Insufficient Memory"


#: The standard quantization ladder used when a model's real files are unknown.
QUANT_SPECS: dict[str, dict[str, Any]] = {
    label: {"bytes_per_param": quant_bpw(label) / 8, "quality": quant_quality(label)}  # type: ignore[operator]
    for label in ("FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
}

#: Bytes per element of llama.cpp KV-cache types (`--cache-type-k/v`, Ollama's OLLAMA_KV_CACHE_TYPE).
KV_CACHE_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}

KNOWN_MODEL_SIZES: dict[str, float] = {
    "llama3.3": 70.0,
    "llama-3.3": 70.0,
    "llama3.2": 3.0,
    "llama-3.2": 3.0,
    "llama3.1": 8.0,
    "llama-3.1": 8.0,
    "llama3": 8.0,
    "llama-3": 8.0,
    "mistral-nemo": 12.0,
    "mistral-small": 24.0,
    "mixtral-8x7b": 46.7,
    "mixtral": 46.7,
    "mistral": 7.0,
    "phi-3-mini": 3.8,
    "phi3": 3.8,
    "phi-3": 3.8,
    "phi4-mini": 3.8,
    "phi4": 14.7,
    "phi-4": 14.7,
    "gemma3": 4.0,
    "gemma2": 9.0,
    "gemma-2": 9.0,
    "gemma": 7.0,
    "deepseek-r1": 7.0,
    "qwen3": 8.0,
    "qwen2.5": 7.0,
    "qwen2": 7.0,
    "gpt-oss": 20.9,
}

_USABLE_RAM_FRACTION = 0.95
_TOTAL_RAM_FRACTION = 0.85
_METAL_FRACTION = 0.75  # macOS GPU working-set limit for unified memory (default ~2/3-3/4 of RAM)
_MAX_RECOMMENDED_GPU_BPW = 8.5  # beyond Q8_0 quality gains are negligible for the memory/speed cost
_MAX_RECOMMENDED_OFFLOAD_BPW = 4.9  # Q4_K_M: the usual sweet spot when weights spill to system RAM
_MIN_PREFERRED_GPU_BPW = 3.5  # below this, a better quant split across GPU+CPU beats a tiny one on the GPU


@dataclass
class HardwareProfile:
    os_name: str
    os_release: str
    architecture: str
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_name: str | None = None
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    is_apple_silicon: bool = False
    has_cuda: bool = False

    def primary_memory_pool_gb(self) -> float:
        """Returns the primary memory pool (in GB) available for model acceleration."""
        if self.is_apple_silicon:
            return self.total_ram_gb * 0.75
        if self.has_cuda and self.total_vram_gb > 0:
            return self.total_vram_gb
        return self.available_ram_gb

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"OS: {self.os_name} {self.os_release} ({self.architecture}, {self.cpu_count} CPUs)",
            f"RAM: {self.total_ram_gb:.2f} GB Total ({self.available_ram_gb:.2f} GB Available)",
        ]
        if self.has_cuda and self.gpu_name:
            parts.append(
                f"GPU: {self.gpu_name} ({self.total_vram_gb:.2f} GB Total VRAM, {self.free_vram_gb:.2f} GB Free)"
            )
        elif self.is_apple_silicon:
            parts.append(f"GPU: {self.gpu_name or 'Apple Silicon Metal'} (Unified Memory Architecture)")
        else:
            parts.append("Accelerator: CPU Inference Only")
        return " | ".join(parts)


def detect_hardware() -> HardwareProfile:
    """Detects host operating system, CPU architecture, system RAM, and GPU accelerators (NVIDIA/Apple Silicon)."""
    os_name = platform.system()
    os_release = platform.release()
    arch = platform.machine()
    cpu_count = os.cpu_count() or 1

    total_ram_gb = 8.0
    available_ram_gb = 4.0

    # 1. Try psutil if available
    try:
        import psutil  # type: ignore[import-untyped]

        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024**3)
        available_ram_gb = vm.available / (1024**3)
    except Exception:
        # Fallback 1: Linux /proc/meminfo
        if os_name == "Linux" and os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_gb = int(line.split()[1]) / (1024 * 1024)
                        elif line.startswith("MemAvailable:"):
                            available_ram_gb = int(line.split()[1]) / (1024 * 1024)
            except Exception:
                pass
        # Fallback 2: macOS sysctl
        elif os_name == "Darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
                total_ram_gb = int(out) / (1024**3)
                available_ram_gb = total_ram_gb * 0.70  # safe heuristic
            except Exception:
                pass
        # Fallback 3: Windows GlobalMemoryStatusEx
        elif sys.platform == "win32":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
                total_ram_gb = stat.ullTotalPhys / (1024**3)
                available_ram_gb = stat.ullAvailPhys / (1024**3)
            except Exception:
                pass

    # 2. GPU Detection
    gpu_name: str | None = None
    gpu_count = 0
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    has_cuda = False
    is_apple_silicon = False

    # Check NVIDIA via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
            ).strip()
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if lines:
                gpu_count = len(lines)
                has_cuda = True
                first_parts = [p.strip() for p in lines[0].split(",")]
                gpu_name = first_parts[0]
                total_vram_gb = float(first_parts[1]) / 1024.0
                free_vram_gb = float(first_parts[2]) / 1024.0
        except Exception:
            pass

    # Check Apple Silicon
    if os_name == "Darwin" and arch in ("arm64", "aarch64"):
        is_apple_silicon = True
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2).strip()
            gpu_name = f"Apple Silicon ({chip})"
        except Exception:
            gpu_name = "Apple Silicon GPU (Metal)"
        total_vram_gb = total_ram_gb * 0.75
        free_vram_gb = available_ram_gb * 0.75

    return HardwareProfile(
        os_name=os_name,
        os_release=os_release,
        architecture=arch,
        cpu_count=cpu_count,
        total_ram_gb=round(total_ram_gb, 2),
        available_ram_gb=round(available_ram_gb, 2),
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_vram_gb=round(total_vram_gb, 2),
        free_vram_gb=round(free_vram_gb, 2),
        is_apple_silicon=is_apple_silicon,
        has_cuda=has_cuda,
    )


def parse_model_parameter_count(model_name: str) -> float | None:
    """Infers the parameter count in billions from common model names or aliases."""
    name = model_name.lower().strip()

    # Explicit MoE patterns: e.g. "8x7b"
    moe_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)[bB]", name)
    if moe_match:
        count = int(moe_match.group(1))
        size = float(moe_match.group(2))
        return count * size * 0.83  # Shared embedding and active expert factor

    # Explicit billion patterns: e.g. "8b", "70b", "3.8b", "0.5b"
    b_match = re.search(r"(?:^|[-_:a-zA-Z])(\d+(?:\.\d+)?)[bB](?:[-_:]|$)", name)
    if b_match:
        return float(b_match.group(1))

    # Explicit million patterns: e.g. "135m", "360m"
    m_match = re.search(r"(?:^|[-_:a-zA-Z])(\d+(?:\.\d+)?)[mM](?:[-_:]|$)", name)
    if m_match:
        return float(m_match.group(1)) / 1000.0

    # Fallback to known lookup
    for key, size in KNOWN_MODEL_SIZES.items():
        if key in name:
            return size

    return None


def estimate_kv_cache_gb(architecture: ModelArchitecture, context_length: int, kv_cache_type: str = "f16") -> float:
    """KV-cache size: 2 tensors x layers x KV heads x head dim x context (GQA/MLA aware). Sliding-window layers
    (Gemma 2/3, gpt-oss) need less in practice, so this errs on the safe side."""
    try:
        bytes_per_element = KV_CACHE_BYTES[kv_cache_type.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown KV cache type '{kv_cache_type}'. Choose from {sorted(KV_CACHE_BYTES)}.") from error
    return architecture.kv_bytes_per_token(bytes_per_element) * context_length / GIB


def estimate_overhead_gb(architecture: ModelArchitecture, context_length: int) -> float:
    """Runtime context plus compute buffers; grows with context and hidden size."""
    hidden = architecture.n_embd or 4096
    return 0.30 + 0.15 + 0.02 * (context_length / 1024) * (hidden / 4096)


def estimate_model_memory(
    param_count_b: float,
    quant: str = "Q4_K_M",
    context_length: int = 4096,
    *,
    architecture: ModelArchitecture | None = None,
    kv_cache_type: str = "f16",
) -> tuple[float, float, float]:
    """Estimates (total, weights, kv_cache) memory in GB for a model size, quantization, and context."""
    arch = architecture or estimated_architecture(param_count_b)
    bpw = quant_bpw(quant) or quant_bpw("Q4_K_M") or 4.85
    weights_gb = param_count_b * 1e9 * bpw / 8 / GIB
    kv_gb = estimate_kv_cache_gb(arch, context_length, kv_cache_type)
    total_gb = weights_gb + kv_gb + estimate_overhead_gb(arch, context_length)
    return round(total_gb, 2), round(weights_gb, 2), round(kv_gb, 2)


@dataclass
class QuantFitResult:
    quant: str
    required_memory_gb: float
    weights_memory_gb: float
    kv_cache_memory_gb: float
    status: QuantFitStatus
    fits: bool
    quality_description: str
    speed_description: str
    notes: str
    overhead_memory_gb: float = 0.0
    n_gpu_layers: int | None = None
    bits_per_weight: float | None = None
    files: tuple[str, ...] = ()
    installed: bool = False
    size_exact: bool = False
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["files"] = list(self.files)
        return d


@dataclass
class _Placement:
    status: QuantFitStatus
    fits: bool
    n_gpu_layers: int | None
    notes: str
    warning: str = ""


def _usable_vram_gb(hardware: HardwareProfile) -> float:
    if not hardware.has_cuda:
        return 0.0
    if hardware.free_vram_gb > 0:
        return hardware.free_vram_gb * 0.95
    return hardware.total_vram_gb * 0.90


def plan_placement(
    weights_gb: float, kv_gb: float, overhead_gb: float, n_layers: int, hardware: HardwareProfile
) -> _Placement:
    """Decides where a model runs: fully on GPU, Metal, split GPU/CPU (with a layer count), CPU, or not at all."""
    total = weights_gb + kv_gb + overhead_gb
    vram = _usable_vram_gb(hardware)
    ram_free = hardware.available_ram_gb * _USABLE_RAM_FRACTION
    ram_max = max(ram_free, hardware.total_ram_gb * _TOTAL_RAM_FRACTION)

    def tight(needed: float) -> str:
        if needed <= ram_free:
            return ""
        return f"Needs {needed:.1f} GB of RAM but only {hardware.available_ram_gb:.1f} GB is free now; close other apps first."

    if hardware.has_cuda and total <= vram:
        return _Placement(QuantFitStatus.GPU_FULL, True, -1, "Fits entirely in GPU VRAM.")
    if hardware.is_apple_silicon and total <= hardware.total_ram_gb * _METAL_FRACTION:
        return _Placement(QuantFitStatus.METAL_UNIFIED, True, -1, "Fits in Apple Silicon unified memory.")

    if hardware.has_cuda and vram > overhead_gb and n_layers > 1:
        per_layer = (weights_gb + kv_gb) / n_layers
        gpu_layers = min(int((vram - overhead_gb) // per_layer), n_layers - 1)
        cpu_gb = (weights_gb + kv_gb) - gpu_layers * per_layer
        if gpu_layers >= 1 and cpu_gb <= ram_max:
            return _Placement(
                QuantFitStatus.PARTIAL,
                True,
                gpu_layers,
                f"{gpu_layers}/{n_layers} layers on GPU, {cpu_gb:.1f} GB in system RAM.",
                tight(cpu_gb),
            )

    if total <= ram_max:
        return _Placement(QuantFitStatus.CPU_RAM, True, 0, "Runs from system RAM.", tight(total))
    return _Placement(QuantFitStatus.OOM, False, None, "Exceeds available memory.")


def _speed(status: QuantFitStatus, architecture: ModelArchitecture, n_gpu_layers: int | None) -> str:
    moe = architecture.is_moe
    if status in (QuantFitStatus.GPU_FULL, QuantFitStatus.METAL_UNIFIED):
        return "Fast"
    if status == QuantFitStatus.PARTIAL:
        share = (n_gpu_layers or 0) / max(architecture.n_layers, 1)
        return "Good (MoE)" if moe else ("Medium" if share >= 0.5 else "Slow-medium")
    if status == QuantFitStatus.CPU_RAM:
        return "Medium (MoE on CPU)" if moe else "Slow (CPU)"
    return "-"


@dataclass
class FitRecommendation:
    model_name: str
    param_count_b: float
    context_length: int
    hardware: HardwareProfile
    results: list[QuantFitResult]
    can_fit: bool
    recommended_quant: str | None
    best_target: QuantFitStatus
    summary_message: str
    suggestions: list[str] = field(default_factory=list)
    architecture: ModelArchitecture | None = None
    source: str = "estimated"
    kv_cache_type: str = "f16"
    repo: str | None = None
    revision: str = "main"
    notes: list[str] = field(default_factory=list)

    @property
    def recommended(self) -> QuantFitResult | None:
        return next((r for r in self.results if r.quant == self.recommended_quant), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "param_count_b": self.param_count_b,
            "context_length": self.context_length,
            "kv_cache_type": self.kv_cache_type,
            "source": self.source,
            "repo": self.repo,
            "architecture": self.architecture.to_dict() if self.architecture else None,
            "hardware": self.hardware.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "can_fit": self.can_fit,
            "recommended_quant": self.recommended_quant,
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "best_target": self.best_target.value,
            "summary_message": self.summary_message,
            "suggestions": self.suggestions,
            "notes": self.notes,
        }

    def format_table(self) -> str:
        hw = self.hardware
        lines = ["🖥️  Detected Hardware Profile", "-" * 100]
        lines.append(f"OS:           {hw.os_name} {hw.os_release} ({hw.architecture}, {hw.cpu_count} CPUs)")
        lines.append(f"System RAM:   {hw.total_ram_gb:.2f} GB Total ({hw.available_ram_gb:.2f} GB Available)")
        if hw.has_cuda and hw.gpu_name:
            lines.append(f"GPU:          {hw.gpu_name} ({hw.total_vram_gb:.2f} GB VRAM, {hw.free_vram_gb:.2f} GB Free)")
        elif hw.is_apple_silicon:
            lines.append(f"GPU:          {hw.gpu_name or 'Apple Silicon Metal'} (Unified Memory)")
        else:
            lines.append("Accelerator:  CPU Only (No dedicated GPU detected)")
        lines.append("")
        arch = self.architecture.describe() if self.architecture else f"~{self.param_count_b:.1f}B params"
        lines.append(f"🧠 Model Fit Analysis: {self.model_name}")
        lines.append(f"   {arch} | context {self.context_length:,} | KV cache {self.kv_cache_type}")
        lines.append(f"   Source: {SOURCE_DESCRIPTIONS.get(self.source, self.source)}")
        lines.append("-" * 100)
        lines.append(
            f"   {'QUANT':<14} {'BPW':>5} {'WEIGHTS':>9} {'KV':>7} {'TOTAL':>9}  {'FIT':<34} {'QUALITY':<18} SPEED"
        )
        lines.append("-" * 100)
        for r in self.results:
            marker = ("→" if r.quant == self.recommended_quant else " ") + ("*" if r.installed else " ")
            icon = (
                "✅"
                if r.status in (QuantFitStatus.GPU_FULL, QuantFitStatus.METAL_UNIFIED)
                else ("⚠️ " if r.fits else "❌")
            )
            status = r.status.value
            if r.status == QuantFitStatus.PARTIAL and r.n_gpu_layers is not None:
                status = f"GPU+CPU ({r.n_gpu_layers} layers on GPU)"
            bpw = f"{r.bits_per_weight:.2f}" if r.bits_per_weight else "?"
            approx = "" if r.size_exact else "~"
            weights = f"{approx}{r.weights_memory_gb:.2f}"
            lines.append(
                f"{marker} {r.quant:<14} {bpw:>5} {weights:>9} {r.kv_cache_memory_gb:>7.2f} "
                f"{r.required_memory_gb:>9.2f}  {icon} {status:<31} {r.quality_description:<18} {r.speed_description}"
            )
        lines.append("-" * 100)
        if any(r.installed for r in self.results):
            lines.append("  * installed   → recommended   ~ estimated size   (sizes in GB)")
        else:
            lines.append("  → recommended   ~ estimated size   (sizes in GB)")
        lines.append("\n💡 Recommendation:")
        lines.append(f"   {self.summary_message}")
        for suggestion in self.suggestions:
            lines.append(f"   {suggestion}")
        for note in self.notes:
            lines.append(f"   ℹ️  {note}")
        return "\n".join(lines)


def _quant_options(info: ModelInfo, param_count: int) -> list[QuantOption]:
    """Exact options for GGUF sources; otherwise the standard ladder, anchored on an installed quant if any."""
    if info.source in ("huggingface-gguf", "gguf-file") and info.quants:
        return list(info.quants)
    installed = next((q for q in info.quants if q.installed), None)
    if installed and installed.size_bytes and installed.bpw:
        param_count = int(installed.size_bytes * 8 / installed.bpw)  # effective params incl. embeddings
    ladder = [QuantOption(label, int(param_count * spec["bytes_per_param"])) for label, spec in QUANT_SPECS.items()]
    if installed:
        ladder = [q for q in ladder if q.label != installed.label] + [installed]
    return ladder


def recommend_model_fit(
    model_name: str,
    context_length: int = 4096,
    param_count_b: float | None = None,
    hardware: HardwareProfile | None = None,
    *,
    info: ModelInfo | None = None,
    kv_cache_type: str = "f16",
) -> FitRecommendation:
    """Evaluates every quantization of a model against the hardware and recommends one.

    Pass `info` (from `omnillm.core.model_info` / `omnillm.core.fit.load_model_info`) for exact architecture and
    file sizes; otherwise the model is estimated from its name or `param_count_b`.
    """
    hardware = hardware or detect_hardware()
    if info is None or (param_count_b is not None and info.source == "estimated"):
        info = estimated_model_info(model_name, param_count_b)
    arch = info.architecture
    param_count = arch.param_count or int((param_count_b or 7.0) * 1e9)
    exact_sizes = info.source in ("huggingface-gguf", "gguf-file")

    kv_gb = estimate_kv_cache_gb(arch, context_length, kv_cache_type)
    overhead_gb = estimate_overhead_gb(arch, context_length)

    results: list[QuantFitResult] = []
    for option in _quant_options(info, param_count):
        size = option.size_bytes or int(param_count * (option.bpw or 4.85) / 8)
        weights_gb = size / GIB
        bpw = option.bpw or (size * 8 / param_count if param_count else None)
        placement = plan_placement(weights_gb, kv_gb, overhead_gb, arch.n_layers, hardware)
        results.append(
            QuantFitResult(
                quant=option.label,
                required_memory_gb=round(weights_gb + kv_gb + overhead_gb, 2),
                weights_memory_gb=round(weights_gb, 2),
                kv_cache_memory_gb=round(kv_gb, 2),
                status=placement.status,
                fits=placement.fits,
                quality_description=quant_quality(option.label) if option.bpw else "Unknown",
                speed_description=_speed(placement.status, arch, placement.n_gpu_layers),
                notes=placement.notes,
                overhead_memory_gb=round(overhead_gb, 2),
                n_gpu_layers=placement.n_gpu_layers,
                bits_per_weight=round(bpw, 2) if bpw else None,
                files=option.files,
                installed=option.installed,
                warning=placement.warning,
                size_exact=exact_sizes or (option.installed and option.size_bytes is not None),
            )
        )
    results.sort(key=lambda r: r.weights_memory_gb, reverse=True)

    def bits(result: QuantFitResult) -> float:
        return result.bits_per_weight or 0.0

    fitting = [r for r in results if r.fits]
    accelerated = [r for r in fitting if r.status in (QuantFitStatus.GPU_FULL, QuantFitStatus.METAL_UNIFIED)]
    suggestions: list[str] = []
    notes = list(info.notes)
    best: QuantFitResult | None = None

    good_accelerated = [r for r in accelerated if bits(r) >= _MIN_PREFERRED_GPU_BPW]
    offloaded = [r for r in fitting if r not in accelerated]
    if good_accelerated:
        capped = [r for r in good_accelerated if bits(r) <= _MAX_RECOMMENDED_GPU_BPW]
        best = (capped or good_accelerated)[0]
    elif offloaded:
        capped = [r for r in offloaded if bits(r) <= _MAX_RECOMMENDED_OFFLOAD_BPW]
        best = capped[0] if capped else offloaded[-1]
        if accelerated:
            alt = accelerated[0]
            suggestions.append(
                f"💡 {alt.quant} ({alt.required_memory_gb:.2f} GB) would fit entirely on the GPU: much faster, lower quality."
            )
    elif accelerated:
        best = accelerated[0]
    if best is not None and bits(best) < 3.6 and best.bits_per_weight is not None:
        suggestions.append("⚠️  Only heavily compressed quants fit; quality will suffer. Consider a smaller model.")

    if best is not None:
        target = best.status.value
        if best.status == QuantFitStatus.PARTIAL:
            target = f"GPU + CPU with n_gpu_layers={best.n_gpu_layers} of {arch.n_layers}"
        summary = f"Recommended: {best.quant} → {target} (~{best.required_memory_gb:.2f} GB total)."
        if best.warning:
            suggestions.insert(0, f"⚠️  {best.warning}")
        if kv_cache_type.lower() in ("f16", "bf16", "f32") and kv_gb >= 0.25 * best.required_memory_gb:
            q8 = estimate_kv_cache_gb(arch, context_length, "q8_0")
            suggestions.append(f"💡 --kv-cache q8_0 would cut the KV cache from {kv_gb:.2f} GB to {q8:.2f} GB.")
        if best.status != QuantFitStatus.GPU_FULL and hardware.has_cuda and arch.is_moe:
            notes.append(
                "MoE model: only a fraction of the weights is used per token, so partial offload / CPU runs "
                "much faster than a dense model of the same size. With llama.cpp, keeping the expert tensors on "
                "CPU and the rest on GPU (--n-cpu-moe or -ot '.ffn_.*_exps.=CPU') is usually fastest."
            )
    else:
        summary = (
            f"{model_name} (~{param_count / 1e9:.1f}B) does not fit in this machine's memory "
            f"({hardware.total_ram_gb:.1f} GB RAM"
            + (f", {hardware.total_vram_gb:.1f} GB VRAM" if hardware.has_cuda else "")
            + ")."
        )
        suggestions.append("💡 Try a smaller model, a shorter --context, or --kv-cache q4_0.")

    if arch.context_length and context_length > arch.context_length:
        notes.append(f"Context {context_length:,} exceeds the model's trained context ({arch.context_length:,}).")

    return FitRecommendation(
        model_name=model_name,
        param_count_b=round(param_count / 1e9, 3),
        context_length=context_length,
        hardware=hardware,
        results=results,
        can_fit=bool(fitting),
        recommended_quant=best.quant if best else None,
        best_target=best.status if best else QuantFitStatus.OOM,
        summary_message=summary,
        suggestions=suggestions,
        architecture=arch,
        source=info.source,
        kv_cache_type=kv_cache_type,
        repo=info.repo,
        revision=info.revision,
        notes=notes,
    )


def suggest_models_for_hardware(hardware: HardwareProfile | None = None) -> list[dict[str, Any]]:
    """Rates popular local models (estimated from their sizes) against the detected hardware."""
    hardware = hardware or detect_hardware()

    candidate_models = [
        ("smollm:135m", 0.135, None, "Ultra-fast lightweight model for embedded & testing"),
        ("llama3.2:1b", 1.0, None, "High-efficiency edge & mobile model"),
        ("llama3.2:3b", 3.2, None, "Strong quality-to-size ratio for laptops"),
        ("qwen3:4b", 4.0, None, "Small reasoning model with tool calling"),
        ("qwen3:8b", 8.2, None, "General-purpose workhorse with thinking mode"),
        ("gemma3:12b", 12.0, None, "Multimodal (vision) mid-size model"),
        ("phi4:14b", 14.7, None, "Strong reasoning for its size"),
        ("gpt-oss:20b", 20.9, 3.6, "MoE: ~3.6B active, fast even with partial offload"),
        ("mistral-small:24b", 24.0, None, "Capable mid-large model for workstations"),
        ("qwen3:30b-a3b", 30.5, 3.3, "MoE: ~3.3B active, fast even with partial offload"),
        ("qwen3:32b", 32.8, None, "Near-frontier dense model for workstations"),
        ("llama3.3:70b", 70.6, None, "Large dense model for multi-GPU / high-memory Macs"),
    ]

    suggestions = []
    for name, size, active, desc in candidate_models:
        info = ModelInfo(name=name, architecture=estimated_architecture(size, active), source="estimated")
        fit = recommend_model_fit(name, context_length=4096, hardware=hardware, info=info)
        best = fit.recommended
        suggestions.append(
            {
                "model": name,
                "params": f"{size}B",
                "can_fit": fit.can_fit,
                "target": fit.best_target.value,
                "recommended_quant": fit.recommended_quant,
                "n_gpu_layers": best.n_gpu_layers if best else None,
                "required_gb": best.required_memory_gb if best else 0.0,
                "description": desc,
            }
        )
    return suggestions
