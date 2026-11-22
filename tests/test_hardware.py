from omnillm.core.hardware import (
    HardwareProfile,
    QuantFitStatus,
    detect_hardware,
    estimate_model_memory,
    parse_model_parameter_count,
    recommend_model_fit,
    suggest_models_for_hardware,
)


def test_detect_hardware() -> None:
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.cpu_count >= 1
    assert hw.total_ram_gb > 0
    assert hw.available_ram_gb >= 0
    assert isinstance(hw.to_dict(), dict)
    assert len(hw.summary()) > 0


def test_parse_model_parameter_count() -> None:
    assert parse_model_parameter_count("llama-3.1-8b") == 8.0
    assert parse_model_parameter_count("llama3:8b") == 8.0
    assert parse_model_parameter_count("llama-3.2-1b") == 1.0
    assert parse_model_parameter_count("llama-3.2-3b") == 3.0
    assert parse_model_parameter_count("llama-3.3-70b") == 70.0
    assert parse_model_parameter_count("phi-3-mini") == 3.8
    assert parse_model_parameter_count("mistral:7b") == 7.0
    assert parse_model_parameter_count("smollm:135m") == 0.135
    assert parse_model_parameter_count("smollm2:360m") == 0.36
    assert parse_model_parameter_count("mixtral-8x7b") is not None
    assert parse_model_parameter_count("unknown-random-model-xyz") is None


def test_estimate_model_memory() -> None:
    # 8B model memory estimation
    total_fp16, w_fp16, kv_fp16 = estimate_model_memory(8.0, quant="FP16", context_length=4096)
    total_q8, w_q8, kv_q8 = estimate_model_memory(8.0, quant="Q8_0", context_length=4096)
    total_q4, w_q4, kv_q4 = estimate_model_memory(8.0, quant="Q4_K_M", context_length=4096)
    total_q2, w_q2, kv_q2 = estimate_model_memory(8.0, quant="Q2_K", context_length=4096)

    # Weights scale monotonically with quantization bitwidth
    assert w_fp16 > w_q8 > w_q4 > w_q2
    assert total_fp16 > total_q8 > total_q4 > total_q2

    # Higher context window increases KV cache
    total_4k, _, kv_4k = estimate_model_memory(8.0, quant="Q4_K_M", context_length=4096)
    total_8k, _, kv_8k = estimate_model_memory(8.0, quant="Q4_K_M", context_length=8192)
    assert kv_8k > kv_4k
    assert total_8k > total_4k


def test_recommend_model_fit_cuda_high_vram() -> None:
    # Mock RTX 4090 with 24 GB VRAM
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=16,
        total_ram_gb=64.0,
        available_ram_gb=48.0,
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        total_vram_gb=24.0,
        free_vram_gb=23.0,
        has_cuda=True,
        is_apple_silicon=False,
    )

    rec = recommend_model_fit("llama-3.1-8b", context_length=4096, hardware=hw)
    assert rec.can_fit is True
    assert rec.best_target == QuantFitStatus.GPU_FULL
    assert rec.recommended_quant in ("Q8_0", "Q5_K_M", "Q4_K_M")
    assert "RTX 4090" in rec.format_table()


def test_recommend_model_fit_cuda_low_vram() -> None:
    # Mock GTX 1650 with 4 GB VRAM and 16 GB RAM
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=8,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        gpu_name="NVIDIA GeForce GTX 1650",
        gpu_count=1,
        total_vram_gb=4.0,
        free_vram_gb=3.5,
        has_cuda=True,
        is_apple_silicon=False,
    )

    # 3B model fits in GPU VRAM
    rec3b = recommend_model_fit("llama-3.2-3b", hardware=hw)
    assert rec3b.can_fit is True
    assert rec3b.best_target == QuantFitStatus.GPU_FULL

    # 8B model doesn't fit in 4 GB VRAM: split between GPU and CPU with a concrete layer count
    rec8b = recommend_model_fit("llama-3.1-8b", hardware=hw)
    assert rec8b.can_fit is True
    assert rec8b.best_target == QuantFitStatus.PARTIAL
    assert rec8b.recommended_quant == "Q4_K_M"
    assert rec8b.recommended is not None
    assert 1 <= (rec8b.recommended.n_gpu_layers or 0) < 32


def test_recommend_model_fit_apple_silicon() -> None:
    # Mock Apple M3 Max with 64 GB Unified Memory
    hw = HardwareProfile(
        os_name="Darwin",
        os_release="23.4.0",
        architecture="arm64",
        cpu_count=14,
        total_ram_gb=64.0,
        available_ram_gb=50.0,
        gpu_name="Apple Silicon (Apple M3 Max)",
        gpu_count=1,
        total_vram_gb=48.0,
        free_vram_gb=37.5,
        has_cuda=False,
        is_apple_silicon=True,
    )

    rec70b = recommend_model_fit("llama-3.3-70b", hardware=hw)
    assert rec70b.can_fit is True
    assert rec70b.best_target == QuantFitStatus.METAL_UNIFIED
    assert rec70b.recommended_quant == "Q4_K_M"


def test_recommend_model_fit_oom() -> None:
    # Mock low-spec machine with 8 GB RAM and no GPU
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=4,
        total_ram_gb=8.0,
        available_ram_gb=4.0,
        has_cuda=False,
        is_apple_silicon=False,
    )

    rec70b = recommend_model_fit("llama-3.3-70b", hardware=hw)
    assert rec70b.can_fit is False
    assert rec70b.best_target == QuantFitStatus.OOM


def test_suggest_models_for_hardware() -> None:
    hw = HardwareProfile(
        os_name="Linux",
        os_release="6.8.0",
        architecture="x86_64",
        cpu_count=8,
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        has_cuda=False,
        is_apple_silicon=False,
    )

    suggestions = suggest_models_for_hardware(hw)
    assert len(suggestions) > 0
    # 135m should fit easily on 16GB RAM
    smollm = next(s for s in suggestions if "135m" in s["model"])
    assert smollm["can_fit"] is True
    # 70B should not fit on 16GB RAM
    llama70b = next(s for s in suggestions if "70b" in s["model"])
    assert llama70b["can_fit"] is False


# ---------------------------------------------------------------- real-metadata planning

from dataclasses import replace  # noqa: E402

import pytest  # noqa: E402

from omnillm.core.hardware import plan_placement  # noqa: E402
from omnillm.core.model_info import ModelInfo, QuantOption, architecture_from_gguf  # noqa: E402

GIB = 1024**3


def _gpu(vram: float = 4.0, free: float = 3.6, ram: float = 16.0, available: float = 12.0) -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        os_release="6.8",
        architecture="x86_64",
        cpu_count=8,
        total_ram_gb=ram,
        available_ram_gb=available,
        gpu_name="GPU",
        gpu_count=1,
        total_vram_gb=vram,
        free_vram_gb=free,
        has_cuda=True,
        is_apple_silicon=False,
    )


def _llama3b_info(quants: list[QuantOption], source: str = "huggingface-gguf") -> ModelInfo:
    arch = architecture_from_gguf(
        {
            "general.architecture": "llama",
            "general.parameter_count": 3_212_749_888,
            "llama.block_count": 28,
            "llama.embedding_length": 3072,
            "llama.attention.head_count": 24,
            "llama.attention.head_count_kv": 8,
            "llama.context_length": 131072,
        }
    )
    assert arch is not None
    return ModelInfo(
        name="org/Llama-3.2-3B-GGUF", architecture=arch, source=source, quants=quants, repo="org/Llama-3.2-3B-GGUF"
    )


def test_parse_moe_name_is_not_mistaken_for_dense_size():
    assert parse_model_parameter_count("mixtral-8x7b") == pytest.approx(8 * 7 * 0.83)
    assert parse_model_parameter_count("qwen3-30b-a3b") == 30.0


def test_plan_placement_tiers():
    hw = _gpu()
    assert plan_placement(2.0, 0.5, 0.5, 28, hw).status == QuantFitStatus.GPU_FULL

    partial = plan_placement(6.0, 1.0, 0.5, 32, hw)
    assert partial.status == QuantFitStatus.PARTIAL
    per_layer = 7.0 / 32
    assert partial.n_gpu_layers == int((3.6 * 0.95 - 0.5) // per_layer)
    assert not partial.warning

    tight = plan_placement(12.0, 1.0, 0.5, 32, _gpu(available=5.0))
    assert tight.status == QuantFitStatus.PARTIAL and "close other apps" in tight.warning

    cpu_only = HardwareProfile("Linux", "6.8", "x86_64", 8, 32.0, 30.0)
    assert plan_placement(10.0, 1.0, 0.5, 32, cpu_only).status == QuantFitStatus.CPU_RAM
    assert plan_placement(40.0, 1.0, 0.5, 32, cpu_only).status == QuantFitStatus.OOM

    mac = HardwareProfile("Darwin", "23", "arm64", 12, 32.0, 20.0, is_apple_silicon=True)
    assert plan_placement(20.0, 1.0, 0.5, 32, mac).status == QuantFitStatus.METAL_UNIFIED


def test_exact_quants_recommend_best_full_gpu_option_capped_at_q8():
    quants = [
        QuantOption("F16", int(6.0 * GIB), ("m-F16.gguf",)),
        QuantOption("Q8_0", int(3.2 * GIB), ("m-Q8_0.gguf",)),
        QuantOption("Q4_K_M", int(1.9 * GIB), ("m-Q4_K_M.gguf",)),
    ]
    big_gpu = _gpu(vram=24.0, free=23.0, ram=64, available=60)

    rec = recommend_model_fit("org/Llama-3.2-3B-GGUF", 8192, hardware=big_gpu, info=_llama3b_info(quants))

    assert rec.recommended_quant == "Q8_0"  # F16 fits too, but Q8_0 is the cap
    assert rec.source == "huggingface-gguf"
    assert all(r.size_exact for r in rec.results)
    assert rec.recommended is not None and rec.recommended.files == ("m-Q8_0.gguf",)
    assert rec.recommended.kv_cache_memory_gb == pytest.approx(0.88, abs=0.01)
    assert rec.to_dict()["recommended"]["files"] == ["m-Q8_0.gguf"]


def test_decent_quant_fully_on_gpu_beats_better_quant_split_across_gpu_and_cpu():
    quants = [QuantOption("Q4_K_M", int(4.6 * GIB), ("q4.gguf",)), QuantOption("IQ3_M", int(2.2 * GIB), ("iq3.gguf",))]

    rec = recommend_model_fit("m", 4096, hardware=_gpu(), info=_llama3b_info(quants))

    assert rec.recommended_quant == "IQ3_M"
    assert rec.best_target == QuantFitStatus.GPU_FULL


def test_partial_offload_prefers_q4_over_heavily_compressed_gpu_only_quant():
    quants = [
        QuantOption("Q8_0", int(8.0 * GIB), ("q8.gguf",)),
        QuantOption("Q4_K_M", int(4.6 * GIB), ("q4.gguf",)),
        QuantOption("Q2_K", int(1.5 * GIB), ("q2.gguf",)),
    ]
    rec = recommend_model_fit("m", 4096, hardware=_gpu(), info=_llama3b_info(quants))

    assert rec.recommended_quant == "Q4_K_M"
    assert rec.best_target == QuantFitStatus.PARTIAL
    assert "n_gpu_layers=" in rec.summary_message
    assert any("Q2_K" in s and "entirely on the GPU" in s for s in rec.suggestions)


def test_kv_cache_hint_trained_context_note_and_kv_type():
    quants = [QuantOption("Q4_K_M", int(1.9 * GIB), ("q4.gguf",))]
    info = _llama3b_info(quants)
    info.architecture = replace(info.architecture, context_length=8192)

    rec = recommend_model_fit("m", 32768, hardware=_gpu(vram=24, free=23), info=info)
    q8 = recommend_model_fit("m", 32768, hardware=_gpu(vram=24, free=23), info=info, kv_cache_type="q8_0")

    assert any("--kv-cache q8_0" in s for s in rec.suggestions)
    assert any("exceeds the model's trained context" in n for n in rec.notes)
    assert q8.results[0].kv_cache_memory_gb < rec.results[0].kv_cache_memory_gb / 1.8
    with pytest.raises(ValueError, match="Unknown KV cache type"):
        recommend_model_fit("m", 4096, hardware=_gpu(), info=info, kv_cache_type="q3")


def test_installed_ollama_quant_anchors_the_estimated_ladder():
    installed = QuantOption("Q4_K_M", 2_019_393_189, installed=True)
    rec = recommend_model_fit("llama3.2:3b", 4096, hardware=_gpu(), info=_llama3b_info([installed], source="ollama"))

    labels = [r.quant for r in rec.results]
    assert labels.count("Q4_K_M") == 1 and "Q8_0" in labels and "FP16" in labels
    row = next(r for r in rec.results if r.quant == "Q4_K_M")
    assert row.installed and row.size_exact and row.weights_memory_gb == pytest.approx(1.88, abs=0.01)
    assert not next(r for r in rec.results if r.quant == "Q8_0").size_exact
    assert "*" in rec.format_table()


def test_moe_note_when_offloading():
    info = _llama3b_info([QuantOption("Q4_K_M", int(16 * GIB), ("q4.gguf",))])
    info.architecture = replace(info.architecture, expert_count=128, expert_used_count=8, active_param_count=int(3.3e9))

    rec = recommend_model_fit("moe", 4096, hardware=_gpu(ram=32, available=28), info=info)

    assert rec.best_target == QuantFitStatus.PARTIAL
    assert any("MoE model" in n and "--n-cpu-moe" in n for n in rec.notes)
    assert rec.results[0].speed_description == "Good (MoE)"
