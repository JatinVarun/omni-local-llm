import argparse
import sys
from unittest.mock import patch

import pytest

from omnillm import ChatChunk, ChatResponse, LocalLLMManager
from omnillm.__main__ import (
    _handle_bench,
    _handle_eval,
    _handle_models,
    _handle_run,
    build_parser,
    main,
)
from tests.fakes import RecordingBackend


def test_build_parser_subcommands():
    parser = build_parser()

    # Chat
    args = parser.parse_args(["chat", "--model", "llama3", "--max-turns", "5"])
    assert args.subcommand == "chat"
    assert args.model == "llama3"
    assert args.max_turns == 5

    # Run
    args = parser.parse_args(["run", "--model", "llama3", "Hello world"])
    assert args.subcommand == "run"
    assert args.prompt == "Hello world"

    # Bench
    args = parser.parse_args(["bench", "--targets", "ollama/llama3", "--runs", "2"])
    assert args.subcommand == "bench"
    assert args.targets == "ollama/llama3"
    assert args.runs == 2

    # Eval
    args = parser.parse_args(["eval", "--model", "llama3", "--needle"])
    assert args.subcommand == "eval"
    assert args.needle is True

    # Models
    args = parser.parse_args(["models", "--backend", "ollama"])
    assert args.subcommand == "models"
    assert args.backend == "ollama"

    # Serve
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.subcommand == "serve"
    assert args.port == 9000

    # Fit
    args = parser.parse_args(["fit", "llama-3.1-8b", "--context", "8192"])
    assert args.subcommand == "fit"
    assert args.model == "llama-3.1-8b"
    assert args.context == 8192


def test_cli_handle_run(capsys):
    backend = RecordingBackend(response=ChatResponse(content="One-shot response"))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        backend="fake",
        model="test",
        prompt="Tell me something",
        temperature=None,
        filename=None,
        stream=False,
    )
    _handle_run(args, manager)

    captured = capsys.readouterr()
    assert "One-shot response" in captured.out


def test_cli_handle_run_stream(capsys):
    backend = RecordingBackend(chunks=(ChatChunk(content="Streamed "), ChatChunk(content="chunk")))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        backend="fake",
        model="test",
        prompt="Stream this",
        temperature=0.7,
        filename=None,
        stream=True,
    )
    _handle_run(args, manager)

    captured = capsys.readouterr()
    assert "Streamed chunk" in captured.out


def test_cli_handle_models(capsys):
    backend = RecordingBackend(models=("m1", "m2"))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(backend="fake")
    _handle_models(args, manager)

    captured = capsys.readouterr()
    assert "fake" in captured.out
    assert "m1" in captured.out
    assert "m2" in captured.out


def test_cli_handle_bench(capsys):
    backend = RecordingBackend(chunks=(ChatChunk(content="Tok 1 "), ChatChunk(content="Tok 2")))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        targets=None,
        backend="fake",
        model="test",
        prompt="Test prompt",
        tokens=32,
        runs=1,
        no_warmup=True,
        json=False,
    )
    _handle_bench(args, manager)

    captured = capsys.readouterr()
    assert "Tokens/sec" in captured.out
    assert "fake" in captured.out


def test_cli_handle_eval(capsys):
    backend = RecordingBackend(response=ChatResponse(content="The answer is Paris."))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        needle=False,
        suite=None,
        backend="fake",
        model="test",
        json=False,
    )
    _handle_eval(args, manager)

    captured = capsys.readouterr()
    assert "Model Evaluation" in captured.out
    assert "PASS" in captured.out


def test_cli_handle_fit_with_model(capsys):
    from omnillm.__main__ import _handle_fit

    args = argparse.Namespace(
        model="llama-3.2-3b",
        context=4096,
        params=None,
        json=False,
    )
    _handle_fit(args)
    captured = capsys.readouterr()
    assert "llama-3.2-3b" in captured.out
    assert "Detected Hardware Profile" in captured.out
    assert "QUANT" in captured.out


def test_cli_handle_fit_matrix(capsys):
    from omnillm.__main__ import _handle_fit

    args = argparse.Namespace(
        model=None,
        context=4096,
        params=None,
        json=False,
    )
    _handle_fit(args)
    captured = capsys.readouterr()
    assert "Model Compatibility Matrix" in captured.out
    assert "smollm:135m" in captured.out


def test_cli_main_help_output(capsys):
    with patch.object(sys, "argv", ["omnillm"]):
        main()
    captured = capsys.readouterr()
    assert "Omni-Local-LLM" in captured.out
    assert "chat" in captured.out
    assert "bench" in captured.out
    assert "fit" in captured.out


def test_cli_config_option_works_before_or_after_subcommand():
    parser = build_parser()

    assert parser.parse_args(["--config", "a.toml", "run", "--model", "m", "hi"]).config == "a.toml"
    assert parser.parse_args(["run", "--config", "b.toml", "--model", "m", "hi"]).config == "b.toml"
    assert parser.parse_args(["run", "--model", "m", "hi"]).backend is None


def test_cli_base_url_registers_ad_hoc_openai_backend(tmp_path, monkeypatch):
    from omnillm import OpenAICompatibleAdapter
    from omnillm.__main__ import build_manager

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OMNILLM_CONFIG", raising=False)
    args = build_parser().parse_args(["run", "--base-url", "http://localhost:1234", "--model", "qwen3", "hi"])

    manager = build_manager(args)

    assert args.backend == "openai"
    assert manager.resolve("qwen3").ref == "openai/qwen3"
    adapter = manager._get_backend("openai")
    assert isinstance(adapter, OpenAICompatibleAdapter) and adapter.base_url == "http://localhost:1234/v1"


def test_cli_models_lists_aliases(capsys):
    manager = LocalLLMManager({"fake": RecordingBackend(models=["llama3"])}, default_backend="fake")
    manager.register_alias("coder", model="fake/qwen", fallbacks=["fake/llama3"])

    _handle_models(argparse.Namespace(backend=None), manager)

    out = capsys.readouterr().out
    assert "coder" in out and "fake/qwen" in out and "fake/llama3" in out


def test_cli_reports_config_errors(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[models.a]\ntemprature = 1\n")
    monkeypatch.setattr(sys, "argv", ["omnillm", "--config", str(bad), "models"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    assert "Config error" in capsys.readouterr().err
