"""Tests for the thorn CLI — specifically ``thorn run --result-file``."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from thorn._cli import main as cli_main
from thorn._provider import (
    FinishChunk,
    MockProvider,
    TextChunk,
    ToolCallChunk,
    UsageChunk,
)


def _mock_provider_factory(provider: MockProvider):
    """Return a callable that ignores arguments and returns *provider*."""
    def factory() -> MockProvider:
        return provider
    return factory


class TestRunResultFile:
    """``thorn run --result-file`` writes structured JSON."""

    def test_success_writes_result(self, tmp_path: Path, monkeypatch):
        provider = MockProvider(canned_responses=[
            [
                TextChunk(text="done"),
                UsageChunk(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                FinishChunk(reason="stop"),
            ],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        monkeypatch.chdir(tmp_path)

        result_file = tmp_path / "result.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "run", "say hello",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
                "--result-file", str(result_file),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["outcome"] == "success"
        assert data["error"] is None
        assert data["duration_s"] >= 0
        assert data["token_usage"]["prompt_tokens"] == 10
        assert data["token_usage"]["completion_tokens"] == 5
        assert data["token_usage"]["total_tokens"] == 15

    def test_agent_error_writes_result(self, tmp_path: Path, monkeypatch):
        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(
                    call_id="c1",
                    name="raise_error",
                    arguments='{"message": "something went wrong"}',
                ),
                FinishChunk(reason="tool_calls"),
            ],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        monkeypatch.chdir(tmp_path)

        result_file = tmp_path / "result.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "run", "do something",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
                "--result-file", str(result_file),
                "--quiet",
            ],
        )
        assert result.exit_code == 1

        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["outcome"] == "agent_error"
        assert "something went wrong" in data["error"]
        assert data["duration_s"] >= 0

    def test_trace_file_recorded_in_result(self, tmp_path: Path, monkeypatch):
        provider = MockProvider(canned_responses=[
            [
                TextChunk(text="ok"),
                UsageChunk(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                FinishChunk(reason="stop"),
            ],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        monkeypatch.chdir(tmp_path)

        result_file = tmp_path / "result.json"
        trace_file = tmp_path / "trace.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "run", "hello",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
                "--result-file", str(result_file),
                "--trace", str(trace_file),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["trace_file"] == "trace.jsonl"
        assert trace_file.exists()

    def test_no_result_file_when_not_requested(self, tmp_path: Path, monkeypatch):
        provider = MockProvider(canned_responses=[
            [TextChunk(text="hi"), FinishChunk(reason="stop")],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "run", "hello",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert not (tmp_path / "result.json").exists()
