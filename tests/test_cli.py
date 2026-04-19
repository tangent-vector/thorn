"""Tests for the thorn CLI — specifically ``thorn run --result-file``."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from thorn._cli import main as cli_main
from thorn.core._provider import (
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


class TestChat:
    """``thorn chat`` REPL drives turns through ``session.prompt``.

    These tests exercise the REPL by feeding stdin via ``CliRunner``;
    because the mock provider yields canned chunks synchronously and the
    REPL exits cleanly on EOF, no real concurrency is in play.  The
    chat command's blocking ``console.input`` call inside an ``async``
    function is fine in this single-user model but is a known watch-item
    once Phase 4 puts a scheduler under the REPL.
    """

    def test_eof_exits_cleanly_with_no_input(
        self, tmp_path: Path, monkeypatch,
    ):
        """Closing stdin before typing anything must produce a clean exit."""
        provider = MockProvider(canned_responses=[])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "chat",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            input="",
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_single_turn_runs_and_persists_history(
        self, tmp_path: Path, monkeypatch,
    ):
        """One user line drives one ``session.prompt`` round and persists."""
        provider = MockProvider(canned_responses=[
            [
                TextChunk(text="hello back"),
                UsageChunk(
                    prompt_tokens=3, completion_tokens=2, total_tokens=5,
                ),
                FinishChunk(reason="stop"),
            ],
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
                "chat",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
            ],
            input="hi there\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "hello back" in result.output

        sessions_dir = (
            tmp_path / ".thorn" / "agents" / "local" / "sessions" / "default"
        )
        assert sessions_dir.is_dir(), (
            "session directory should have been created"
        )
        # History file contents are an implementation detail of the
        # serializer; just confirm something was written.
        assert any(sessions_dir.iterdir()), (
            "session directory should not be empty after a turn"
        )

    def test_multiple_turns_share_session_history(
        self, tmp_path: Path, monkeypatch,
    ):
        """Two turns in one chat invocation should both reach the provider."""
        provider = MockProvider(canned_responses=[
            [
                TextChunk(text="answer one"),
                FinishChunk(reason="stop"),
            ],
            [
                TextChunk(text="answer two"),
                FinishChunk(reason="stop"),
            ],
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
                "chat",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
            ],
            input="first question\nsecond question\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "answer one" in result.output
        assert "answer two" in result.output
        assert provider.canned_responses == [], (
            "both canned responses should have been consumed"
        )

    def test_resuming_session_shows_history_entry_count(
        self, tmp_path: Path, monkeypatch,
    ):
        """Re-invoking ``thorn chat`` should report the resumed history."""
        first_provider = MockProvider(canned_responses=[
            [
                TextChunk(text="one"),
                FinishChunk(reason="stop"),
            ],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(first_provider),
        )
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "chat",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            input="hello\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        second_provider = MockProvider(canned_responses=[
            [
                TextChunk(text="two"),
                FinishChunk(reason="stop"),
            ],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(second_provider),
        )
        result = runner.invoke(
            cli_main,
            [
                "chat",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
            ],
            input="hello again\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Resuming session" in result.output

    def test_skill_error_does_not_terminate_repl(
        self, tmp_path: Path, monkeypatch,
    ):
        """A failed first turn (raise_error) leaves the REPL alive for a second."""
        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(
                    call_id="c1",
                    name="raise_error",
                    arguments='{"message": "bad"}',
                ),
                FinishChunk(reason="tool_calls"),
            ],
            [
                TextChunk(text="recovered"),
                FinishChunk(reason="stop"),
            ],
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
                "chat",
                "--no-tools", "--no-discover", "--no-mcp",
                "--workspace", str(tmp_path),
            ],
            input="please fail\nplease succeed\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Agent error" in result.output
        assert "bad" in result.output
        assert "recovered" in result.output
