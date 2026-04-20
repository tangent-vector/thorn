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
        # The trace listener is a subscriber on the runtime's
        # EventBus; if the bus is wired up correctly, the agent
        # round's events should have reached it.  This guards against
        # a regression where ``_build_runtime`` builds a bus with no
        # subscriptions or where the bus drops events silently.
        trace_lines = [
            ln for ln in trace_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert trace_lines, (
            "expected the trace listener subscribed on the EventBus to "
            "receive at least one event during the agent round"
        )
        events = [json.loads(ln)["event"] for ln in trace_lines]
        # The agent loop always pushes a scope and emits text chunks,
        # so the trace must contain at least these.  Asserting a
        # superset of expected events keeps the test robust to
        # additions of new event types.
        assert "scope_enter" in events
        assert "scope_exit" in events

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


class TestRunPipelineStructure:
    """``thorn run`` should flow through the in-process scheduler+inbox.

    Phase 2 of the CLI/gateway unification routes ``thorn run`` through
    an :class:`AgentScheduler` driven by ``make_cli_prompt_dispatcher``,
    posting the user's prompt as a notification on a per-invocation
    session inbox.  These tests verify the structural side effects of
    that pipeline (inbox directory created, then drained) rather than
    just the user-facing result, so a future regression that bypasses
    the scheduler -- e.g. by calling ``run_agent_loop`` directly again
    -- would be caught.
    """

    def test_session_inbox_dir_created_and_drained(
        self, tmp_path: Path, monkeypatch,
    ):
        provider = MockProvider(canned_responses=[
            [TextChunk(text="ok"), FinishChunk(reason="stop")],
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

        # Per-invocation session keys are ``run-<8 hex chars>``;
        # inbox dirs land at <home>/agents/local/sessions/<key>/inbox/.
        sessions_root = tmp_path / ".thorn" / "agents" / "local" / "sessions"
        assert sessions_root.is_dir(), (
            "the runtime should have created an agent sessions root"
        )
        run_session_dirs = [
            d for d in sessions_root.iterdir()
            if d.is_dir() and d.name.startswith("run-")
        ]
        assert len(run_session_dirs) == 1, (
            f"expected exactly one ephemeral run-* session dir, "
            f"found {sorted(d.name for d in run_session_dirs)}"
        )
        inbox_dir = run_session_dirs[0] / "inbox"
        assert inbox_dir.is_dir(), (
            "the SessionInbox constructor should have mkdir'd the inbox dir"
        )
        # The dispatcher deletes the notification it processes, so no
        # ``*.json`` notification files should remain at the inbox root.
        leftover_notifications = [
            p for p in inbox_dir.iterdir()
            if p.is_file() and p.suffix == ".json"
        ]
        assert leftover_notifications == [], (
            "the CLI dispatcher must remove the notification it "
            "processed; otherwise the scheduler would loop on it"
        )


class TestRunEventBusWiring:
    """``thorn run`` should fan events through the runtime's :class:`EventBus`.

    Phase 3 of the CLI/gateway unification moves the runtime from
    "one runtime, one sink" to "one runtime, one bus, many filtered
    listeners".  These tests verify the wiring: a listener subscribed
    to the runtime's bus *before* the agent round runs receives the
    expected events, and a listener whose filter excludes the run
    session sees nothing.
    """

    def test_subscribed_listener_receives_session_events(
        self, tmp_path: Path, monkeypatch,
    ):
        from thorn.core._event_bus import EventBus
        from thorn._cli import _build_runtime

        # Wrap _build_runtime so the test can capture the runtime that
        # ``thorn run`` constructed and subscribe a listener to its
        # bus before the agent round starts.  We can't subscribe
        # after ``runner.invoke`` returns -- the runtime is torn
        # down by then.  A pre-subscription via ``_build_runtime``
        # patching is the lightest-touch hook.
        captured_events: list[tuple[str, str | None]] = []

        class _CapturingSink:
            async def on_response_chunk(self, chunk, scope=None):
                key = scope.metadata.get("session_key") if scope else None
                # Walk outer chain too (chunks come from inner scopes).
                cur = scope
                while cur is not None and key is None:
                    key = cur.metadata.get("session_key")
                    cur = cur.outer
                captured_events.append(("chunk", key))

            async def on_status(self, message, scope=None):
                key = None
                cur = scope
                while cur is not None and key is None:
                    key = cur.metadata.get("session_key")
                    cur = cur.outer
                captured_events.append(("status", key))

            async def on_scope_enter(self, scope):
                key = None
                cur = scope
                while cur is not None and key is None:
                    key = cur.metadata.get("session_key")
                    cur = cur.outer
                captured_events.append(("scope_enter", key))

            async def on_scope_exit(self, scope, *, duration_s=None):
                key = None
                cur = scope
                while cur is not None and key is None:
                    key = cur.metadata.get("session_key")
                    cur = cur.outer
                captured_events.append(("scope_exit", key))

            async def on_tool_start(self, name, arguments, *, scope=None): ...
            async def on_tool_end(self, name, **kwargs): ...
            async def on_completion_end(self, **kwargs): ...
            async def on_advisory(self, source, content, *, scope=None): ...

        original_build = _build_runtime

        def _patched_build(*args, **kwargs):
            rt = original_build(*args, **kwargs)
            # Subscribe an *unfiltered* observer so we can inspect
            # what session_key tag the events carry; the assertions
            # below check that tag, not the filter behaviour (which
            # has its own dedicated unit tests).
            assert isinstance(rt.event_sink, EventBus), (
                "Phase 3 requires the CLI runtime's event sink to be "
                "an EventBus"
            )
            rt.event_sink.subscribe(_CapturingSink())
            return rt

        monkeypatch.setattr("thorn._cli._build_runtime", _patched_build)

        provider = MockProvider(canned_responses=[
            [TextChunk(text="ok"), FinishChunk(reason="stop")],
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

        # The agent round must have produced at least scope_enter and
        # scope_exit events tagged with the run-XXX session key.
        scope_events = [
            (ev, key) for ev, key in captured_events
            if ev in {"scope_enter", "scope_exit"}
        ]
        assert scope_events, (
            f"expected scope events on the bus subscriber; got "
            f"{captured_events}"
        )
        # Every captured scope event should be tagged with the
        # ephemeral run session key.
        for ev, key in scope_events:
            assert key is not None and key.startswith("run-"), (
                f"event {ev!r} carried session_key={key!r}; expected "
                f"a 'run-*' tag"
            )

    def test_listener_filtered_to_other_session_sees_nothing(
        self, tmp_path: Path, monkeypatch,
    ):
        from thorn.core._event_bus import EventBus, in_session
        from thorn._cli import _build_runtime

        captured: list[str] = []

        class _SilentExceptStatus:
            async def on_response_chunk(self, chunk, scope=None): ...
            async def on_status(self, message, scope=None):
                captured.append(message)
            async def on_scope_enter(self, scope): ...
            async def on_scope_exit(self, scope, *, duration_s=None): ...
            async def on_tool_start(self, name, arguments, *, scope=None): ...
            async def on_tool_end(self, name, **kwargs): ...
            async def on_completion_end(self, **kwargs): ...
            async def on_advisory(self, source, content, *, scope=None): ...

        original_build = _build_runtime

        def _patched_build(*args, **kwargs):
            rt = original_build(*args, **kwargs)
            assert isinstance(rt.event_sink, EventBus)
            # Filter is for a session that will never exist in this
            # invocation: the run-* keys are random hex, "other-key"
            # cannot collide.
            rt.event_sink.subscribe(
                _SilentExceptStatus(), scope_filter=in_session("other-key"),
            )
            return rt

        monkeypatch.setattr("thorn._cli._build_runtime", _patched_build)

        provider = MockProvider(canned_responses=[
            [TextChunk(text="ok"), FinishChunk(reason="stop")],
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
        assert captured == [], (
            f"a listener filtered to a non-matching session must not "
            f"receive any events; got {captured}"
        )


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
