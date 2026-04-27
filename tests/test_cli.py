"""Tests for the thorn CLI — specifically ``thorn run --result-file``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from thorn._cli import main as cli_main
from thorn.core._provider import (
    FinishChunk,
    MockProvider,
    TextChunk,
    ToolCallChunk,
    UsageChunk,
)


@pytest.fixture(autouse=True)
def _isolate_cli_agency_home(monkeypatch, tmp_path):
    """Redirect the CLI's default ``~/.thorn`` agency home into ``tmp_path``.

    Phase 5 of the CLI/gateway unification switched the CLI's default
    agency home to ``~/.thorn``.  Without this fixture every test that
    invokes ``thorn run`` or ``thorn chat`` without an explicit
    ``--agency`` would create agent directories under the developer's
    real home directory, polluting it and breaking test isolation.

    We patch :func:`pathlib.Path.home` to point at ``tmp_path``; the
    CLI's agency-home resolver calls ``Path.home() / '.thorn'``, so
    the effective default agency becomes ``tmp_path/.thorn`` for the
    lifetime of each test.  Tests that want to inspect the resulting
    agency tree continue to assert on ``tmp_path / '.thorn' / ...``
    just like they did under the pre-Phase-5 ``for_cli`` layout.

    Autouse so new tests added here inherit the isolation by default;
    a test that explicitly wants the real home dir can override the
    fixture in its own scope.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))


def _mock_provider_factory(provider: MockProvider):
    """Return a callable that ignores arguments and returns *provider*."""
    def factory() -> MockProvider:
        return provider
    return factory


def _find_session_state_dirs(
    sessions_root: Path,
    *,
    key_prefix: str | None = None,
) -> list[Path]:
    """Return every session ``_state`` directory under *sessions_root*.

    Hierarchical session keys produce nested directories under
    ``sessions/`` (e.g. ``cli/<workspace-basename>/<8 hex>``), with
    each session's framework files living one level deeper, in a
    ``_state`` sentinel subdirectory (see
    ``AgencyPaths.session_metadata_dir``).  This helper walks that
    layout, optionally filtering by the leading components of the
    decoded session key (e.g. ``key_prefix='cli'``).
    """
    from thorn.runtime._paths import (
        SESSION_STATE_DIR,
        session_key_from_path,
    )
    if not sessions_root.is_dir():
        return []
    matches: list[Path] = []
    for state_dir in sessions_root.rglob(SESSION_STATE_DIR):
        if not state_dir.is_dir() or state_dir.name != SESSION_STATE_DIR:
            continue
        rel = state_dir.parent.relative_to(sessions_root)
        if not rel.parts or SESSION_STATE_DIR in rel.parts:
            continue
        if key_prefix is not None:
            key = session_key_from_path(rel)
            head = key.components[: len(key_prefix.split("/"))]
            if "/".join(head) != key_prefix:
                continue
        matches.append(state_dir)
    return matches


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
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert not (tmp_path / "result.json").exists()


class TestAgencyHomeResolution:
    """``thorn run`` / ``thorn chat`` honour ``--agency`` and the ``~/.thorn`` default.

    Phase 5 of the CLI/gateway unification switched the CLI's default
    agency home from ``{ws}/.thorn/`` (old ``for_cli`` nested layout)
    to ``~/.thorn/`` (local-agency convention).  These tests lock
    down both paths: an explicit ``--agency`` override wins over the
    default, and the default actually lands at ``Path.home() /
    '.thorn'`` (which the autouse fixture redirects to ``tmp_path``
    so we are really asserting "the resolver consulted
    ``Path.home()``").
    """

    def test_explicit_agency_override_beats_default(
        self, tmp_path: Path, monkeypatch,
    ):
        """``--agency`` routes agent state to a caller-chosen directory."""
        provider = MockProvider(canned_responses=[
            [TextChunk(text="ok"), FinishChunk(reason="stop")],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        monkeypatch.chdir(tmp_path)

        agency_dir = tmp_path / "custom-agency"
        workspace_dir = tmp_path / "elsewhere"
        workspace_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "run", "hello",
                "--workspace", str(workspace_dir),
                "--agency", str(agency_dir),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # The agency home should hold the agent tree; the workspace
        # must stay untouched (no stray .thorn/ inside it).
        sessions_root = agency_dir / "agents" / "local" / "sessions"
        assert sessions_root.is_dir(), (
            "explicit --agency directory should hold the agent/session tree"
        )
        cli_dirs = _find_session_state_dirs(sessions_root, key_prefix="cli")
        assert len(cli_dirs) == 1, (
            "one ephemeral cli/... session dir should live under --agency"
        )
        assert not (workspace_dir / ".thorn").exists(), (
            "Phase 5 decouples workspace from agency home; the "
            "workspace must not sprout a nested .thorn/ when --agency "
            "is set explicitly"
        )

    def test_default_agency_is_home_dot_thorn(
        self, tmp_path: Path, monkeypatch,
    ):
        """With no ``--agency``, state lands under ``Path.home() / '.thorn'``.

        The autouse ``_isolate_cli_agency_home`` fixture redirects
        ``Path.home()`` to ``tmp_path``, so the effective default is
        ``tmp_path/.thorn``.  Asserting on that directory exercises
        the full resolution path (resolver consulted ``Path.home()``,
        appended ``.thorn``, and the runtime used the result).
        """
        provider = MockProvider(canned_responses=[
            [TextChunk(text="ok"), FinishChunk(reason="stop")],
        ])
        monkeypatch.setattr(
            "thorn._cli.load_provider_from_env",
            _mock_provider_factory(provider),
        )
        workspace_dir = tmp_path / "some-workspace"
        workspace_dir.mkdir()
        monkeypatch.chdir(workspace_dir)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "run", "hello",
                "--workspace", str(workspace_dir),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        default_home = tmp_path / ".thorn"
        assert default_home.is_dir(), (
            "the default agency home should be auto-created at "
            "~/.thorn on first use"
        )
        sessions_root = default_home / "agents" / "local" / "sessions"
        assert sessions_root.is_dir(), (
            "agent/session tree should live under the default ~/.thorn"
        )
        cli_dirs = _find_session_state_dirs(sessions_root, key_prefix="cli")
        assert len(cli_dirs) == 1

        # And we should *not* have written a nested .thorn inside the
        # workspace -- the Phase 5 clean-break behavior.
        assert not (workspace_dir / ".thorn").exists(), (
            "Phase 5 clean break: the default CLI path no longer "
            "drops .thorn/ inside the workspace root"
        )


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
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # Per-invocation CLI session keys are
        # ``cli/<workspace-basename>/<8 hex chars>``; framework files
        # land at ``<home>/agents/local/sessions/<key-as-path>/_state/``
        # with ``inbox/`` next to ``session.json`` inside the
        # ``_state/`` sentinel.
        sessions_root = tmp_path / ".thorn" / "agents" / "local" / "sessions"
        assert sessions_root.is_dir(), (
            "the runtime should have created an agent sessions root"
        )
        cli_state_dirs = _find_session_state_dirs(sessions_root, key_prefix="cli")
        assert len(cli_state_dirs) == 1, (
            f"expected exactly one ephemeral cli/... session, "
            f"found state dirs: {[str(d) for d in cli_state_dirs]}"
        )
        inbox_dir = cli_state_dirs[0] / "inbox"
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
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # The agent round must have produced at least scope_enter and
        # scope_exit events tagged with the ephemeral CLI session key.
        scope_events = [
            (ev, key) for ev, key in captured_events
            if ev in {"scope_enter", "scope_exit"}
        ]
        assert scope_events, (
            f"expected scope events on the bus subscriber; got "
            f"{captured_events}"
        )
        # Every captured scope event should be tagged with a
        # ``cli/...`` ephemeral key (Phase 5 naming convention).
        for ev, key in scope_events:
            assert key is not None and key.startswith("cli/"), (
                f"event {ev!r} carried session_key={key!r}; expected "
                f"a 'cli/...' tag"
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
            # invocation: the ``cli/...`` keys are random-hex-suffixed,
            # "other-key" cannot collide.
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
    """``thorn chat`` REPL drives turns through the in-process scheduler.

    Phase 4 of the CLI/gateway unification routes each user input
    through ``ChatPromptRouter`` + ``AgentScheduler`` (rather than
    calling ``session.prompt`` directly).  These tests exercise the
    REPL by feeding stdin via ``CliRunner``; because the mock provider
    yields canned chunks synchronously and the REPL exits cleanly on
    EOF, no real concurrency is in play, but the scheduler's drain
    loop and per-turn save callback are exercised end-to-end.
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
                "--workspace", str(tmp_path),
            ],
            input="hi there\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "hello back" in result.output

        # Phase 5: chat keys are ``cli/<basename>/<uuid8>``; look up
        # whatever single session was created in this invocation.
        sessions_root = tmp_path / ".thorn" / "agents" / "local" / "sessions"
        cli_state_dirs = _find_session_state_dirs(sessions_root, key_prefix="cli")
        assert len(cli_state_dirs) == 1, (
            f"expected exactly one chat session, got "
            f"{[str(d) for d in cli_state_dirs]}"
        )
        # History file contents are an implementation detail of the
        # serializer; just confirm something was written into the
        # session's framework dir.
        assert any(cli_state_dirs[0].iterdir()), (
            "session _state directory should not be empty after a turn"
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

    def test_each_invocation_creates_fresh_session(
        self, tmp_path: Path, monkeypatch,
    ):
        """Two ``thorn chat`` invocations must land in two separate session dirs.

        Phase 5 replaces the static ``default`` session key with a
        per-invocation ``cli/<basename>/<uuid8>`` key.  Running chat
        twice in the same workspace must therefore produce two
        distinct on-disk session directories, neither of which shows
        the "Resuming session" banner (there is nothing to resume --
        the session is brand-new).  This test guards against a
        regression in which the key-generation helper is bypassed and
        a stable ``default`` key sneaks back in.
        """
        def _one_turn(text: str) -> str:
            provider = MockProvider(canned_responses=[
                [TextChunk(text=text), FinishChunk(reason="stop")],
            ])
            monkeypatch.setattr(
                "thorn._cli.load_provider_from_env",
                _mock_provider_factory(provider),
            )
            runner = CliRunner()
            result = runner.invoke(
                cli_main,
                [
                    "chat",
                    "--workspace", str(tmp_path),
                ],
                input="hi\n",
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert "Resuming session" not in result.output, (
                "Phase 5 dropped implicit per-workspace resume; the "
                "REPL must not claim to be resuming anything"
            )
            return result.output

        monkeypatch.chdir(tmp_path)

        _one_turn("answer alpha")
        _one_turn("answer beta")

        sessions_root = tmp_path / ".thorn" / "agents" / "local" / "sessions"
        cli_state_dirs = _find_session_state_dirs(sessions_root, key_prefix="cli")
        assert len(cli_state_dirs) == 2, (
            f"expected two distinct ephemeral chat sessions on disk, "
            f"got {[str(d) for d in cli_state_dirs]}"
        )

    def test_session_inbox_dir_created_and_drained(
        self, tmp_path: Path, monkeypatch,
    ):
        """Phase 4 wiring: chat posts to a session inbox the scheduler drains.

        The notification per turn must be removed by the router before
        ``router.turn`` returns, so after the REPL exits the inbox
        directory exists (created by ``SessionInbox.__init__``) but
        contains no leftover notification files.  Catches a regression
        in which the chat REPL silently bypasses the scheduler/router
        path and goes back to calling ``session.prompt`` directly.
        """
        provider = MockProvider(canned_responses=[
            [TextChunk(text="answer"), FinishChunk(reason="stop")],
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
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            input="please answer\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # Phase 5: chat keys are ``cli/<basename>/<uuid8>``; find the
        # one session created this invocation.
        sessions_root = tmp_path / ".thorn" / "agents" / "local" / "sessions"
        cli_state_dirs = _find_session_state_dirs(sessions_root, key_prefix="cli")
        assert len(cli_state_dirs) == 1, (
            f"expected exactly one chat session, got "
            f"{[str(d) for d in cli_state_dirs]}"
        )
        inbox_dir = cli_state_dirs[0] / "inbox"
        assert inbox_dir.is_dir(), (
            "the SessionInbox constructor should have mkdir'd "
            "the per-session inbox dir"
        )
        leftover_notifications = [
            p for p in inbox_dir.iterdir()
            if p.is_file() and p.suffix == ".json"
        ]
        assert leftover_notifications == [], (
            "ChatPromptRouter must remove notifications after each "
            "turn; otherwise the scheduler's progress guarantee "
            "would eventually evict them"
        )

    def test_shutdown_housekeeping_runs_by_default(
        self, tmp_path: Path, monkeypatch,
    ):
        """Clean REPL exit triggers one more ``session.prompt`` for housekeeping.

        The provider is pre-loaded with *two* canned responses but only
        one user turn is typed.  With shutdown housekeeping enabled
        (the default), the second canned response is consumed by the
        framework-driven housekeeping turn that runs after ``_chat_loop``
        returns.  Guarding on ``canned_responses == []`` therefore
        proves the housekeeping code path actually reaches the
        provider; counting any other way (history length, session
        files) would be fooled by the no-op echo fallback.
        """
        provider = MockProvider(canned_responses=[
            [TextChunk(text="answer"), FinishChunk(reason="stop")],
            [TextChunk(text="noted"), FinishChunk(reason="stop")],
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
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            input="hi there\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert provider.canned_responses == [], (
            "shutdown housekeeping should have consumed the second "
            "canned response on clean REPL exit"
        )

    def test_no_housekeeping_flag_skips_shutdown_turn(
        self, tmp_path: Path, monkeypatch,
    ):
        """``--no-housekeeping`` must suppress the extra shutdown prompt.

        Complement to ``test_shutdown_housekeeping_runs_by_default``:
        with the opt-out flag set, the second canned response is left
        untouched, which is the only observable difference from the
        default behaviour under the mock provider.  Tests that pass
        the flag implicitly (e.g. to avoid provider-queue
        bookkeeping) rely on this semantic.
        """
        provider = MockProvider(canned_responses=[
            [TextChunk(text="answer"), FinishChunk(reason="stop")],
            [TextChunk(text="should-not-be-used"), FinishChunk(reason="stop")],
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
                "--workspace", str(tmp_path),
                "--no-housekeeping",
                "--quiet",
            ],
            input="hi there\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(provider.canned_responses) == 1, (
            "with --no-housekeeping, the second canned response must "
            "be left untouched -- there is no shutdown turn to "
            "consume it"
        )

    def test_shutdown_housekeeping_skipped_on_no_input(
        self, tmp_path: Path, monkeypatch,
    ):
        """EOF without a single turn must not provoke a housekeeping call.

        Running housekeeping on a brand-new, never-prompted session
        would be pure overhead (nothing in history to journal) and
        would burn a provider round in the common case of typing
        ``thorn chat`` and immediately realising you meant something
        else.  Empty-history fast-exit path.
        """
        provider = MockProvider(canned_responses=[
            [TextChunk(text="should-not-be-used"), FinishChunk(reason="stop")],
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
                "--workspace", str(tmp_path),
                "--quiet",
            ],
            input="",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(provider.canned_responses) == 1, (
            "no user turn was typed, so nothing should have gone to "
            "the provider -- including the housekeeping prompt"
        )

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
                "--workspace", str(tmp_path),
            ],
            input="please fail\nplease succeed\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Agent error" in result.output
        assert "bad" in result.output
        assert "recovered" in result.output
