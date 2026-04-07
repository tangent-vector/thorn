"""Tests for thorn.gateway -- EventSource, Gateway, GitLabTODOsSource, and CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.core._agent import Agent
from thorn.core._provider import MockProvider
from thorn.core._session import Session, _SessionPromptAccessor
from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._gateway import Gateway
from thorn.runtime import AgentID, Runtime, SessionKey


# ---------------------------------------------------------------------------
# IncomingEvent
# ---------------------------------------------------------------------------


class TestIncomingEvent:
    def test_fields(self):
        event = IncomingEvent(
            source="gitlab",
            session_key=SessionKey("gitlab_123_Issue_42"),
            content="You were mentioned.",
            metadata={"todo_id": 99},
        )
        assert event.source == "gitlab"
        assert event.session_key == SessionKey("gitlab_123_Issue_42")
        assert event.content == "You were mentioned."
        assert event.metadata == {"todo_id": 99}

    def test_frozen(self):
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k"),
            content="hi",
        )
        with pytest.raises(AttributeError):
            event.source = "other"  # type: ignore[misc]

    def test_default_metadata(self):
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k"),
            content="hi",
        )
        assert event.metadata == {}

    def test_agent_id_defaults_to_none(self):
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k"),
            content="hi",
        )
        assert event.agent_id is None

    def test_agent_id_set(self):
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k"),
            content="hi",
            agent_id=AgentID("my-agent"),
        )
        assert event.agent_id == AgentID("my-agent")


# ---------------------------------------------------------------------------
# Helpers: stub EventSource
# ---------------------------------------------------------------------------


class StubSource(EventSource):
    """EventSource that emits a fixed list of events, then stops."""

    def __init__(self, events: list[IncomingEvent]) -> None:
        self._events = events
        self._stop = asyncio.Event()

    async def start(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        for event in self._events:
            if self._stop.is_set():
                break
            await on_event(event)
        self._stop.set()

    async def stop(self) -> None:
        self._stop.set()


class SlowSource(EventSource):
    """EventSource that waits until stopped."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def start(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        await self._stop.wait()

    async def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class TestGateway:
    def _make_runtime(self, tmp_path: Path) -> Runtime:
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_handles_events(self, tmp_path: Path):
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("test_1"),
            content="Hello agent",
        )
        source = StubSource([event])
        runtime = self._make_runtime(tmp_path)

        handled: list[IncomingEvent] = []
        original_handle = Gateway._handle_event

        async def tracking_handle(self_gw: Gateway, evt: IncomingEvent) -> None:
            handled.append(evt)

        with patch.object(Gateway, "_handle_event", tracking_handle):
            gateway = Gateway(runtime=runtime, sources=[source])
            await gateway.run()

        assert len(handled) == 1
        assert handled[0].session_key == SessionKey("test_1")

    @pytest.mark.asyncio
    async def test_creates_agent_and_session_for_event(self, tmp_path: Path):
        runtime = self._make_runtime(tmp_path)
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("agent_key"),
            content="Do something",
        )

        gateway = Gateway(runtime=runtime, sources=[], tools=[])

        async with runtime:
            agent = runtime.get_or_create_agent(AgentID("default"))
            assert agent.id == AgentID("default")
            assert isinstance(agent, Agent)

            session = runtime.get_or_create_session(agent, event.session_key)
            assert session.key == SessionKey("agent_key")
            assert session.agent is agent
            assert isinstance(session, Session)

    @pytest.mark.asyncio
    async def test_shutdown_stops_sources(self, tmp_path: Path):
        source = SlowSource()
        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[source])

        async def run_and_stop():
            task = asyncio.create_task(gateway.run())
            await asyncio.sleep(0.05)
            await gateway.shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run_and_stop()
        assert source._stop.is_set()

    @pytest.mark.asyncio
    async def test_error_in_session_prompt_does_not_crash(self, tmp_path: Path):
        """If session.prompt() fails, the gateway logs and continues."""
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("err_key"),
            content="Trigger error",
        )
        source = StubSource([event])
        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[source], tools=[])

        with patch.object(
            _SessionPromptAccessor, "__call__",
            side_effect=RuntimeError("Agent failed"),
        ):
            await gateway.run()

        assert not runtime.sessions.session_exists(AgentID("default"), "err_key")

    @pytest.mark.asyncio
    async def test_handle_event_persists_agent(self, tmp_path: Path):
        """_handle_event saves the agent so it survives gateway restarts."""
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k1"),
            content="Hello",
        )
        source = StubSource([event])
        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[source], tools=[])

        with patch.object(
            _SessionPromptAccessor, "__call__",
            return_value="ok",
        ):
            await gateway.run()

        assert runtime.sessions.agent_exists(AgentID("default"))

    @pytest.mark.asyncio
    async def test_multiple_sources(self, tmp_path: Path):
        event1 = IncomingEvent(
            source="a", session_key=SessionKey("k1"), content="from a",
        )
        event2 = IncomingEvent(
            source="b", session_key=SessionKey("k2"), content="from b",
        )
        source_a = StubSource([event1])
        source_b = StubSource([event2])
        runtime = self._make_runtime(tmp_path)

        handled: list[SessionKey] = []

        async def tracking_handle(self_gw: Gateway, evt: IncomingEvent) -> None:
            handled.append(evt.session_key)

        with patch.object(Gateway, "_handle_event", tracking_handle):
            gateway = Gateway(runtime=runtime, sources=[source_a, source_b])
            await gateway.run()

        assert SessionKey("k1") in handled
        assert SessionKey("k2") in handled


# ---------------------------------------------------------------------------
# GitLabTODOsSource (mocked)
# ---------------------------------------------------------------------------


def _make_mock_todo(
    todo_id: int = 1,
    project_id: int = 123,
    noteable_type: str = "Issue",
    noteable_iid: int = 42,
    action_name: str = "mentioned",
    body: str = "Hey @thorn-bot, look at this!",
) -> MagicMock:
    todo = MagicMock()
    todo.id = todo_id
    todo.project = {"id": project_id, "path_with_namespace": "org/repo"}
    todo.target_type = noteable_type
    todo.target = {"iid": noteable_iid}
    todo.action_name = action_name
    todo.body = body
    return todo


class TestGitLabTODOsSourceEventFormatting:
    def test_make_session_key(self):
        from thorn.gateway.sources._gitlab import _make_session_key

        todo = _make_mock_todo(project_id=456, noteable_type="MergeRequest", noteable_iid=7)
        key = _make_session_key(todo)
        assert key == SessionKey("gitlab_456_MergeRequest_7")
        assert isinstance(key, SessionKey)

    def test_session_key_is_filesystem_safe(self):
        from thorn.gateway.sources._gitlab import _make_session_key

        todo = _make_mock_todo()
        key = _make_session_key(todo)
        forbidden = set('/:*?"<>|\\')
        assert not any(c in forbidden for c in str(key))

    def test_make_event(self):
        from thorn.gateway.sources._gitlab import _make_event

        todo = _make_mock_todo(
            todo_id=99,
            project_id=123,
            noteable_type="Issue",
            noteable_iid=42,
            action_name="mentioned",
            body="Please help",
        )
        event = _make_event(todo)

        assert event.source == "gitlab"
        assert event.session_key == SessionKey("gitlab_123_Issue_42")
        assert "mentioned" in event.content
        assert "Issue #42" in event.content
        assert "Please help" in event.content
        assert "gitlab_mark_todo_done" in event.content
        assert event.metadata["todo_id"] == 99
        assert event.metadata["project_id"] == 123

    def test_format_event_content_includes_tools(self):
        from thorn.gateway.sources._gitlab import _format_event_content

        todo = _make_mock_todo()
        content = _format_event_content(todo)
        assert "post_comment" in content
        assert "gitlab_mark_todo_done" in content
        assert "read_issue" in content


class TestGitLabTODOsSourcePolling:
    @pytest.mark.asyncio
    async def test_polls_and_emits_events(self):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib") as mock_gl_mod,
        ):
            mock_gl_instance = MagicMock()
            mock_gl_mod.Gitlab.return_value = mock_gl_instance

            mock_user = MagicMock()
            mock_user.id = 1
            mock_user.username = "thorn-bot"
            mock_user.name = "Thorn Bot"
            mock_user.web_url = "https://gitlab.example.com/thorn-bot"
            mock_gl_instance.user = mock_user

            todo = _make_mock_todo()
            mock_gl_instance.todos.list.return_value = [todo]

            from thorn.gateway.sources._gitlab import (
                GitLabSourceConfig,
                GitLabTODOsSource,
            )

            config = GitLabSourceConfig(
                url="https://gitlab.example.com",
                token="test-token",
                poll_interval=5,
            )
            source = GitLabTODOsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)
                await source.stop()

            with patch.object(source, "_configure_tools_client"):
                await asyncio.wait_for(source.start(on_event), timeout=5.0)

            assert len(events) == 1
            assert events[0].source == "gitlab"
            assert events[0].session_key == SessionKey("gitlab_123_Issue_42")

    @pytest.mark.asyncio
    async def test_deduplicates_todos(self):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib") as mock_gl_mod,
        ):
            mock_gl_instance = MagicMock()
            mock_gl_mod.Gitlab.return_value = mock_gl_instance

            mock_user = MagicMock()
            mock_user.id = 1
            mock_user.username = "thorn-bot"
            mock_user.name = "Thorn Bot"
            mock_user.web_url = "https://gitlab.example.com/thorn-bot"
            mock_gl_instance.user = mock_user

            todo = _make_mock_todo(todo_id=42)

            poll_count = 0

            def fake_list(**kwargs: Any) -> list[MagicMock]:
                nonlocal poll_count
                poll_count += 1
                return [todo]

            mock_gl_instance.todos.list.side_effect = fake_list

            from thorn.gateway.sources._gitlab import (
                GitLabSourceConfig,
                GitLabTODOsSource,
            )

            config = GitLabSourceConfig(
                url="https://gitlab.example.com",
                token="test-token",
                poll_interval=5,
            )
            source = GitLabTODOsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)

            with patch.object(source, "_configure_tools_client"):
                await source._poll_once(on_event)
                await source._poll_once(on_event)

            assert len(events) == 1


# ---------------------------------------------------------------------------
# GitLabSourceConfig
# ---------------------------------------------------------------------------


class TestGitLabSourceConfig:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib"),
        ):
            from thorn.gateway.sources._gitlab import GitLabSourceConfig

            monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
            monkeypatch.setenv("GITLAB_TOKEN", "glpat-secret")
            monkeypatch.setenv("THORN_GITLAB_USERNAME", "my-bot")
            monkeypatch.setenv("THORN_POLL_INTERVAL", "15")

            config = GitLabSourceConfig.from_env()
            assert config.url == "https://gitlab.example.com"
            assert config.token == "glpat-secret"
            assert config.username == "my-bot"
            assert config.poll_interval == 15

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib"),
        ):
            from thorn.gateway.sources._gitlab import GitLabSourceConfig

            monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
            monkeypatch.setenv("GITLAB_TOKEN", "glpat-secret")
            monkeypatch.delenv("THORN_GITLAB_USERNAME", raising=False)
            monkeypatch.delenv("THORN_POLL_INTERVAL", raising=False)

            config = GitLabSourceConfig.from_env()
            assert config.username == "thorn-bot"
            assert config.poll_interval == 30

    def test_from_env_missing_raises(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway.sources._gitlab import GitLabSourceConfig

        monkeypatch.delenv("GITLAB_URL", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        with pytest.raises(ValueError, match="GITLAB_URL"):
            GitLabSourceConfig.from_env()


# ---------------------------------------------------------------------------
# gitlab_mark_todo_done tool
# ---------------------------------------------------------------------------


class TestGitLabMarkTodoDoneTool:
    async def test_calls_client_method(self):
        from thorn.tools.gitlab import gitlab_mark_todo_done, set_client

        mock_client = MagicMock()
        set_client(mock_client)
        try:
            result = await gitlab_mark_todo_done(99)
            mock_client.mark_todo_done.assert_called_once_with(99)
            assert "99" in result
            assert "done" in result.lower()
        finally:
            set_client(None)


class TestGitLabClientMarkTodoDone:
    def test_calls_http_post(self):
        with (
            patch("thorn.tools.gitlab._gitlab_lib") as mock_gl_mod,
            patch("thorn.tools.gitlab._HAS_GITLAB", True),
        ):
            from thorn.tools.gitlab import GitLabClient, GitLabConfig

            mock_gl_instance = MagicMock()
            mock_gl_mod.Gitlab.return_value = mock_gl_instance

            config = GitLabConfig(url="https://gitlab.example.com", token="test")
            client = GitLabClient(config)
            client.mark_todo_done(42)

            mock_gl_instance.http_post.assert_called_once_with(
                "/todos/42/mark_as_done"
            )


class TestGitLabToolsList:
    def test_includes_mark_todo_done(self):
        from thorn.tools.gitlab import GITLAB_TOOLS

        names = [getattr(fn, "__name__", "?") for fn in GITLAB_TOOLS]
        assert "gitlab_mark_todo_done" in names

    def test_expected_count(self):
        from thorn.tools.gitlab import GITLAB_TOOLS

        assert len(GITLAB_TOOLS) == 6


# ---------------------------------------------------------------------------
# SessionStore filesystem safety
# ---------------------------------------------------------------------------


class TestSessionStoreSafeDirnames:
    def test_safe_dirname_passthrough(self):
        from thorn.runtime._store import _safe_dirname

        assert _safe_dirname("gitlab_123_Issue_42") == "gitlab_123_Issue_42"

    def test_safe_dirname_encodes_colon(self):
        from thorn.runtime._store import _safe_dirname

        encoded = _safe_dirname("gitlab:org/repo:Issue:42")
        assert ":" not in encoded
        assert "/" not in encoded

    def test_unsafe_dirname_roundtrip(self):
        from thorn.runtime._store import _safe_dirname, _unsafe_dirname

        original = "gitlab:org/repo:Issue:42"
        encoded = _safe_dirname(original)
        recovered = _unsafe_dirname(encoded)
        assert recovered == original

    def test_store_save_load_agent_with_special_chars(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(tmp_path)
        agent = Agent(id=AgentID("a:b/c"), name="special")
        store.save_agent(agent)

        assert store.agent_exists(AgentID("a:b/c"))
        loaded = store.load_agent(AgentID("a:b/c"))
        assert loaded.name == "special"

    def test_store_list_agent_ids_decodes(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(tmp_path)
        store.save_agent(Agent(id=AgentID("x:y"), name="xy"))
        store.save_agent(Agent(id=AgentID("simple"), name="simple"))

        ids = store.list_agent_ids()
        assert AgentID("simple") in ids
        assert AgentID("x:y") in ids

    def test_store_delete_agent_with_special_chars(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(tmp_path)
        store.save_agent(Agent(id=AgentID("del:me/now"), name="del"))
        assert store.agent_exists(AgentID("del:me/now"))
        store.delete_agent(AgentID("del:me/now"))
        assert not store.agent_exists(AgentID("del:me/now"))

    def test_clean_agent_ids_unchanged(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(tmp_path)
        store.save_agent(Agent(id=AgentID("gitlab_123_Issue_42"), name="test"))

        file_names = [f.stem for f in tmp_path.iterdir() if f.is_file()]
        assert "gitlab_123_Issue_42" in file_names

    def test_store_session_with_special_chars(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(tmp_path)
        agent = Agent(id=AgentID("test_agent"), name="test")
        store.save_agent(agent)

        session = Session(agent=agent, key=SessionKey("a:b/c"))
        store.save_session(session)

        assert store.session_exists(AgentID("test_agent"), SessionKey("a:b/c"))
        loaded = store.load_session(agent, SessionKey("a:b/c"))
        assert loaded.key == SessionKey("a:b/c")


# ---------------------------------------------------------------------------
# CLI serve group
# ---------------------------------------------------------------------------


class TestServeCliGroup:
    def test_serve_is_group(self):
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output.lower()

    def test_serve_mcp_help(self):
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, ["serve", "mcp", "--help"])
        assert result.exit_code == 0
        assert "transport" in result.output.lower()

    def test_serve_without_gitlab_config_fails_gracefully(self, monkeypatch: pytest.MonkeyPatch):
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        monkeypatch.delenv("GITLAB_URL", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        runner = CliRunner()
        result = runner.invoke(cli_main, ["serve"])
        assert result.exit_code != 0
        assert "GITLAB_URL" in result.output or "GITLAB_TOKEN" in result.output


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestGatewayReExports:
    def test_gateway_importable(self):
        from thorn.gateway import EventSource, Gateway, IncomingEvent

        assert Gateway is not None
        assert EventSource is not None
        assert IncomingEvent is not None

    def test_sources_importable(self):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib"),
        ):
            from thorn.gateway.sources import GitLabSourceConfig, GitLabTODOsSource

            assert GitLabSourceConfig is not None
            assert GitLabTODOsSource is not None
