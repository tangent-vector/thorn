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

    Config = type("Config", (), {})  # type: ignore[assignment]

    def __init__(self, events: list[IncomingEvent]) -> None:
        self._events = events
        self._stop = asyncio.Event()

    @property
    def name(self) -> str:
        return "stub"

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

    Config = type("Config", (), {})  # type: ignore[assignment]

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    @property
    def name(self) -> str:
        return "slow"

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
    async def test_resolve_agent_uses_persisted_coordinator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """When a coordinator is bootstrapped, the gateway routes events to it."""
        from thorn.gateway._agents import ProjectCoordinator
        from thorn.gateway._bootstrap import bootstrap_coordinator

        monkeypatch.setenv("GITLAB_TOKEN", "fake-token-for-test")

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            clone_url="https://example.com/proj.git",
        )

        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k1"),
            content="Hello",
        )

        async with runtime:
            agent = gateway._resolve_agent(event)

        assert isinstance(agent, ProjectCoordinator)
        assert agent.id == AgentID("my-coord")

    @pytest.mark.asyncio
    async def test_resolve_agent_falls_back_to_default(self, tmp_path: Path):
        """Without a bootstrapped coordinator, falls back to a bare Agent."""
        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("k1"),
            content="Hello",
        )

        async with runtime:
            agent = gateway._resolve_agent(event)

        assert type(agent) is Agent
        assert agent.id == AgentID("default")

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
    async def test_same_session_key_accumulates_history(self, tmp_path: Path):
        """Two events with the same session_key share one session whose
        history grows across both.  The second event's prompt sees the
        full conversation from the first."""
        from thorn.core._history import TurnNode, UserPromptNode

        key = SessionKey("shared_session")
        event1 = IncomingEvent(
            source="test", session_key=key, content="First message",
        )
        event2 = IncomingEvent(
            source="test", session_key=key, content="Second message",
        )
        source = StubSource([event1, event2])
        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[source])

        await gateway.run()

        agent_id = runtime.sessions.list_agent_ids()[0]
        agent = runtime.sessions.load_agent(agent_id)
        session = runtime.sessions.load_session(agent, key)

        nodes = session._history.nodes
        assert len(nodes) == 4, (
            f"Expected 4 nodes (2 user + 2 turn), got {len(nodes)}: "
            f"{[type(n).__name__ for n in nodes]}"
        )

        assert isinstance(nodes[0], UserPromptNode)
        assert isinstance(nodes[1], TurnNode)
        assert isinstance(nodes[2], UserPromptNode)
        assert isinstance(nodes[3], TurnNode)

        assert "First message" in nodes[0].message.content
        assert "Second message" in nodes[2].message.content

    @pytest.mark.asyncio
    async def test_same_session_key_provider_sees_prior_history(
        self, tmp_path: Path,
    ):
        """On the second event with the same session_key, the provider
        receives the full prior conversation (not just the new message)."""
        key = SessionKey("shared_session")
        event1 = IncomingEvent(
            source="test", session_key=key, content="First message",
        )
        event2 = IncomingEvent(
            source="test", session_key=key, content="Second message",
        )
        source = StubSource([event1, event2])

        provider = MockProvider()
        runtime = Runtime(provider=provider, workspace_root=tmp_path)

        calls_messages: list[list[Any]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            calls_messages.append(list(messages))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]
        gateway = Gateway(runtime=runtime, sources=[source])
        await gateway.run()

        assert len(calls_messages) == 2

        first_call_msgs = calls_messages[0]
        assert len(first_call_msgs) == 1  # just the user prompt

        second_call_msgs = calls_messages[1]
        assert len(second_call_msgs) == 3  # user + assistant + user

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


    @pytest.mark.asyncio
    async def test_same_agent_events_serialized(self, tmp_path: Path):
        """Two events for the same agent should execute sequentially
        (the per-agent lock serializes them)."""
        timestamps: list[tuple[str, float]] = []
        delay = 0.1

        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])

        agent_id = AgentID("serial-agent")

        async with runtime:
            agent = runtime.get_or_create_agent(agent_id)
            runtime.save_agent(agent)

            original_prompt = _SessionPromptAccessor.__call__

            async def slow_prompt(self_accessor, text, **kwargs):
                timestamps.append(("enter", asyncio.get_event_loop().time()))
                await asyncio.sleep(delay)
                timestamps.append(("exit", asyncio.get_event_loop().time()))
                return "ok"

            event_a = IncomingEvent(
                source="test",
                session_key=SessionKey("k1"),
                content="first",
                agent_id=agent_id,
            )
            event_b = IncomingEvent(
                source="test",
                session_key=SessionKey("k2"),
                content="second",
                agent_id=agent_id,
            )

            with patch.object(
                _SessionPromptAccessor, "__call__", slow_prompt,
            ):
                await asyncio.gather(
                    gateway._handle_event(event_a),
                    gateway._handle_event(event_b),
                )

        assert len(timestamps) == 4
        first_exit = timestamps[1][1]
        second_enter = timestamps[2][1]
        assert second_enter >= first_exit, (
            "Second event started before first finished — lock not working"
        )

    @pytest.mark.asyncio
    async def test_different_agents_events_parallel(self, tmp_path: Path):
        """Two events for different agents should execute in parallel."""
        timestamps: dict[str, list[float]] = {}
        delay = 0.1

        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])

        id_a = AgentID("agent-a")
        id_b = AgentID("agent-b")

        async with runtime:
            runtime.get_or_create_agent(id_a)
            runtime.save_agent(runtime.get_or_create_agent(id_a))
            runtime.get_or_create_agent(id_b)
            runtime.save_agent(runtime.get_or_create_agent(id_b))

            async def slow_prompt(self_accessor, text, **kwargs):
                agent_id = str(self_accessor._session.agent.id)
                if agent_id not in timestamps:
                    timestamps[agent_id] = []
                timestamps[agent_id].append(asyncio.get_event_loop().time())
                await asyncio.sleep(delay)
                timestamps[agent_id].append(asyncio.get_event_loop().time())
                return "ok"

            event_a = IncomingEvent(
                source="test",
                session_key=SessionKey("k1"),
                content="from a",
                agent_id=id_a,
            )
            event_b = IncomingEvent(
                source="test",
                session_key=SessionKey("k2"),
                content="from b",
                agent_id=id_b,
            )

            with patch.object(
                _SessionPromptAccessor, "__call__", slow_prompt,
            ):
                t0 = asyncio.get_event_loop().time()
                await asyncio.gather(
                    gateway._handle_event(event_a),
                    gateway._handle_event(event_b),
                )
                total = asyncio.get_event_loop().time() - t0

        assert total < delay * 1.8, (
            f"Two different agents took {total:.2f}s — expected parallel "
            f"execution (~{delay}s, not ~{delay * 2}s)"
        )


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
    todo.project = {
        "id": project_id,
        "path_with_namespace": "org/repo",
        "http_url_to_repo": f"https://gitlab.example.com/org/repo.git",
        "default_branch": "main",
        "web_url": f"https://gitlab.example.com/org/repo",
    }
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
        assert key == SessionKey("gitlab/456/change-request/7")
        assert isinstance(key, SessionKey)

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
        assert event.session_key == SessionKey("gitlab/123/issue/42")
        assert "mentioned" in event.content
        assert "Issue #42" in event.content
        assert "Please help" in event.content
        assert "forge_mark_notification_done" in event.content
        assert event.metadata["todo_id"] == 99
        assert event.metadata["project_id"] == 123
        assert event.metadata["clone_url"] == "https://gitlab.example.com/org/repo.git"
        assert event.metadata["default_branch"] == "main"
        assert event.metadata["web_url"] == "https://gitlab.example.com/org/repo"

    def test_same_noteable_different_todo_ids_share_session_key(self):
        """Two TODOs on the same noteable (different todo.id) produce
        identical session keys -- this is how multi-turn on a single
        issue or MR works."""
        from thorn.gateway.sources._gitlab import _make_session_key

        todo_a = _make_mock_todo(todo_id=100, project_id=42, noteable_type="Issue", noteable_iid=7)
        todo_b = _make_mock_todo(todo_id=200, project_id=42, noteable_type="Issue", noteable_iid=7)
        assert _make_session_key(todo_a) == _make_session_key(todo_b)

    def test_issue_and_mr_produce_different_session_keys(self):
        """An Issue TODO and a MergeRequest TODO on the same project
        produce distinct session keys, even with the same iid."""
        from thorn.gateway.sources._gitlab import _make_session_key

        issue_todo = _make_mock_todo(
            todo_id=1, project_id=42, noteable_type="Issue", noteable_iid=7,
        )
        mr_todo = _make_mock_todo(
            todo_id=2, project_id=42, noteable_type="MergeRequest", noteable_iid=7,
        )
        assert _make_session_key(issue_todo) != _make_session_key(mr_todo)

    def test_format_event_content_includes_project_info(self):
        from thorn.gateway.sources._gitlab import _format_event_content

        todo = _make_mock_todo()
        content = _format_event_content(todo)
        assert "forge_mark_notification_done" in content
        assert "Clone URL:" in content
        assert "Default branch:" in content
        assert "Project URL:" in content


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

            await asyncio.wait_for(source.start(on_event), timeout=5.0)

            assert len(events) == 1
            assert events[0].source == "gitlab"
            assert events[0].session_key == SessionKey("gitlab/123/issue/42")

    @pytest.mark.asyncio
    async def test_emits_both_todos_with_same_session_key(self):
        """Two TODOs with different ids but the same noteable (same
        session key) should both be emitted.  _seen deduplicates by
        todo.id, not by session key."""
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

            todo_a = _make_mock_todo(todo_id=10, noteable_type="Issue", noteable_iid=7)
            todo_b = _make_mock_todo(todo_id=20, noteable_type="Issue", noteable_iid=7)
            mock_gl_instance.todos.list.return_value = [todo_a, todo_b]

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

            await source._poll_once(on_event)

            assert len(events) == 2
            assert events[0].session_key == events[1].session_key
            assert events[0].metadata["todo_id"] != events[1].metadata["todo_id"]

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

    def test_includes_project_info(self):
        from thorn.tools.gitlab import GITLAB_TOOLS

        names = [getattr(fn, "__name__", "?") for fn in GITLAB_TOOLS]
        assert "gitlab_get_project_info" in names

    def test_expected_count(self):
        from thorn.tools.gitlab import GITLAB_TOOLS

        assert len(GITLAB_TOOLS) == 9


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

    def test_serve_without_gateway_config_fails_gracefully(self, tmp_path: Path):
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, ["serve", "--workspace", str(tmp_path)])
        assert result.exit_code != 0
        assert "Gateway configuration file not found" in result.output


# ---------------------------------------------------------------------------
# ProjectCoordinator
# ---------------------------------------------------------------------------


class TestProjectCoordinator:
    def test_is_registered_subclass(self):
        from thorn.gateway._agents import ProjectCoordinator

        assert "ProjectCoordinator" in Agent._registry
        assert Agent._registry["ProjectCoordinator"] is ProjectCoordinator

    def test_has_system_prompts(self):
        from thorn.gateway._agents import ProjectCoordinator

        prompts = ProjectCoordinator._collect_system_prompts()
        assert len(prompts) >= 1
        assert "project coordinator" in prompts[0].lower()

    def test_has_forge_tools(self):
        from thorn.gateway._agents import ProjectCoordinator

        tools = ProjectCoordinator._collect_tools()
        tool_names = {getattr(t, "__name__", str(t)) for t in tools}
        assert "forge_read_issue" in tool_names
        assert "forge_post_comment" in tool_names
        assert "forge_create_change_request" in tool_names
        assert "forge_mark_notification_done" in tool_names
        assert "forge_get_project_info" in tool_names

    def test_has_git_tools(self):
        from thorn.gateway._agents import ProjectCoordinator

        tools = ProjectCoordinator._collect_tools()
        tool_names = {getattr(t, "__name__", str(t)) for t in tools}
        assert "git_clone" in tool_names
        assert "git_push" in tool_names
        assert "git_worktree_add" in tool_names
        assert "git_add" in tool_names
        assert "git_commit" in tool_names

    def test_has_file_tools(self):
        from thorn.gateway._agents import ProjectCoordinator

        tools = ProjectCoordinator._collect_tools()
        tool_names = {getattr(t, "__name__", str(t)) for t in tools}
        assert "read_file" in tool_names
        assert "edit_file" in tool_names

    def test_serialization_round_trip(self, tmp_path: Path):
        from thorn.gateway._agents import ProjectCoordinator
        from thorn.runtime._serializer import JsonSessionSerializer

        agent = ProjectCoordinator(
            id=AgentID("test-coordinator"),
            name="test",
            metadata={"project": "my-proj"},
        )
        serializer = JsonSessionSerializer()
        path = tmp_path / "agent.json"
        serializer.save_agent(agent, path)
        loaded = serializer.load_agent(path)

        assert isinstance(loaded, ProjectCoordinator)
        assert loaded.id == AgentID("test-coordinator")
        assert loaded.metadata["project"] == "my-proj"


# ---------------------------------------------------------------------------
# End-to-end wiring verification
# ---------------------------------------------------------------------------


class TestEndToEndWiring:
    """Verify that all vertical-slice pieces connect correctly."""

    def _bootstrap_runtime(self, tmp_path: Path) -> Runtime:
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="e2e-coord",
            project_name="test-proj",
            clone_url="https://gitlab.example.com/group/test-proj.git",
            default_branch="main",
            project_id=999,
        )
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_coordinator_resolved_with_correct_class(self, tmp_path: Path):
        from thorn.gateway._agents import ProjectCoordinator

        runtime = self._bootstrap_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="gitlab",
            session_key=SessionKey("gitlab_999_Issue_1"),
            content="You were mentioned",
        )
        async with runtime:
            agent = gateway._resolve_agent(event)
        assert isinstance(agent, ProjectCoordinator)

    @pytest.mark.asyncio
    async def test_coordinator_has_all_required_tools(self, tmp_path: Path):
        runtime = self._bootstrap_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="gitlab",
            session_key=SessionKey("k"),
            content="test",
        )
        async with runtime:
            agent = gateway._resolve_agent(event)
            tools = type(agent)._collect_tools()
            tool_names = {getattr(t, "__name__", str(t)) for t in tools}

        required = {
            "forge_read_issue", "forge_post_comment",
            "forge_create_change_request", "forge_mark_notification_done",
            "forge_get_project_info",
            "git_clone", "git_push", "git_add", "git_commit", "git_worktree_add",
            "read_file", "edit_file", "create_file",
        }
        assert required.issubset(tool_names), (
            f"Missing tools: {required - tool_names}"
        )

    @pytest.mark.asyncio
    async def test_coordinator_metadata_has_project_ref(self, tmp_path: Path):
        runtime = self._bootstrap_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="gitlab",
            session_key=SessionKey("k"),
            content="test",
        )
        async with runtime:
            agent = gateway._resolve_agent(event)

        assert agent.metadata["project"] == "test-proj"

    @pytest.mark.asyncio
    async def test_coordinator_workspace_has_memory(self, tmp_path: Path):
        runtime = self._bootstrap_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="gitlab",
            session_key=SessionKey("k"),
            content="test",
        )
        async with runtime:
            agent = gateway._resolve_agent(event)

        memory_path = agent.workspace / "MEMORY.md"
        assert memory_path.is_file()
        content = memory_path.read_text(encoding="utf-8")
        assert "test-proj" in content
        assert "https://gitlab.example.com/group/test-proj.git" in content

    @pytest.mark.asyncio
    async def test_coordinator_system_prompts_rendered(self, tmp_path: Path):
        runtime = self._bootstrap_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        event = IncomingEvent(
            source="gitlab",
            session_key=SessionKey("k"),
            content="test",
        )
        async with runtime:
            agent = gateway._resolve_agent(event)
            prompts = agent._render_system_prompts()

        assert any("project coordinator" in p.lower() for p in prompts)
        assert any("forge_create_change_request" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_event_content_includes_project_metadata(self):
        from thorn.gateway.sources._gitlab import _format_event_content

        todo = _make_mock_todo(
            project_id=999,
            noteable_type="Issue",
            noteable_iid=5,
        )
        content = _format_event_content(todo)

        assert "Clone URL:" in content
        assert "Default branch:" in content
        assert "forge_mark_notification_done" in content


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# expand_env_vars
# ---------------------------------------------------------------------------


class TestExpandEnvVars:
    def test_expands_string_reference(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway._config import expand_env_vars

        monkeypatch.setenv("MY_VAR", "hello")
        assert expand_env_vars("$MY_VAR") == "hello"

    def test_leaves_plain_string_unchanged(self):
        from thorn.gateway._config import expand_env_vars

        assert expand_env_vars("no-dollar") == "no-dollar"

    def test_raises_on_missing_env_var(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway._config import expand_env_vars

        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        with pytest.raises(ValueError, match="NONEXISTENT_VAR_12345"):
            expand_env_vars("$NONEXISTENT_VAR_12345")

    def test_recurses_into_dict(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway._config import expand_env_vars

        monkeypatch.setenv("URL", "https://example.com")
        monkeypatch.setenv("TOK", "secret")
        result = expand_env_vars({"url": "$URL", "token": "$TOK", "count": 5})
        assert result == {"url": "https://example.com", "token": "secret", "count": 5}

    def test_recurses_into_list(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway._config import expand_env_vars

        monkeypatch.setenv("A", "aaa")
        result = expand_env_vars(["$A", "plain", 42])
        assert result == ["aaa", "plain", 42]

    def test_nested_dict_in_list(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway._config import expand_env_vars

        monkeypatch.setenv("X", "expanded")
        result = expand_env_vars([{"key": "$X"}])
        assert result == [{"key": "expanded"}]

    def test_passthrough_non_string_types(self):
        from thorn.gateway._config import expand_env_vars

        assert expand_env_vars(42) == 42
        assert expand_env_vars(3.14) == 3.14
        assert expand_env_vars(True) is True
        assert expand_env_vars(None) is None


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class TestSourceRegistry:
    def test_gitlab_registered(self):
        from thorn.gateway.sources import get_registered_source
        from thorn.gateway.sources._gitlab import GitLabTODOsSource

        assert get_registered_source("gitlab") is GitLabTODOsSource

    def test_unknown_type_raises_key_error(self):
        from thorn.gateway.sources import get_registered_source

        with pytest.raises(KeyError, match="nonexistent"):
            get_registered_source("nonexistent")

    def test_register_and_retrieve(self):
        from thorn.gateway.sources import (
            _SOURCE_REGISTRY,
            register_source,
            get_registered_source,
        )
        from pydantic import BaseModel

        class DummyConfig(BaseModel):
            value: str = "test"

        class DummySource(EventSource):
            Config = DummyConfig

            def __init__(self, config: DummyConfig) -> None:
                self._config = config

            @property
            def name(self) -> str:
                return "dummy"

            async def start(self, on_event):
                pass

            async def stop(self):
                pass

        register_source("dummy_test", DummySource)
        try:
            assert get_registered_source("dummy_test") is DummySource
        finally:
            _SOURCE_REGISTRY.pop("dummy_test", None)

    def test_duplicate_registration_raises(self):
        from thorn.gateway.sources import register_source

        with pytest.raises(ValueError, match="already registered"):
            register_source("gitlab", EventSource)  # type: ignore[arg-type]

    def test_gitlab_source_has_config_attribute(self):
        from thorn.gateway.sources._gitlab import GitLabSourceConfig, GitLabTODOsSource

        assert GitLabTODOsSource.Config is GitLabSourceConfig


# ---------------------------------------------------------------------------
# GatewayConfig loading
# ---------------------------------------------------------------------------


class TestGatewayConfigLoading:
    def test_load_valid_config(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        config_data = {
            "services": [
                {
                    "name": "my-gitlab",
                    "type": "gitlab",
                    "config": {
                        "url": "https://gitlab.example.com",
                        "token": "test-token",
                    },
                }
            ]
        }
        (thorn_dir / "gateway.json").write_text(
            json.dumps(config_data), encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert len(config.services) == 1
        assert config.services[0].name == "my-gitlab"
        assert config.services[0].type == "gitlab"
        assert config.services[0].config["url"] == "https://gitlab.example.com"

    def test_load_missing_file_raises(self, tmp_path: Path):
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="gateway.json"):
            load_gateway_config(thorn_dir)

    def test_load_empty_services(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"services": []}), encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.services == []

    def test_load_defaults_to_empty_services(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({}), encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.services == []


class TestInstantiateSources:
    def test_instantiates_gitlab_event_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib"),
        ):
            from thorn.gateway._config import (
                GatewayConfig,
                ServiceSpec,
                instantiate_sources,
            )
            from thorn.gateway.sources._gitlab import GitLabTODOsSource

            monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
            monkeypatch.setenv("GITLAB_TOKEN", "glpat-test")

            config = GatewayConfig(services=[
                ServiceSpec(
                    name="gl-poller",
                    type="gitlab-events",
                    config={"url": "$GITLAB_URL", "token": "$GITLAB_TOKEN"},
                ),
            ])
            sources = instantiate_sources(config)
            assert len(sources) == 1
            assert isinstance(sources[0], GitLabTODOsSource)

    def test_forge_type_not_returned_by_instantiate_sources(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Forge services (type='gitlab') are not event sources and
        should not appear in the list returned by instantiate_sources."""
        monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
        monkeypatch.setenv("GITLAB_TOKEN", "glpat-test")

        from thorn.gateway._config import (
            GatewayConfig,
            ServiceSpec,
            instantiate_sources,
        )

        config = GatewayConfig(services=[
            ServiceSpec(
                name="gl",
                type="gitlab",
                config={"url": "$GITLAB_URL", "token": "$GITLAB_TOKEN"},
            ),
        ])
        sources = instantiate_sources(config)
        assert sources == []

    def test_unknown_type_raises(self):
        from thorn.gateway._config import (
            GatewayConfig,
            ServiceSpec,
            instantiate_sources,
        )

        config = GatewayConfig(services=[
            ServiceSpec(name="bad", type="nonexistent", config={}),
        ])
        with pytest.raises(KeyError, match="nonexistent"):
            instantiate_sources(config)

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch):
        from thorn.gateway._config import (
            GatewayConfig,
            ServiceSpec,
            instantiate_sources,
        )

        monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)

        config = GatewayConfig(services=[
            ServiceSpec(
                name="test",
                type="gitlab",
                config={"url": "$MISSING_VAR_XYZ", "token": "literal"},
            ),
        ])
        with pytest.raises(ValueError, match="MISSING_VAR_XYZ"):
            instantiate_sources(config)

    def test_empty_config_returns_empty_list(self):
        from thorn.gateway._config import GatewayConfig, instantiate_sources

        config = GatewayConfig(services=[])
        assert instantiate_sources(config) == []


class TestInstantiateServices:
    def test_instantiates_forge_service(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        from thorn.gateway._config import (
            GatewayConfig,
            ServiceSpec,
            instantiate_services,
        )
        from thorn.tools.forge import GitLabForgeService

        monkeypatch.setenv("GL_URL", "https://gitlab.example.com")
        monkeypatch.setenv("GL_TOKEN", "glpat-test")

        config = GatewayConfig(services=[
            ServiceSpec(
                name="my-gl",
                type="gitlab",
                config={"url": "$GL_URL", "token": "$GL_TOKEN"},
            ),
        ])
        services = instantiate_services(config)
        assert len(services) == 1
        assert isinstance(services[0], GitLabForgeService)
        assert services[0].name == "my-gl"

    def test_instantiates_project_service(self):
        from thorn.gateway._config import (
            GatewayConfig,
            ServiceSpec,
            instantiate_services,
        )
        from thorn.tools.forge import ProjectService

        config = GatewayConfig(services=[
            ServiceSpec(
                name="my-proj",
                type="project",
                config={
                    "forge": "my-gl",
                    "native_id": "42",
                    "path": "org/repo",
                    "clone_url": "https://example.com/repo.git",
                    "default_branch": "main",
                },
            ),
        ])
        services = instantiate_services(config)
        assert len(services) == 1
        assert isinstance(services[0], ProjectService)
        assert services[0].name == "my-proj"

    def test_instantiates_mixed_services(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib"),
        ):
            from thorn.gateway._config import (
                GatewayConfig,
                ServiceSpec,
                instantiate_services,
            )
            from thorn.gateway.sources._gitlab import GitLabTODOsSource
            from thorn.tools.forge import GitLabForgeService, ProjectService

            monkeypatch.setenv("GL_URL", "https://gl.example.com")
            monkeypatch.setenv("GL_TOKEN", "secret")

            config = GatewayConfig(services=[
                ServiceSpec(
                    name="gl",
                    type="gitlab",
                    config={"url": "$GL_URL", "token": "$GL_TOKEN"},
                ),
                ServiceSpec(
                    name="proj",
                    type="project",
                    config={"forge": "gl", "native_id": "1"},
                ),
                ServiceSpec(
                    name="gl-poller",
                    type="gitlab-events",
                    config={"url": "$GL_URL", "token": "$GL_TOKEN"},
                ),
            ])
            services = instantiate_services(config)
            assert len(services) == 3
            assert isinstance(services[0], GitLabForgeService)
            assert isinstance(services[1], ProjectService)
            assert isinstance(services[2], GitLabTODOsSource)


# ---------------------------------------------------------------------------
# New-format config: instantiate_new_format + infer_event_sources
# ---------------------------------------------------------------------------


class TestInstantiateNewFormat:
    def test_creates_forge_and_project_services(self):
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            ProjectSpec,
            instantiate_new_format,
        )
        from thorn.tools.forge import GitLabForgeService, ProjectService

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", base_url="https://gl.example.com")],
            projects=[ProjectSpec(
                name="my-proj", forge="gl", native_id="42",
                clone_url="https://gl.example.com/org/repo.git",
            )],
        )
        services = instantiate_new_format(config)
        assert len(services) == 2
        assert isinstance(services[0], GitLabForgeService)
        assert services[0].name == "gl"
        assert isinstance(services[1], ProjectService)
        assert services[1].name == "my-proj"

    def test_github_forge_service(self):
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            instantiate_new_format,
        )
        from thorn.tools.forge import GitHubForgeService

        config = GatewayConfig(
            forges=[ForgeSpec(name="gh", type="github", base_url="https://api.github.com")],
        )
        services = instantiate_new_format(config)
        assert len(services) == 1
        assert isinstance(services[0], GitHubForgeService)

    def test_unknown_forge_type_raises(self):
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            instantiate_new_format,
        )

        config = GatewayConfig(
            forges=[ForgeSpec(name="x", type="unknown")],
        )
        with pytest.raises(ValueError, match="Unknown forge type"):
            instantiate_new_format(config)

    def test_is_new_format_property(self):
        from thorn.gateway._config import ForgeSpec, GatewayConfig

        new = GatewayConfig(forges=[ForgeSpec(name="gl", type="gitlab")])
        assert new.is_new_format is True

        legacy = GatewayConfig(services=[])
        assert legacy.is_new_format is False


class TestInferEventSources:
    def test_infers_gitlab_source_from_agent_account(self):
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib"),
        ):
            from thorn.core._account import (
                AgentAccountsConfig,
                ForgeAccountConfig,
                GitLabCredentials,
            )
            from thorn.core._agent import Agent
            from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources
            from thorn.gateway.sources._gitlab import GitLabTODOsSource

            config = GatewayConfig(
                forges=[ForgeSpec(name="gl", type="gitlab", base_url="https://gl.example.com")],
            )
            agent = Agent(
                name="bot",
                accounts=AgentAccountsConfig(forge_accounts=[
                    ForgeAccountConfig(
                        forge="gl",
                        credentials=GitLabCredentials(token="tok"),
                    ),
                ]),
            )
            sources = infer_event_sources(config, [agent])
            assert len(sources) == 1
            assert isinstance(sources[0], GitLabTODOsSource)
            assert "bot-gl-events" in sources[0].name

    def test_infers_github_source_from_agent_account(self):
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github"),
            patch("thorn.tools.github.build_pygithub_auth", return_value=None),
        ):
            from thorn.core._account import AgentAccountsConfig, ForgeAccountConfig
            from thorn.core._agent import Agent
            from thorn.gateway._config import (
                ForgeSpec,
                GatewayConfig,
                ProjectSpec,
                infer_event_sources,
            )
            from thorn.gateway.sources._github import GitHubNotificationsSource
            from thorn.tools._github_connection import GitHubPatAuth

            config = GatewayConfig(
                forges=[ForgeSpec(name="gh", type="github", base_url="https://api.github.com")],
                projects=[ProjectSpec(name="repo", forge="gh", native_id="owner/repo")],
            )
            agent = Agent(
                name="bot",
                accounts=AgentAccountsConfig(forge_accounts=[
                    ForgeAccountConfig(
                        forge="gh",
                        credentials=GitHubPatAuth(token="ghp-tok"),
                    ),
                ]),
            )
            sources = infer_event_sources(config, [agent])
            assert len(sources) == 1
            assert isinstance(sources[0], GitHubNotificationsSource)

    def test_no_sources_when_agent_has_no_accounts(self):
        from thorn.core._agent import Agent
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", base_url="https://gl.example.com")],
        )
        agent = Agent(name="bot")
        sources = infer_event_sources(config, [agent])
        assert sources == []

    def test_no_sources_when_no_agents(self):
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", base_url="https://gl.example.com")],
        )
        sources = infer_event_sources(config, [])
        assert sources == []

    def test_skips_unknown_forge_in_agent_account(self):
        from thorn.core._account import (
            AgentAccountsConfig,
            ForgeAccountConfig,
            GitLabCredentials,
        )
        from thorn.core._agent import Agent
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", base_url="https://gl.example.com")],
        )
        agent = Agent(
            name="bot",
            accounts=AgentAccountsConfig(forge_accounts=[
                ForgeAccountConfig(
                    forge="nonexistent",
                    credentials=GitLabCredentials(token="tok"),
                ),
            ]),
        )
        sources = infer_event_sources(config, [agent])
        assert sources == []

    def test_github_skipped_when_no_project_repos(self):
        """GitHub source is not created when no projects are on that forge."""
        from thorn.core._account import AgentAccountsConfig, ForgeAccountConfig
        from thorn.core._agent import Agent
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources
        from thorn.tools._github_connection import GitHubPatAuth

        config = GatewayConfig(
            forges=[ForgeSpec(name="gh", type="github", base_url="https://api.github.com")],
        )
        agent = Agent(
            name="bot",
            accounts=AgentAccountsConfig(forge_accounts=[
                ForgeAccountConfig(
                    forge="gh",
                    credentials=GitHubPatAuth(token="ghp-tok"),
                ),
            ]),
        )
        sources = infer_event_sources(config, [agent])
        assert sources == []


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapCoordinator:
    def test_creates_identity_memory_and_gateway_config(self, tmp_path: Path):
        from thorn.gateway._bootstrap import bootstrap_coordinator

        aid = bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="test-coord",
            project_name="my-project",
            clone_url="https://gitlab.example.com/group/my-project.git",
            default_branch="develop",
            native_project_id="42",
        )

        assert str(aid) == "test-coord"

        identity = tmp_path / ".thorn" / "agents" / "test-coord.json"
        assert identity.is_file()

        import json
        data = json.loads(identity.read_text(encoding="utf-8"))
        assert data["agent_class"] == "ProjectCoordinator"
        assert data["metadata"]["project"] == "my-project"

        acct = data["accounts"]["forge_accounts"][0]
        assert acct["git_user_name"] == "test-coord"
        assert acct["git_user_email"] == "test-coord@thorn"
        assert acct["credentials"]["kind"] == "gitlab-pat"
        assert acct["credentials"]["token"] == "$GITLAB_TOKEN"

        memory = tmp_path / ".thorn" / "agents" / "test-coord" / "MEMORY.md"
        assert memory.is_file()
        content = memory.read_text(encoding="utf-8")
        assert "my-project" in content
        assert "develop" in content

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        assert gateway_config.is_file()
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))

        assert len(gw_data["forges"]) == 1
        assert len(gw_data["projects"]) == 1
        assert "services" not in gw_data

        forge = gw_data["forges"][0]
        assert forge["type"] == "gitlab"

        proj = gw_data["projects"][0]
        assert proj["name"] == "my-project"
        assert proj["native_id"] == "42"
        assert proj["clone_url"] == "https://gitlab.example.com/group/my-project.git"
        assert proj["default_branch"] == "develop"

    def test_bootstrap_appends_to_existing_gateway_config(self, tmp_path: Path):
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="first-coord",
            project_name="proj-a",
            clone_url="https://example.com/a.git",
        )
        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="second-coord",
            project_name="proj-b",
            clone_url="https://example.com/b.git",
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        proj_names = [p["name"] for p in gw_data["projects"]]
        assert "proj-a" in proj_names
        assert "proj-b" in proj_names

    def test_bootstrap_updates_existing_entry_by_name(self, tmp_path: Path):
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            clone_url="https://example.com/proj.git",
        )
        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            clone_url="https://example.com/proj-v2.git",
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        assert len(gw_data["forges"]) == 1
        assert len(gw_data["projects"]) == 1
        assert gw_data["projects"][0]["clone_url"] == "https://example.com/proj-v2.git"

    def test_bootstrap_custom_token_env(self, tmp_path: Path):
        """Custom access_token_env is written into agent credentials."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="custom",
            project_name="proj",
            clone_url="https://example.com/proj.git",
            access_token_env="MY_TOKEN",
        )

        identity = tmp_path / ".thorn" / "agents" / "custom.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["credentials"]["token"] == "$MY_TOKEN"

    def test_loads_via_session_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from thorn.gateway._agents import ProjectCoordinator
        from thorn.gateway._bootstrap import bootstrap_coordinator
        from thorn.runtime._store import SessionStore

        monkeypatch.setenv("GITLAB_TOKEN", "fake-token")

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="loadable",
            project_name="proj",
            clone_url="https://example.com/proj.git",
        )

        store = SessionStore(tmp_path / ".thorn" / "agents")
        agent = store.load_agent(AgentID("loadable"))

        assert isinstance(agent, ProjectCoordinator)
        assert agent.metadata["project"] == "proj"
        assert hasattr(agent, "accounts")
        assert len(agent.accounts.forge_accounts) == 1

    def test_cli_bootstrap_command(self, tmp_path: Path):
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "cli-test",
            "--project-name", "test-proj",
            "--clone-url", "https://example.com/test-proj.git",
            "--workspace", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "cli-test" in result.output
        assert (tmp_path / ".thorn" / "agents" / "cli-test.json").is_file()
        assert (tmp_path / ".thorn" / "gateway.json").is_file()
        assert "gateway.json" in result.output

    def test_legacy_project_id_parameter(self, tmp_path: Path):
        """The legacy ``project_id: int`` parameter is accepted and
        mapped to ``native_project_id``."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="legacy",
            project_name="proj",
            clone_url="https://example.com/proj.git",
            project_id=999,
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        proj = gw_data["projects"][0]
        assert proj["native_id"] == "999"

    def test_github_bootstrap_pat_default(self, tmp_path: Path):
        """Bootstrap with forge_type='github' defaults to PAT auth."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="gh-coord",
            project_name="my-repo",
            clone_url="https://github.com/owner/repo.git",
            native_project_id="owner/repo",
            forge_type="github",
            forge_base_url="https://api.github.com",
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))

        forge = gw_data["forges"][0]
        assert forge["type"] == "github"
        assert forge["base_url"] == "https://api.github.com"

        proj = gw_data["projects"][0]
        assert proj["native_id"] == "owner/repo"
        assert proj["forge"] == "my-repo-forge"

        identity = tmp_path / ".thorn" / "agents" / "gh-coord.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        assert data["metadata"]["project"] == "my-repo"
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["git_user_name"] == "gh-coord"
        assert acct["git_user_email"] == "gh-coord@thorn"
        assert acct["credentials"]["kind"] == "pat"
        assert acct["credentials"]["token"] == "$GITHUB_TOKEN"

    def test_github_bootstrap_app_mode(self, tmp_path: Path):
        """Bootstrap with github_auth_mode='app' produces App auth blocks."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="gh-app-coord",
            project_name="my-repo",
            clone_url="https://github.com/owner/repo.git",
            native_project_id="owner/repo",
            forge_type="github",
            github_auth_mode="app",
        )

        identity = tmp_path / ".thorn" / "agents" / "gh-app-coord.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["credentials"]["kind"] == "app"
        assert acct["credentials"]["app_id"] == "$GITHUB_APP_ID"

    def test_github_bootstrap_without_custom_url(self, tmp_path: Path):
        """When no base_url is given, the forge entry omits base_url."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="gh-coord",
            project_name="my-repo",
            clone_url="https://github.com/owner/repo.git",
            native_project_id="owner/repo",
            forge_type="github",
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        forge = gw_data["forges"][0]
        assert "base_url" not in forge or forge.get("base_url") == ""

    def test_bootstrap_writes_git_identity(self, tmp_path: Path):
        """Bootstrap writes git identity into agent accounts."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="id-test",
            project_name="proj",
            clone_url="https://example.com/proj.git",
        )

        identity = tmp_path / ".thorn" / "agents" / "id-test.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["git_user_name"] == "id-test"
        assert acct["git_user_email"] == "id-test@thorn"

    def test_bootstrap_custom_git_identity(self, tmp_path: Path):
        """Explicit git_user_name/email override the defaults."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="custom-id",
            project_name="proj",
            clone_url="https://example.com/proj.git",
            git_user_name="My Bot",
            git_user_email="bot@example.com",
        )

        identity = tmp_path / ".thorn" / "agents" / "custom-id.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["git_user_name"] == "My Bot"
        assert acct["git_user_email"] == "bot@example.com"

    def test_cli_bootstrap_github_pat_default(self, tmp_path: Path):
        """CLI GitHub bootstrap defaults to PAT auth."""
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "gh-cli-test",
            "--project-name", "test-repo",
            "--clone-url", "https://github.com/owner/repo.git",
            "--forge-type", "github",
            "--native-project-id", "owner/repo",
            "--workspace", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output
        assert "gh-cli-test" in result.output
        assert "GITHUB_TOKEN" in result.output

        import json
        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        forge = gw_data["forges"][0]
        assert forge["type"] == "github"

    def test_cli_bootstrap_github_app_mode(self, tmp_path: Path):
        """CLI --github-auth-mode app produces App auth blocks."""
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "gh-app-cli",
            "--project-name", "test-repo",
            "--clone-url", "https://github.com/owner/repo.git",
            "--forge-type", "github",
            "--native-project-id", "owner/repo",
            "--github-auth-mode", "app",
            "--workspace", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output
        assert "GITHUB_APP_ID" in result.output

        import json
        identity = tmp_path / ".thorn" / "agents" / "gh-app-cli.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["credentials"]["kind"] == "app"

    def test_cli_bootstrap_defaults_to_gitlab(self, tmp_path: Path):
        """Without --forge-type, the CLI defaults to gitlab."""
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "gl-test",
            "--project-name", "proj",
            "--clone-url", "https://example.com/proj.git",
            "--workspace", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output

        import json
        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        forge = gw_data["forges"][0]
        assert forge["type"] == "gitlab"

        identity = tmp_path / ".thorn" / "agents" / "gl-test.json"
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"]["forge_accounts"][0]
        assert acct["credentials"]["kind"] == "gitlab-pat"
        assert acct["credentials"]["token"] == "$GITLAB_TOKEN"


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


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


class TestRouteGithubEvent:
    def test_non_noteable_event_key_format(self):
        from thorn.gateway._routing import route_github_event

        key = route_github_event(
            repo_id=42, event_type="PushEvent", event_id="abc123",
        )
        assert key == SessionKey("github/42/pushevent/abc123")

    def test_non_noteable_event_type_lowercased(self):
        from thorn.gateway._routing import route_github_event

        key = route_github_event(
            repo_id=1, event_type="My Event", event_id="e1",
        )
        assert key == SessionKey("github/1/my_event/e1")

    def test_noteable_issue_key_format(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_github_event

        key = route_github_event(
            repo_id=42,
            noteable=Noteable(NoteableKind.ISSUE, 7),
            event_type="IssuesEvent",
            event_id="e1",
        )
        assert key == SessionKey("github/42/issue/7")

    def test_noteable_change_request_key_format(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_github_event

        key = route_github_event(
            repo_id=42,
            noteable=Noteable(NoteableKind.CHANGE_REQUEST, 3),
            event_type="PullRequestEvent",
            event_id="e1",
        )
        assert key == SessionKey("github/42/change-request/3")

    def test_different_non_noteable_events_get_distinct_keys(self):
        from thorn.gateway._routing import route_github_event

        k1 = route_github_event(
            repo_id=99, event_type="Push", event_id="e1",
        )
        k2 = route_github_event(
            repo_id=99, event_type="Issue", event_id="e2",
        )
        assert k1 != k2

    def test_returns_session_key_type(self):
        from thorn.gateway._routing import route_github_event

        key = route_github_event(
            repo_id=42, event_type="PushEvent", event_id="abc",
        )
        assert isinstance(key, SessionKey)


class TestRouteGitlabTodo:
    def test_issue_key_format(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        key = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.ISSUE, 5),
        )
        assert key == SessionKey("gitlab/10/issue/5")

    def test_change_request_key_format(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        key = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.CHANGE_REQUEST, 2),
        )
        assert key == SessionKey("gitlab/10/change-request/2")

    def test_different_noteables_get_distinct_keys(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        k1 = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.ISSUE, 1),
        )
        k2 = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.CHANGE_REQUEST, 2),
        )
        assert k1 != k2

    def test_returns_session_key_type(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        key = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.ISSUE, 5),
        )
        assert isinstance(key, SessionKey)


# ---------------------------------------------------------------------------
# Gateway: centralized workspace derivation
# ---------------------------------------------------------------------------


class TestGatewayWorkspaceRouting:
    """Verify that _handle_event derives the session workspace from
    ``<runtime_ws>/<agent_id>/<session_key>/`` and pre-creates the
    directory on disk."""

    def _make_runtime(self, tmp_path: Path) -> Runtime:
        return Runtime(provider=MockProvider(), workspace_root=tmp_path)

    @pytest.mark.asyncio
    async def test_session_workspace_derived_and_created(self, tmp_path: Path):
        """The gateway derives the session workspace mechanically and
        the directory exists on disk after _handle_event."""
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("github/42/issue/7"),
            content="Hello",
        )
        source = StubSource([event])
        runtime = self._make_runtime(tmp_path)

        with patch.object(
            _SessionPromptAccessor, "__call__", return_value="ok",
        ):
            gateway = Gateway(runtime=runtime, sources=[source])
            await gateway.run()

        expected_ws = tmp_path / "default" / "github" / "42" / "issue" / "7"
        assert expected_ws.is_dir()

        agent_ids = runtime.sessions.list_agent_ids()
        agent = runtime.sessions.load_agent(agent_ids[0])
        loaded = runtime.sessions.load_session(
            agent, SessionKey("github/42/issue/7"),
        )
        assert loaded.workspace_root == expected_ws

    @pytest.mark.asyncio
    async def test_workspace_uses_agent_id_in_path(self, tmp_path: Path):
        """When a specific agent handles the event, the workspace path
        includes that agent's ID."""
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            runtime_root=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            clone_url="https://example.com/proj.git",
        )
        runtime = self._make_runtime(tmp_path)
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("gitlab/10/issue/5"),
            content="Hello",
        )
        source = StubSource([event])

        with patch.object(
            _SessionPromptAccessor, "__call__", return_value="ok",
        ):
            gateway = Gateway(runtime=runtime, sources=[source])
            await gateway.run()

        expected_ws = (
            tmp_path / "my-coord" / "gitlab" / "10" / "issue" / "5"
        )
        assert expected_ws.is_dir()

    @pytest.mark.asyncio
    async def test_existing_session_workspace_not_overwritten(
        self, tmp_path: Path,
    ):
        """A second event with the same session key does not change
        the persisted workspace (workspace_root is only applied at
        session creation time)."""
        runtime = self._make_runtime(tmp_path)
        key = SessionKey("github/99/issue/1")

        event1 = IncomingEvent(
            source="test", session_key=key, content="First",
        )
        event2 = IncomingEvent(
            source="test", session_key=key, content="Second",
        )
        source = StubSource([event1, event2])

        with patch.object(
            _SessionPromptAccessor, "__call__", return_value="ok",
        ):
            gateway = Gateway(runtime=runtime, sources=[source])
            await gateway.run()

        agent_ids = runtime.sessions.list_agent_ids()
        agent = runtime.sessions.load_agent(agent_ids[0])
        loaded = runtime.sessions.load_session(agent, key)

        expected_ws = tmp_path / "default" / "github" / "99" / "issue" / "1"
        assert loaded.workspace_root == expected_ws

    @pytest.mark.asyncio
    async def test_flat_session_key_workspace(self, tmp_path: Path):
        """A simple (non-slashed) session key produces a single-level
        directory under the agent ID."""
        event = IncomingEvent(
            source="test",
            session_key=SessionKey("simple_key"),
            content="Hello",
        )
        source = StubSource([event])
        runtime = self._make_runtime(tmp_path)

        with patch.object(
            _SessionPromptAccessor, "__call__", return_value="ok",
        ):
            gateway = Gateway(runtime=runtime, sources=[source])
            await gateway.run()

        expected_ws = tmp_path / "default" / "simple_key"
        assert expected_ws.is_dir()

        agent_ids = runtime.sessions.list_agent_ids()
        agent = runtime.sessions.load_agent(agent_ids[0])
        loaded = runtime.sessions.load_session(
            agent, SessionKey("simple_key"),
        )
        assert loaded.workspace_root == expected_ws
