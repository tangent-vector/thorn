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
from thorn.gateway._routing import NoteableKind
from thorn.runtime import AgentID, Runtime, SessionKey
from thorn.runtime._paths import AgencyPaths


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
        """If session.prompt() fails, the gateway logs and continues.

        Under the inbox/scheduler model, the session and its inbox are
        created proactively when the event arrives, so the session is
        persisted even though the prompt dispatcher fails.  The point of
        the test is that no exception escapes from ``gateway.run()``.
        """
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

        assert runtime.sessions.session_exists(AgentID("default"), "err_key")

    @pytest.mark.asyncio
    async def test_resolve_agent_uses_persisted_coordinator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """When a coordinator is bootstrapped, the gateway routes events to it."""
        from thorn.gateway._agents import ProjectCoordinator
        from thorn.gateway._bootstrap import bootstrap_coordinator

        monkeypatch.setenv("GITLAB_TOKEN", "fake-token-for-test")

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            project_url="https://gitlab.com/group/proj",
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
    async def test_same_agent_different_sessions_parallel(self, tmp_path: Path):
        """Two events for different sessions of the same agent should
        execute concurrently (governed by the per-agent scheduler's
        concurrency cap, default 3).  Only events on the *same session*
        are serialized, which happens naturally because each session has
        a single driver.
        """
        timestamps: list[tuple[str, float]] = []
        delay = 0.1

        agent_id = AgentID("multi-session-agent")

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
        source = StubSource([event_a, event_b])

        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[source], tools=[])

        from thorn.runtime._address import SessionAddress
        from thorn.runtime._dispatch import apply_handling_transition
        from thorn.runtime._notification import NotificationStatus
        from thorn.runtime._inbox import SessionInbox

        async def slow_prompt(self_accessor, text, **kwargs):
            # Mark the oldest inbox item handled so the driver makes
            # progress and exits after a single round per session.
            session = self_accessor._session
            address = SessionAddress(session.agent.id, session.key)
            inbox = runtime.address_book.get(address)
            assert isinstance(inbox, SessionInbox)
            pending = inbox.prompt_pending()
            timestamps.append(("enter", asyncio.get_event_loop().time()))
            await asyncio.sleep(delay)
            timestamps.append(("exit", asyncio.get_event_loop().time()))
            if pending:
                apply_handling_transition(
                    inbox,
                    pending[0].id,
                    NotificationStatus.HANDLED,
                    address_book=runtime.address_book,
                )
            return "ok"

        with patch.object(
            _SessionPromptAccessor, "__call__", slow_prompt,
        ):
            t0 = asyncio.get_event_loop().time()
            await gateway.run()
            total = asyncio.get_event_loop().time() - t0

        # Both sessions should have run their prompt exactly once.
        assert len(timestamps) == 4, (
            f"Expected 4 timestamps (enter/exit × 2 sessions), got "
            f"{len(timestamps)}"
        )
        # With two concurrent sessions under a single agent scheduler
        # (concurrency cap >= 2), total wall time should be closer to
        # ``delay`` than to ``2 * delay``.
        assert total < delay * 1.8, (
            f"Two sessions of the same agent took {total:.2f}s — expected "
            f"concurrent execution (~{delay}s, not ~{delay * 2}s)"
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
# Gateway integration with the inbox / scheduler model
#
# These tests exercise the post-refactor invariants: lazy-but-durable
# session creation, per-session serialization via the scheduler, the
# N-strikes progress guarantee, RSVP delivery to a registered service
# queue, and crash recovery via the startup sweep.  They are written as
# black-box tests against the ``Gateway`` public surface wherever
# possible and reach into ``thorn.runtime`` internals only to seed or
# observe on-disk state.
# ---------------------------------------------------------------------------


class TestGatewayHealthMonitor:
    """Phase 2 wiring: a single shared monitor reaches every scheduler."""

    def _make_runtime(self, tmp_path: Path) -> Runtime:
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_default_monitor_constructed(self, tmp_path: Path) -> None:
        from thorn.runtime import ProviderHealthMonitor

        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])
        assert isinstance(gateway.health_monitor, ProviderHealthMonitor)
        # Convenience snapshot accessor.
        snap = gateway.health_snapshot()
        assert snap.state.value == "healthy"

    @pytest.mark.asyncio
    async def test_explicit_monitor_is_shared_across_schedulers(
        self, tmp_path: Path,
    ) -> None:
        # When the same monitor is wired into multiple schedulers,
        # ``_ensure_scheduler_for_agent`` must hand each scheduler
        # the same instance.  This is the property the gateway-wide
        # circuit breaker relies on.
        from thorn.runtime import ProviderHealthMonitor

        monitor = ProviderHealthMonitor()
        runtime = self._make_runtime(tmp_path)
        gateway = Gateway(
            runtime=runtime,
            sources=[],
            health_monitor=monitor,
        )

        async with runtime:
            agent_a = Agent(id=AgentID("a"), name="a")
            agent_b = Agent(id=AgentID("b"), name="b")
            sched_a = gateway._ensure_scheduler_for_agent(agent_a)
            sched_b = gateway._ensure_scheduler_for_agent(agent_b)

            # Both schedulers should reference the gateway's monitor.
            assert sched_a._health_monitor is monitor
            assert sched_b._health_monitor is monitor
            assert gateway.health_monitor is monitor

            await sched_a.shutdown(timeout=0)
            await sched_b.shutdown(timeout=0)


class TestGatewayInboxIntegration:
    # Integration tests under this class drive the real ``Gateway``
    # lifecycle (``gateway.run()`` + shutdown).  They override the
    # default shutdown grace period so the tests don't pay the full
    # 30-second production budget on every teardown.  The override is
    # applied via :meth:`_make_gateway` below.
    _TEST_SHUTDOWN_TIMEOUT = 1.0

    def _make_runtime(self, tmp_path: Path) -> Runtime:
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )

    def _make_gateway(
        self,
        runtime: Runtime,
        sources: list[EventSource],
        *,
        prompt_dispatcher: Any = None,
        agent_concurrency: int = 3,
        shutdown_timeout: float | None = None,
    ) -> Gateway:
        return Gateway(
            runtime=runtime,
            sources=sources,
            tools=[],
            agent_concurrency=agent_concurrency,
            prompt_dispatcher=prompt_dispatcher,
            shutdown_timeout=(
                shutdown_timeout
                if shutdown_timeout is not None
                else self._TEST_SHUTDOWN_TIMEOUT
            ),
        )

    @pytest.mark.asyncio
    async def test_same_session_events_serialized(self, tmp_path: Path):
        """Two events for the same session should be processed serially.

        Each session has exactly one driver inside its agent's scheduler,
        so overlapping prompts for the same ``(agent_id, session_key)``
        pair cannot run concurrently -- even if the agent concurrency
        cap would otherwise allow multiple prompts in flight.
        """
        from thorn.runtime._address import SessionAddress
        from thorn.runtime._dispatch import apply_handling_transition
        from thorn.runtime._inbox import SessionInbox
        from thorn.runtime._notification import NotificationStatus

        agent_id = AgentID("serial-session-agent")
        key = SessionKey("shared-session")
        delay = 0.1

        events = [
            IncomingEvent(
                source="test",
                session_key=key,
                content=f"message {i}",
                agent_id=agent_id,
            )
            for i in range(2)
        ]
        source = StubSource(events)
        runtime = self._make_runtime(tmp_path)
        gateway = self._make_gateway(runtime, [source])

        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def slow_prompt(self_accessor, text, **kwargs):
            nonlocal active, max_active
            session = self_accessor._session
            address = SessionAddress(session.agent.id, session.key)
            inbox = runtime.address_book.get(address)
            assert isinstance(inbox, SessionInbox)
            async with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                await asyncio.sleep(delay)
            finally:
                async with lock:
                    active -= 1
            pending = inbox.prompt_pending()
            if pending:
                apply_handling_transition(
                    inbox,
                    pending[0].id,
                    NotificationStatus.HANDLED,
                    address_book=runtime.address_book,
                )
            return "ok"

        with patch.object(
            _SessionPromptAccessor, "__call__", slow_prompt,
        ):
            await gateway.run()

        assert max_active == 1, (
            f"Per-session serialization violated: observed {max_active} "
            f"concurrent prompts for a single session"
        )

    @pytest.mark.asyncio
    async def test_startup_activates_session_with_pending_work(
        self, tmp_path: Path,
    ):
        """A session left with unprocessed inbox items is activated at
        startup without needing a fresh incoming event.

        This is the contract the activation pass enforces: if an
        inbox has anything in ``{pending, in_progress}`` on restart,
        the gateway submits it to its scheduler during ``_startup``
        so the driver drains it.  Previously this test relied on a
        fresh event to wake the driver; we now verify the stronger
        property that activation happens even with an empty source.
        """
        from thorn.runtime._address import SessionAddress
        from thorn.runtime._dispatch import apply_handling_transition
        from thorn.runtime._inbox import SessionInbox
        from thorn.runtime._notification import (
            NotificationSpec,
            NotificationStatus,
        )

        agent_id = AgentID("resume-agent")
        key = SessionKey("resume-key")

        # Pre-seed the on-disk inbox with an ``in_progress`` item
        # (simulating a prior-process crash mid-prompt) and a
        # ``pending`` item (a queued event that never got picked up).
        runtime_seed = self._make_runtime(tmp_path)
        async with runtime_seed:
            agent = runtime_seed.get_or_create_agent(agent_id)
            runtime_seed.save_agent(agent)
            ws = runtime_seed.paths.session_workspace(agent_id, key)
            ws.mkdir(parents=True, exist_ok=True)
            session = runtime_seed.get_or_create_session(
                agent, key, workspace_root=ws,
            )
            runtime_seed.save_session(session)
            inbox_dir_seed = runtime_seed.paths.session_inbox_dir(
                agent_id, key,
            )
            seed_inbox = SessionInbox(
                inbox_dir_seed, SessionAddress(agent_id, key),
            )
            in_progress_n = seed_inbox.post(NotificationSpec(
                source="test",
                content="mid-prompt work",
                target=SessionAddress(agent_id, key),
            ))
            seed_inbox.update_status(
                in_progress_n.id, NotificationStatus.IN_PROGRESS,
            )
            pending_n = seed_inbox.post(NotificationSpec(
                source="test",
                content="queued work",
                target=SessionAddress(agent_id, key),
            ))

        # No incoming events on this run -- the activation pass is
        # what must drive the driver.
        source = StubSource([])
        runtime = self._make_runtime(tmp_path)

        handled_ids: list[str] = []

        async def dispatcher(session_obj, inbox_obj):
            for item in list(inbox_obj.prompt_pending()):
                handled_ids.append(item.id)
                apply_handling_transition(
                    inbox_obj,
                    item.id,
                    NotificationStatus.HANDLED,
                    address_book=runtime.address_book,
                )

        gateway = self._make_gateway(
            runtime, [source], prompt_dispatcher=dispatcher,
        )
        await gateway.run()

        # Both items were handled by the activation-woken driver,
        # and the ``in_progress`` item retained its status until the
        # driver touched it (the sweep no longer reverts to pending).
        assert set(handled_ids) == {in_progress_n.id, pending_n.id}, (
            f"Expected both seeded items handled; got {handled_ids}"
        )
        inbox_dir = runtime.paths.session_inbox_dir(agent_id, key)
        remaining = [p.name for p in inbox_dir.glob("*.json")]
        assert remaining == [], (
            f"Inbox should be empty after handling, found {remaining}"
        )

    @pytest.mark.asyncio
    async def test_startup_sweep_dispatches_stuck_handled(
        self, tmp_path: Path,
    ):
        """A ``handled`` notification without an RSVP that survived a
        mid-step-2 crash is cleaned up (deleted) by the startup sweep.
        """
        from thorn.runtime._address import SessionAddress
        from thorn.runtime._inbox import SessionInbox
        from thorn.runtime._notification import (
            NotificationSpec,
            NotificationStatus,
        )

        agent_id = AgentID("stuck-handled-agent")
        key = SessionKey("stuck-session")

        runtime_seed = self._make_runtime(tmp_path)
        async with runtime_seed:
            agent = runtime_seed.get_or_create_agent(agent_id)
            runtime_seed.save_agent(agent)
            ws = runtime_seed.paths.session_workspace(agent_id, key)
            ws.mkdir(parents=True, exist_ok=True)
            session = runtime_seed.get_or_create_session(
                agent, key, workspace_root=ws,
            )
            runtime_seed.save_session(session)
            inbox_dir = runtime_seed.paths.session_inbox_dir(agent_id, key)
            inbox = SessionInbox(inbox_dir, SessionAddress(agent_id, key))
            spec = NotificationSpec(
                source="test",
                content="completed but not cleaned up",
                target=SessionAddress(agent_id, key),
            )
            notification = inbox.post(spec)
            inbox.update_status(
                notification.id, NotificationStatus.HANDLED,
            )

        runtime = self._make_runtime(tmp_path)
        gateway = self._make_gateway(runtime, [StubSource([])])
        await gateway.run()

        # Step 2 for a handled, no-RSVP item is an in-place delete; the
        # inbox should now be empty.
        inbox_dir = runtime.paths.session_inbox_dir(agent_id, key)
        remaining = [p.name for p in inbox_dir.glob("*.json")]
        assert remaining == [], (
            f"Sweep should have deleted the stuck handled item; "
            f"found {remaining}"
        )

    @pytest.mark.asyncio
    async def test_rsvp_delivered_to_service_queue(self, tmp_path: Path):
        """When an inbox item carries an ``rsvp_to`` service address and
        the agent marks it handled, the dispatch step moves the file to
        the target service's notification queue.

        Real sources will populate ``rsvp_to`` on the
        :class:`NotificationSpec` they post; today's gateway does not
        yet build specs that way from :class:`IncomingEvent` (see the
        pending ``source-refactors`` plan item).  This test pre-seeds
        a spec with an RSVP on disk and then triggers the driver with
        an ordinary event to keep the end-to-end scheduler / dispatch
        path in scope.
        """
        from thorn.runtime._address import ServiceAddress, SessionAddress
        from thorn.runtime._dispatch import apply_handling_transition
        from thorn.runtime._inbox import SessionInbox
        from thorn.runtime._notification import (
            NotificationSpec,
            NotificationStatus,
        )
        from thorn.runtime._notification_queue import NotificationQueue

        agent_id = AgentID("rsvp-agent")
        key = SessionKey("rsvp-session")
        service_name = "test-service"
        service_addr = ServiceAddress(service_name)

        # Seed the on-disk inbox with a notification that RSVPs to the
        # service queue, using a throwaway runtime.
        seed_runtime = self._make_runtime(tmp_path)
        async with seed_runtime:
            agent = seed_runtime.get_or_create_agent(agent_id)
            seed_runtime.save_agent(agent)
            ws = seed_runtime.paths.session_workspace(agent_id, key)
            ws.mkdir(parents=True, exist_ok=True)
            session = seed_runtime.get_or_create_session(
                agent, key, workspace_root=ws,
            )
            seed_runtime.save_session(session)
            seed_inbox_dir = seed_runtime.paths.session_inbox_dir(
                agent_id, key,
            )
            rsvp_spec = NotificationSpec(
                source="test",
                content="please rsvp",
                target=SessionAddress(agent_id, key),
                rsvp_to=service_addr,
            )
            rsvp_notification = SessionInbox(
                seed_inbox_dir, SessionAddress(agent_id, key),
            ).post(rsvp_spec)

        runtime = self._make_runtime(tmp_path)
        service_queue_dir = runtime.paths.service_queue_dir(service_name)
        service_queue_dir.mkdir(parents=True, exist_ok=True)

        # A fresh event so the driver has something to drive on.
        wake_event = IncomingEvent(
            source="test",
            session_key=key,
            content="wake",
            agent_id=agent_id,
        )
        source = StubSource([wake_event])

        async def handling_dispatcher(session_obj, inbox_obj):
            # Register the service queue on the runtime's address
            # book the first time we run.  This mirrors what a real
            # service startup path would do (e.g. registering a
            # GitLab service's RSVP queue at gateway boot).
            if service_addr not in runtime.address_book:
                runtime.address_book.register(
                    service_addr,
                    NotificationQueue(service_queue_dir, service_addr),
                )
            for item in list(inbox_obj.prompt_pending()):
                apply_handling_transition(
                    inbox_obj,
                    item.id,
                    NotificationStatus.HANDLED,
                    address_book=runtime.address_book,
                )

        gateway = self._make_gateway(
            runtime, [source], prompt_dispatcher=handling_dispatcher,
        )
        await gateway.run()

        # After handling, the RSVP'd item should have been moved out
        # of the session inbox and into the service queue.
        inbox_dir = runtime.paths.session_inbox_dir(agent_id, key)
        assert list(inbox_dir.glob("*.json")) == [], (
            f"Session inbox should be empty after RSVP + handled; "
            f"found {[p.name for p in inbox_dir.glob('*.json')]}"
        )

        service_items = list(service_queue_dir.glob("*.json"))
        assert len(service_items) == 1, (
            f"Expected 1 notification in service queue, found "
            f"{[p.name for p in service_items]}"
        )
        # And it should be the RSVP'd one (not the wake event).
        assert service_items[0].stem == rsvp_notification.id

    @pytest.mark.asyncio
    async def test_n_strikes_evicts_oldest_item(self, tmp_path: Path):
        """If the dispatcher makes no progress for ``progress_strikes``
        consecutive rounds, the default progress evictor errors the
        oldest inbox item so the session can move on.

        This test keeps the event source alive (a :class:`SlowSource`)
        so the gateway does not start shutting down before the driver
        has had a chance to exhaust its strike budget.  A polling await
        loop watches the session's ``errored/`` directory on disk as
        the externally-visible signal of eviction.
        """
        agent_id = AgentID("stalling-agent")
        key = SessionKey("stuck-key")

        event = IncomingEvent(
            source="test",
            session_key=key,
            content="never handled",
            agent_id=agent_id,
        )
        source = SlowSource()
        runtime = self._make_runtime(tmp_path)

        rounds = 0

        async def no_progress_dispatcher(session_obj, inbox_obj):
            nonlocal rounds
            rounds += 1
            # Yield the event loop between rounds so the driver does
            # not hog it and so shutdown (when it eventually runs) has
            # a chance to be observed.
            await asyncio.sleep(0)

        gateway = self._make_gateway(
            runtime,
            [source],
            prompt_dispatcher=no_progress_dispatcher,
            shutdown_timeout=0.5,
        )

        gateway_task = asyncio.create_task(gateway.run())
        try:
            # Wait for the gateway to enter its run loop, then post
            # the event directly via _handle_event (bypassing the
            # source so we control timing).
            while not gateway._started:  # type: ignore[attr-defined]
                await asyncio.sleep(0.01)
            await gateway._handle_event(event)

            errored_dir = runtime.paths.session_inbox_errored_dir(
                agent_id, key,
            )

            # Poll for eviction to land on disk.  Budget is generous
            # relative to the ~zero work per round and the 3-strike
            # threshold, but bounded so a genuine bug is caught.
            deadline = asyncio.get_event_loop().time() + 2.0
            while asyncio.get_event_loop().time() < deadline:
                if errored_dir.is_dir() and list(errored_dir.glob("*.json")):
                    break
                await asyncio.sleep(0.01)
        finally:
            await gateway.shutdown()
            await gateway_task

        assert rounds >= 3, f"Dispatcher was only invoked {rounds} times"

        errored_items = list(errored_dir.glob("*.json"))
        assert len(errored_items) == 1, (
            f"Expected oldest item evicted to errored/; found "
            f"{[p.name for p in errored_items]}"
        )

        inbox_dir = runtime.paths.session_inbox_dir(agent_id, key)
        assert [p for p in inbox_dir.glob("*.json")] == []

    @pytest.mark.asyncio
    async def test_agent_concurrency_cap_serializes_sessions(
        self, tmp_path: Path,
    ):
        """With ``agent_concurrency=1``, two different sessions of the
        same agent must run sequentially even though they have distinct
        drivers.  This verifies that the per-agent semaphore does what
        it claims.
        """
        from thorn.runtime._address import SessionAddress
        from thorn.runtime._dispatch import apply_handling_transition
        from thorn.runtime._inbox import SessionInbox
        from thorn.runtime._notification import NotificationStatus

        agent_id = AgentID("cap-agent")
        delay = 0.1

        events = [
            IncomingEvent(
                source="test",
                session_key=SessionKey(f"k{i}"),
                content=f"msg {i}",
                agent_id=agent_id,
            )
            for i in range(2)
        ]
        source = StubSource(events)
        runtime = self._make_runtime(tmp_path)

        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def serialized_prompt(self_accessor, text, **kwargs):
            nonlocal active, max_active
            session = self_accessor._session
            address = SessionAddress(session.agent.id, session.key)
            inbox = runtime.address_book.get(address)
            assert isinstance(inbox, SessionInbox)
            async with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                await asyncio.sleep(delay)
            finally:
                async with lock:
                    active -= 1
            pending = inbox.prompt_pending()
            if pending:
                apply_handling_transition(
                    inbox,
                    pending[0].id,
                    NotificationStatus.HANDLED,
                    address_book=runtime.address_book,
                )
            return "ok"

        gateway = self._make_gateway(
            runtime, [source], agent_concurrency=1,
        )
        with patch.object(
            _SessionPromptAccessor, "__call__", serialized_prompt,
        ):
            await gateway.run()

        assert max_active == 1, (
            f"agent_concurrency=1 violated: observed {max_active} "
            f"concurrent prompts under a single agent"
        )

    @pytest.mark.asyncio
    async def test_event_with_in_flight_external_key_is_dropped(
        self, tmp_path: Path,
    ):
        """A fresh event whose ``external_key`` is already recorded in
        the runtime's :class:`InFlightIndex` must be silently dropped
        by the gateway -- no inbox post, no session persistence, no
        scheduler submit.

        This is the contract sources rely on to avoid re-delivering
        the same external entity across polls (and across restarts,
        via the startup rebuild of the index).  We drive the gateway
        startup explicitly rather than through :meth:`Gateway.run`
        because the behaviour we care about is entirely on the
        receiving-side ``_handle_event`` path.
        """
        from thorn.runtime._address import SessionAddress

        agent_id = AgentID("dedup-agent")
        key = SessionKey("dedup-session")
        runtime = self._make_runtime(tmp_path)

        async with runtime:
            gateway = self._make_gateway(runtime, [])
            await gateway._startup()
            # Pretend an earlier post already placed this key in the
            # index.  A real agency would reach this state either via
            # the rebuild-from-disk or via a prior successful post.
            runtime.in_flight_index.add("gitlab:example:todo:42")

            duplicate = IncomingEvent(
                source="test",
                session_key=key,
                content="should be dropped",
                agent_id=agent_id,
                external_key="gitlab:example:todo:42",
            )
            await gateway._handle_event(duplicate)

            # No agent was persisted, no session created, no inbox
            # registered -- the gateway short-circuited before any
            # of that work.
            assert not runtime.sessions.agent_exists(agent_id)
            assert not runtime.sessions.session_exists(agent_id, key)
            assert SessionAddress(agent_id, key) not in gateway._inboxes

    @pytest.mark.asyncio
    async def test_event_with_fresh_external_key_is_posted(
        self, tmp_path: Path,
    ):
        """An event whose ``external_key`` is not in flight must post
        normally and register the key in the index, so a subsequent
        identical event would be deduplicated."""
        agent_id = AgentID("fresh-key-agent")
        key = SessionKey("fresh-key-session")
        runtime = self._make_runtime(tmp_path)

        async with runtime:
            gateway = self._make_gateway(runtime, [])
            await gateway._startup()
            assert "gitlab:example:todo:99" not in runtime.in_flight_index

            event = IncomingEvent(
                source="test",
                session_key=key,
                content="fresh",
                agent_id=agent_id,
                external_key="gitlab:example:todo:99",
            )
            await gateway._handle_event(event)

            assert runtime.sessions.session_exists(agent_id, key)
            assert "gitlab:example:todo:99" in runtime.in_flight_index

            await gateway.shutdown()


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
        assert "forge_mark_notification_done" not in event.content
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

    def test_make_session_key_with_project_name(self):
        from thorn.gateway.sources._gitlab import _make_session_key

        todo = _make_mock_todo(project_id=456, noteable_type="MergeRequest", noteable_iid=7)
        key = _make_session_key(todo, project_id_to_name={"456": "lace"})
        assert key == SessionKey("lace/change-request/7")

    def test_make_event_with_project_name(self):
        from thorn.gateway.sources._gitlab import _make_event

        todo = _make_mock_todo(
            todo_id=99, project_id=123, noteable_type="Issue",
            noteable_iid=42, action_name="mentioned",
        )
        event = _make_event(todo, project_id_to_name={"123": "my-proj"})
        assert event.session_key == SessionKey("my-proj/issue/42")
        assert event.metadata["project_name"] == "my-proj"

    def test_make_event_unknown_project_falls_back(self):
        from thorn.gateway.sources._gitlab import _make_event

        todo = _make_mock_todo(
            todo_id=99, project_id=999, noteable_type="Issue", noteable_iid=1,
        )
        event = _make_event(todo, project_id_to_name={"123": "other"})
        assert event.session_key == SessionKey("gitlab/999/issue/1")
        assert event.metadata["project_name"] == ""

    def test_format_event_content_includes_project_info(self):
        from thorn.gateway.sources._gitlab import _format_event_content

        todo = _make_mock_todo()
        content = _format_event_content(todo)
        assert "forge_mark_notification_done" not in content
        assert "marked done on your behalf" in content
        assert "Clone URL:" in content
        assert "Default branch:" in content
        assert "Project URL:" in content

    def test_make_event_sets_namespaced_external_key(self):
        from thorn.gateway.sources._gitlab import _make_event

        todo = _make_mock_todo(
            todo_id=99, project_id=123, noteable_type="Issue",
            noteable_iid=42, action_name="mentioned",
        )
        event = _make_event(
            todo, gitlab_url="https://gitlab.example.com",
        )
        # Key must be source-namespaced and unique per TODO so the
        # InFlightIndex can dedupe across polls and restarts without
        # cross-source collisions.
        assert event.external_key == (
            "gitlab:https://gitlab.example.com:todo:99"
        )


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
    async def test_marks_todo_as_done_after_post(self):
        """GitLab source should mark each TODO as done on GitLab's side
        as soon as it has safely posted the event.  This keeps the
        pending-TODO list bounded even when sessions take their time
        handling notifications."""
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib") as mock_gl_mod,
        ):
            mock_gl_instance = MagicMock()
            mock_gl_mod.Gitlab.return_value = mock_gl_instance

            todo = _make_mock_todo(todo_id=77)
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

            async def on_event(_event: IncomingEvent) -> None:
                pass

            await source._poll_once(on_event)

            todo.mark_as_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_failure_skips_mark_done(self):
        """If the gateway raises, we must not mark the TODO done --
        otherwise the TODO would be lost on both sides."""
        with (
            patch("thorn.gateway.sources._gitlab._HAS_GITLAB", True),
            patch("thorn.gateway.sources._gitlab._gitlab_lib") as mock_gl_mod,
        ):
            mock_gl_instance = MagicMock()
            mock_gl_mod.Gitlab.return_value = mock_gl_instance

            todo = _make_mock_todo(todo_id=78)
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

            async def on_event(_event: IncomingEvent) -> None:
                raise RuntimeError("boom")

            await source._poll_once(on_event)

            todo.mark_as_done.assert_not_called()

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

        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = Agent(id=AgentID("a:b/c"), name="special")
        store.save_agent(agent)

        assert store.agent_exists(AgentID("a:b/c"))
        loaded = store.load_agent(AgentID("a:b/c"))
        assert loaded.name == "special"

    def test_store_list_agent_ids_decodes(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.save_agent(Agent(id=AgentID("x:y"), name="xy"))
        store.save_agent(Agent(id=AgentID("simple"), name="simple"))

        ids = store.list_agent_ids()
        assert AgentID("simple") in ids
        assert AgentID("x:y") in ids

    def test_store_delete_agent_with_special_chars(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.save_agent(Agent(id=AgentID("del:me/now"), name="del"))
        assert store.agent_exists(AgentID("del:me/now"))
        store.delete_agent(AgentID("del:me/now"))
        assert not store.agent_exists(AgentID("del:me/now"))

    def test_clean_agent_ids_unchanged(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        paths = AgencyPaths(home_root=tmp_path, workspace_root=tmp_path)
        store = SessionStore(paths)
        store.save_agent(Agent(id=AgentID("gitlab_123_Issue_42"), name="test"))

        # Clean IDs round-trip to unmolested directory names.
        agent_dir_names = [
            d.name for d in paths.agents_root.iterdir() if d.is_dir()
        ]
        assert "gitlab_123_Issue_42" in agent_dir_names

    def test_store_session_with_special_chars(self, tmp_path: Path):
        from thorn.runtime._store import SessionStore

        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
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
        # Point --agency at an existing-but-empty tmp_path so the test
        # never accidentally probes the developer's real ~/.thorn home.
        result = runner.invoke(cli_main, ["serve", "--agency", str(tmp_path)])
        assert result.exit_code != 0
        assert (
            "gateway.json" in result.output
            or "Gateway configuration file not found" in result.output
        )


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

    def test_serialization_round_trip(self, tmp_path: Path, monkeypatch):
        from thorn.gateway._agents import ProjectCoordinator
        from thorn.runtime._serializer import JsonSessionSerializer

        monkeypatch.setenv("GITLAB_TOKEN", "x")

        agent = ProjectCoordinator(
            id=AgentID("test-coordinator"),
            name="test",
            metadata={"project": "my-proj"},
        )
        serializer = JsonSessionSerializer()
        # Parent directory name is the source of truth for AgentID
        # under the Phase-A layout; save under ``<id>/agent.json``.
        agent_dir = tmp_path / "test-coordinator"
        agent_dir.mkdir()
        path = agent_dir / "agent.json"
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
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="e2e-coord",
            project_name="test-proj",
            project_url="https://gitlab.com/group/test-proj",
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

        memory_path = agent.home / "MEMORY.md"
        assert memory_path.is_file()
        content = memory_path.read_text(encoding="utf-8")
        assert "test-proj" in content
        # The bootstrap MEMORY now records the project URL (the
        # human-facing one), not a derived clone URL.
        assert "https://gitlab.com/group/test-proj" in content

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
        assert "marked done on your behalf" in content


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
            "forges": [
                {
                    "name": "my-gitlab",
                    "type": "gitlab",
                    "url": "https://gitlab.example.com",
                },
            ],
            "projects": [
                {
                    "name": "my-proj",
                    "forks": [{
                        "forge": "my-gitlab",
                        "url": "https://gitlab.example.com/org/repo",
                        "name": "upstream",
                    }],
                },
            ],
        }
        (thorn_dir / "gateway.json").write_text(
            json.dumps(config_data), encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert len(config.forges) == 1
        assert config.forges[0].name == "my-gitlab"
        assert config.forges[0].type == "gitlab"
        # ``api_url`` is derived from ``url`` for GitLab (python-gitlab
        # appends ``/api/v4`` internally, so the API URL is the
        # instance URL itself).
        assert config.forges[0].url == "https://gitlab.example.com"
        assert config.forges[0].api_url == "https://gitlab.example.com"
        assert len(config.projects) == 1
        assert config.projects[0].name == "my-proj"

    def test_load_missing_file_raises(self, tmp_path: Path):
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="gateway.json"):
            load_gateway_config(thorn_dir)

    def test_load_empty_arrays(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"forges": [], "projects": []}), encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.forges == []
        assert config.projects == []

    def test_load_defaults_to_empty_arrays(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({}), encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.forges == []
        assert config.projects == []
        # Sandbox is opt-in: omitting the block leaves it None so the
        # runtime keeps the Phase-A subprocess executor.
        assert config.sandbox is None

    def test_load_with_sandbox_block(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({
                "sandbox": {
                    "oci_runtime": "podman",
                    "image": "thorn-sandbox:1.2",
                    "env_passthrough": ["LANG", "TZ"],
                    "dev_mount_runtime": True,
                    "container_ready_timeout_s": 45.0,
                },
            }),
            encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.sandbox is not None
        assert config.sandbox.backend == "container"
        assert config.sandbox.oci_runtime == "podman"
        assert config.sandbox.image == "thorn-sandbox:1.2"
        assert config.sandbox.env_passthrough == ["LANG", "TZ"]
        assert config.sandbox.dev_mount_runtime is True
        assert config.sandbox.container_ready_timeout_s == 45.0

    def test_load_sandbox_block_subprocess_backend(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"sandbox": {"backend": "subprocess"}}),
            encoding="utf-8",
        )
        config = load_gateway_config(thorn_dir)
        assert config.sandbox is not None
        assert config.sandbox.backend == "subprocess"

    def test_sandbox_invalid_backend_rejected(self, tmp_path: Path):
        import json
        from thorn.gateway._config import load_gateway_config

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"sandbox": {"backend": "vm"}}),
            encoding="utf-8",
        )
        with pytest.raises(Exception):
            load_gateway_config(thorn_dir)


# ---------------------------------------------------------------------------
# Service type registry & instantiate_services
# ---------------------------------------------------------------------------


class TestServiceTypeRegistry:
    def test_lookup_returns_known_forge_types(self):
        from thorn.gateway._config import get_service_type_registry

        registry = get_service_type_registry()
        assert set(registry.known_types("forge")) == {"gitlab", "github"}

    def test_unknown_type_raises_value_error(self):
        from pydantic import BaseModel

        from thorn.gateway._config import ForgeSpec, get_service_type_registry

        registry = get_service_type_registry()
        # ``url`` is required by the new validator; pick a host that
        # isn't a well-known forge so the type really is "unknown".
        spec = ForgeSpec(
            name="x", type="not-a-real-forge", url="https://example.com",
        )
        with pytest.raises(ValueError, match="Unknown forge type"):
            registry.instantiate(
                "forge", "not-a-real-forge", spec=spec, name="x",
            )
        # Also confirm a totally bogus category is reported similarly.
        class _FakeSpec(BaseModel):
            pass

        with pytest.raises(ValueError, match="Unknown bogus type"):
            registry.instantiate(
                "bogus", "x", spec=_FakeSpec(), name="x",
            )

    def test_register_routes_through_typed_array(self):
        """The forge typed-array path actually consults the registry."""
        from pydantic import BaseModel

        from thorn.core._service import Service
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            get_service_type_registry,
            instantiate_services,
        )

        registry = get_service_type_registry()

        class _FakeForgeConfig(BaseModel):
            label: str

        class _FakeForge(Service):
            def __init__(
                self, config: _FakeForgeConfig, *, service_name: str,
            ) -> None:
                self._config = config
                self._service_name = service_name

            @property
            def name(self) -> str:
                return self._service_name

        registry.register(
            "forge", "fake-forge",
            _FakeForge, _FakeForgeConfig,
            spec_to_config=lambda spec: {"label": f"hello-{spec.name}"},
        )
        try:
            config = GatewayConfig(
                forges=[ForgeSpec(
                    name="ff", type="fake-forge",
                    url="https://example.com",
                )],
            )
            services = instantiate_services(config)
            assert len(services) == 1
            assert isinstance(services[0], _FakeForge)
            assert services[0]._config.label == "hello-ff"
        finally:
            # Restore the built-in registrations so other tests are unaffected.
            from thorn.tools._github_connection import GitHubConnectionConfig
            from thorn.tools.forge import GitHubForgeService
            from thorn.gateway._config import _github_forge_spec_to_config

            registry.register(
                "forge", "github",
                GitHubForgeService, GitHubConnectionConfig,
                spec_to_config=_github_forge_spec_to_config,
            )
            # Drop the fake registration; rebuild minimal known-types set.
            registry._entries.pop(("forge", "fake-forge"), None)


class TestForgeURLInference:
    """Tests for the URL/name/type inference helpers.

    These cover the small but visible behaviours that operators rely on
    to avoid restating defaults: deriving forge type from host,
    deriving forge name from host, deriving the API base URL from the
    instance URL + type, and parsing fork URLs into ``(native_id,
    clone_url)`` for the supported forges.
    """

    def test_derive_forge_type_for_github_com(self):
        from thorn.gateway._config import derive_forge_type_from_url

        assert derive_forge_type_from_url("https://github.com") == "github"
        assert (
            derive_forge_type_from_url("https://github.com/owner/repo")
            == "github"
        )
        # The ``api.`` prefix on an API host should still resolve to
        # the same forge type.
        assert (
            derive_forge_type_from_url("https://api.github.com") == "github"
        )

    def test_derive_forge_type_for_gitlab_com(self):
        from thorn.gateway._config import derive_forge_type_from_url

        assert derive_forge_type_from_url("https://gitlab.com") == "gitlab"
        assert (
            derive_forge_type_from_url("https://gitlab.com/group/project")
            == "gitlab"
        )

    def test_derive_forge_type_returns_none_for_unknown_hosts(self):
        """Self-hosted hosts cannot have their type guessed."""
        from thorn.gateway._config import derive_forge_type_from_url

        assert (
            derive_forge_type_from_url("https://gitlab.example.com/g/p")
            is None
        )
        assert (
            derive_forge_type_from_url("https://github.example.com/o/r")
            is None
        )

    def test_derive_forge_name_for_well_known_hosts(self):
        from thorn.gateway._config import derive_forge_name_from_url

        assert derive_forge_name_from_url("https://github.com") == "github"
        assert derive_forge_name_from_url("https://gitlab.com") == "gitlab"
        # ``api.`` prefix is stripped before naming.
        assert (
            derive_forge_name_from_url("https://api.github.com") == "github"
        )

    def test_derive_forge_name_for_self_hosted_uses_hyphenated_host(self):
        from thorn.gateway._config import derive_forge_name_from_url

        assert (
            derive_forge_name_from_url("https://gitlab.example.com")
            == "gitlab-example-com"
        )

    def test_derive_forge_name_rejects_url_without_host(self):
        from thorn.gateway._config import derive_forge_name_from_url

        with pytest.raises(ValueError, match="no hostname"):
            derive_forge_name_from_url("not-a-url")

    def test_derive_api_url_github_com(self):
        from thorn.gateway._config import derive_api_url

        assert (
            derive_api_url("github", "https://github.com")
            == "https://api.github.com"
        )
        assert (
            derive_api_url("github", "https://api.github.com")
            == "https://api.github.com"
        )

    def test_derive_api_url_github_enterprise(self):
        """GitHub Enterprise uses the ``/api/v3`` suffix on the host."""
        from thorn.gateway._config import derive_api_url

        assert (
            derive_api_url("github", "https://github.example.com")
            == "https://github.example.com/api/v3"
        )

    def test_derive_api_url_gitlab_uses_instance_url(self):
        """python-gitlab adds ``/api/v4`` itself, so the API URL *is*
        the instance URL."""
        from thorn.gateway._config import derive_api_url

        assert (
            derive_api_url("gitlab", "https://gitlab.com")
            == "https://gitlab.com"
        )
        assert (
            derive_api_url("gitlab", "https://gitlab.example.com")
            == "https://gitlab.example.com"
        )

    def test_derive_api_url_strips_legacy_api_v4_suffix_for_gitlab(self):
        """Old gateway configs sometimes embedded ``/api/v4`` directly
        (because the old config required it).  The new derivation
        strips it so the dogfood configs in the wild don't break."""
        from thorn.gateway._config import derive_api_url

        assert (
            derive_api_url("gitlab", "https://gitlab.example.com/api/v4")
            == "https://gitlab.example.com"
        )

    def test_parse_fork_url_github(self):
        from thorn.gateway._config import parse_fork_url

        loc = parse_fork_url("github", "https://github.com/owner/repo")
        assert loc.native_id == "owner/repo"
        assert loc.clone_url == "https://github.com/owner/repo.git"

    def test_parse_fork_url_github_strips_dot_git(self):
        from thorn.gateway._config import parse_fork_url

        loc = parse_fork_url(
            "github", "https://github.com/owner/repo.git",
        )
        assert loc.native_id == "owner/repo"
        assert loc.clone_url == "https://github.com/owner/repo.git"

    def test_parse_fork_url_gitlab_subgroup_path(self):
        """GitLab paths can be arbitrarily deep due to subgroups; the
        full ``group/subgroup/project`` path becomes the native id."""
        from thorn.gateway._config import parse_fork_url

        loc = parse_fork_url(
            "gitlab",
            "https://gitlab.com/group/subgroup/project",
        )
        assert loc.native_id == "group/subgroup/project"
        assert (
            loc.clone_url
            == "https://gitlab.com/group/subgroup/project.git"
        )

    def test_parse_fork_url_gitlab_strips_dash_suffix(self):
        """GitLab URLs frequently include ``/-/issues/N`` or similar
        suffixes when copy-pasted from a browser; the project portion
        of the path should be preserved, the rest dropped."""
        from thorn.gateway._config import parse_fork_url

        loc = parse_fork_url(
            "gitlab",
            "https://gitlab.com/group/project/-/issues/7",
        )
        assert loc.native_id == "group/project"

    def test_parse_fork_url_rejects_empty_path(self):
        from thorn.gateway._config import parse_fork_url

        with pytest.raises(ValueError, match="no project path"):
            parse_fork_url("github", "https://github.com")

    def test_parse_fork_url_rejects_single_segment_github_path(self):
        from thorn.gateway._config import parse_fork_url

        with pytest.raises(ValueError, match="owner/repo"):
            parse_fork_url("github", "https://github.com/owner")

    def test_parse_fork_url_rejects_single_segment_gitlab_path(self):
        from thorn.gateway._config import parse_fork_url

        with pytest.raises(ValueError, match="group and project"):
            parse_fork_url("gitlab", "https://gitlab.com/owner")


class TestForgeSpecValidator:
    """The ForgeSpec model validator fills in defaults from URL."""

    def test_name_defaults_to_url_host(self):
        from thorn.gateway._config import ForgeSpec

        spec = ForgeSpec(url="https://github.com")
        assert spec.name == "github"

    def test_type_inferred_for_well_known_hosts(self):
        from thorn.gateway._config import ForgeSpec

        spec = ForgeSpec(url="https://gitlab.com")
        assert spec.type == "gitlab"

    def test_self_hosted_url_requires_explicit_type(self):
        from thorn.gateway._config import ForgeSpec

        with pytest.raises(ValueError, match="Cannot infer forge type"):
            ForgeSpec(url="https://gitlab.example.com")

    def test_api_url_filled_from_url_and_type(self):
        from thorn.gateway._config import ForgeSpec

        spec = ForgeSpec(
            url="https://gitlab.example.com", type="gitlab",
        )
        assert spec.api_url == "https://gitlab.example.com"


class TestProjectInferredForges:
    """Forge entries are synthesized from project URLs at load time."""

    def test_well_known_forge_synthesized_when_array_omitted(self):
        """A project on github.com works even when ``forges`` is empty."""
        from thorn.gateway._config import (
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )
        from thorn.tools.forge import GitHubForgeService, ProjectService

        config = GatewayConfig(
            projects=[ProjectSpec(
                name="proj", url="https://github.com/owner/repo",
            )],
        )
        services = instantiate_services(config)
        assert len(services) == 2
        forge_svcs = [s for s in services if isinstance(s, GitHubForgeService)]
        proj_svcs = [s for s in services if isinstance(s, ProjectService)]
        assert len(forge_svcs) == 1 and forge_svcs[0].name == "github"
        # The synthesized ProjectService records the path-based native id.
        assert proj_svcs[0].native_id == "owner/repo"

    def test_self_hosted_project_url_requires_explicit_forge_entry(self):
        """For non-well-known hosts the user must declare the forge."""
        from thorn.gateway._config import (
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )

        config = GatewayConfig(
            projects=[ProjectSpec(
                name="proj",
                url="https://gitlab.example.com/group/project",
            )],
        )
        with pytest.raises(ValueError, match="not a well-known forge"):
            instantiate_services(config)

    def test_gitlab_subgroup_path_used_as_native_id(self):
        """GitLab projects keep their full path-with-namespace."""
        from thorn.gateway._config import (
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )
        from thorn.tools.forge import ProjectService

        config = GatewayConfig(
            projects=[ProjectSpec(
                name="proj",
                url="https://gitlab.com/group/subgroup/project",
            )],
        )
        services = instantiate_services(config)
        proj_svcs = [s for s in services if isinstance(s, ProjectService)]
        assert proj_svcs[0].native_id == "group/subgroup/project"

    def test_default_fork_name_is_origin_for_single_fork(self):
        """When a project has exactly one fork its remote name is
        ``"origin"`` (matching git's own default)."""
        from thorn.gateway._config import (
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )
        from thorn.tools.forge import ProjectService

        config = GatewayConfig(
            projects=[ProjectSpec(
                name="proj", url="https://github.com/owner/repo",
            )],
        )
        services = instantiate_services(config)
        proj = next(s for s in services if isinstance(s, ProjectService))
        assert proj.forks[0].name == "origin"

    def test_default_fork_name_is_forge_name_for_multi_fork(self):
        """Multi-fork projects use the forge name to discriminate.

        Two forks on different forges therefore get the two forge
        names as their remote names, with no ``origin`` collision.
        """
        from thorn.gateway._config import (
            ForgeSpec,
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )
        from thorn.tools.forge import ProjectService

        config = GatewayConfig(
            forges=[
                ForgeSpec(url="https://github.com"),
                ForgeSpec(url="https://gitlab.com"),
            ],
            projects=[ProjectSpec(
                name="proj",
                forks=[
                    ForkSpec(url="https://github.com/owner/repo"),
                    ForkSpec(url="https://gitlab.com/group/repo"),
                ],
            )],
        )
        services = instantiate_services(config)
        proj = next(s for s in services if isinstance(s, ProjectService))
        names = sorted(f.name for f in proj.forks)
        assert names == ["github", "gitlab"]


class TestProjectSpecValidator:
    """Project-level validation: ``url`` and ``forks`` are mutually exclusive."""

    def test_url_xor_forks_required(self):
        from thorn.gateway._config import ProjectSpec

        with pytest.raises(ValueError, match="must specify either"):
            ProjectSpec(name="empty")

    def test_url_and_forks_cannot_coexist(self):
        from thorn.gateway._config import ForkSpec, ProjectSpec

        with pytest.raises(ValueError, match="cannot specify both"):
            ProjectSpec(
                name="ambiguous",
                url="https://github.com/o/r",
                forks=[ForkSpec(url="https://github.com/o/r2")],
            )


class TestInstantiateServices:
    def test_creates_forge_and_project_services(self):
        from thorn.gateway._config import (
            ForgeSpec,
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )
        from thorn.tools.forge import GitLabForgeService, ProjectService

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", url="https://gl.example.com")],
            projects=[ProjectSpec(
                name="my-proj",
                forks=[ForkSpec(
                    forge="gl",
                    url="https://gl.example.com/org/repo",
                )],
            )],
        )
        services = instantiate_services(config)
        assert len(services) == 2
        assert isinstance(services[0], GitLabForgeService)
        assert services[0].name == "gl"
        assert isinstance(services[1], ProjectService)
        assert services[1].name == "my-proj"

    def test_github_forge_service(self):
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            instantiate_services,
        )
        from thorn.tools.forge import GitHubForgeService

        config = GatewayConfig(
            forges=[ForgeSpec(name="gh", type="github", url="https://github.com")],
        )
        services = instantiate_services(config)
        assert len(services) == 1
        assert isinstance(services[0], GitHubForgeService)

    def test_github_forge_uses_default_api_url_when_url_omitted(self):
        """When ``url`` is set to ``https://github.com``, the API URL is
        derived as ``https://api.github.com``."""
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            instantiate_services,
        )
        from thorn.tools.forge import GitHubForgeService

        config = GatewayConfig(
            forges=[ForgeSpec(url="https://github.com")],
        )
        services = instantiate_services(config)
        assert isinstance(services[0], GitHubForgeService)
        assert services[0].base_url == "https://api.github.com"

    def test_gitlab_forge_requires_url(self):
        """A GitLab forge entry without a URL has no API endpoint."""
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
        )

        with pytest.raises(ValueError, match="ForgeSpec requires"):
            GatewayConfig(
                forges=[ForgeSpec(name="gl", type="gitlab")],
            )

    def test_unknown_forge_type_raises(self):
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            instantiate_services,
        )

        # ``url`` is required by the new validator; pick a host that
        # isn't a well-known forge so the type really is "unknown".
        config = GatewayConfig(
            forges=[ForgeSpec(
                name="x", type="unknown", url="https://example.com",
            )],
        )
        with pytest.raises(ValueError, match="Unknown forge type"):
            instantiate_services(config)

    def test_empty_config_returns_empty_list(self):
        from thorn.gateway._config import GatewayConfig, instantiate_services

        assert instantiate_services(GatewayConfig()) == []

    def test_project_spec_resolved_forks_from_explicit(self):
        from thorn.gateway._config import ForkSpec, ProjectSpec

        proj = ProjectSpec(
            name="p",
            forks=[
                ForkSpec(url="https://github.com/a/repo", name="upstream"),
                ForkSpec(url="https://github.com/b/repo", name="origin"),
            ],
        )
        forks = proj.resolved_forks()
        assert len(forks) == 2
        assert forks[0].url == "https://github.com/a/repo"
        assert forks[1].url == "https://github.com/b/repo"

    def test_project_spec_url_shorthand(self):
        """Single-fork projects can use the top-level ``url`` shorthand
        which is equivalent to a single-element ``forks`` array."""
        from thorn.gateway._config import ProjectSpec

        proj = ProjectSpec(
            name="p", url="https://github.com/owner/repo",
        )
        forks = proj.resolved_forks()
        assert len(forks) == 1
        assert forks[0].url == "https://github.com/owner/repo"

    def test_project_with_forks_creates_correct_project_service(self):
        from thorn.gateway._config import (
            ForgeSpec,
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
            instantiate_services,
        )
        from thorn.tools.forge import ProjectService

        config = GatewayConfig(
            forges=[ForgeSpec(name="gh", type="github", url="https://github.com")],
            projects=[ProjectSpec(
                name="my-proj",
                forks=[
                    ForkSpec(forge="gh",
                             url="https://github.com/owner/upstream",
                             name="upstream"),
                    ForkSpec(forge="gh",
                             url="https://github.com/bot/fork",
                             name="origin"),
                ],
                default_branch="main",
            )],
        )
        services = instantiate_services(config)
        proj_svc = [s for s in services if isinstance(s, ProjectService)][0]
        assert len(proj_svc.forks) == 2
        assert proj_svc.forge_name == "gh"
        assert proj_svc.native_id == "owner/upstream"


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
                forges=[ForgeSpec(name="gl", type="gitlab", url="https://gl.example.com")],
            )
            agent = Agent(
                name="bot",
                accounts=AgentAccountsConfig(accounts=[
                    ForgeAccountConfig(
                        service="gl",
                        credentials=GitLabCredentials(token="tok"),
                    ),
                ]),
            )
            sources = infer_event_sources(config, [agent])
            assert len(sources) == 1
            assert isinstance(sources[0], GitLabTODOsSource)
            assert "bot-gl-events" in sources[0].name

    def test_infers_github_source_from_agent_account(self):
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
            forges=[ForgeSpec(name="gh", type="github", url="https://github.com")],
            projects=[ProjectSpec(
                name="repo", url="https://github.com/owner/repo",
            )],
        )
        agent = Agent(
            name="bot",
            accounts=AgentAccountsConfig(accounts=[
                ForgeAccountConfig(
                    service="gh",
                    credentials=GitHubPatAuth(token="ghp-tok"),
                ),
            ]),
        )
        sources = infer_event_sources(config, [agent])
        assert len(sources) == 1
        assert isinstance(sources[0], GitHubNotificationsSource)
        assert sources[0]._config.native_id_to_project_name == {"owner/repo": "repo"}

    def test_no_sources_when_agent_has_no_accounts(self):
        from thorn.core._agent import Agent
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", url="https://gl.example.com")],
        )
        agent = Agent(name="bot")
        sources = infer_event_sources(config, [agent])
        assert sources == []

    def test_no_sources_when_no_agents(self):
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources

        config = GatewayConfig(
            forges=[ForgeSpec(name="gl", type="gitlab", url="https://gl.example.com")],
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
            forges=[ForgeSpec(name="gl", type="gitlab", url="https://gl.example.com")],
        )
        agent = Agent(
            name="bot",
            accounts=AgentAccountsConfig(accounts=[
                ForgeAccountConfig(
                    service="nonexistent",
                    credentials=GitLabCredentials(token="tok"),
                ),
            ]),
        )
        sources = infer_event_sources(config, [agent])
        assert sources == []

    def test_github_created_even_without_project_repos(self):
        """GitHub notifications source is user-scoped and created even
        without project repos (unlike the old per-repo events source)."""
        from thorn.core._account import AgentAccountsConfig, ForgeAccountConfig
        from thorn.core._agent import Agent
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources
        from thorn.gateway.sources._github import GitHubNotificationsSource
        from thorn.tools._github_connection import GitHubPatAuth

        config = GatewayConfig(
            forges=[ForgeSpec(name="gh", type="github", url="https://github.com")],
        )
        agent = Agent(
            name="bot",
            accounts=AgentAccountsConfig(accounts=[
                ForgeAccountConfig(
                    service="gh",
                    credentials=GitHubPatAuth(token="ghp-tok"),
                ),
            ]),
        )
        sources = infer_event_sources(config, [agent])
        assert len(sources) == 1
        assert isinstance(sources[0], GitHubNotificationsSource)

    def test_github_skipped_for_app_credentials(self):
        """GitHub App credentials are rejected since the Notifications
        API requires a personal access token."""
        from thorn.core._account import AgentAccountsConfig, ForgeAccountConfig
        from thorn.core._agent import Agent
        from thorn.gateway._config import ForgeSpec, GatewayConfig, infer_event_sources
        from thorn.tools._github_connection import GitHubAppAuth

        config = GatewayConfig(
            forges=[ForgeSpec(name="gh", type="github", url="https://github.com")],
        )
        agent = Agent(
            name="bot",
            accounts=AgentAccountsConfig(accounts=[
                ForgeAccountConfig(
                    service="gh",
                    credentials=GitHubAppAuth(
                        app_id="123",
                        installation_id=456,
                        private_key_pem="-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
                    ),
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
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="test-coord",
            project_name="my-project",
            project_url="https://gitlab.com/group/my-project",
        )

        assert str(aid) == "test-coord"

        identity = (
            tmp_path / ".thorn" / "agents" / "test-coord" / "agent.json"
        )
        assert identity.is_file()

        import json
        data = json.loads(identity.read_text(encoding="utf-8"))
        assert data["agent_class"] == "ProjectCoordinator"
        assert data["metadata"]["project"] == "my-project"

        # New shape: ``accounts`` is a flat array discriminated on
        # ``service`` (no inner ``forge_accounts`` wrapper, no
        # ``forge`` field).
        acct = data["accounts"][0]
        assert acct["service"] == "gitlab"
        assert acct["git_user_name"] == "test-coord"
        assert acct["git_user_email"] == "test-coord@thorn"
        assert acct["credentials"]["kind"] == "gitlab-pat"
        assert acct["credentials"]["token"] == "$GITLAB_TOKEN"

        memory = (
            tmp_path / ".thorn" / "agents" / "test-coord" / "home" / "MEMORY.md"
        )
        assert memory.is_file()
        content = memory.read_text(encoding="utf-8")
        assert "my-project" in content

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        assert gateway_config.is_file()
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))

        # The new gateway.json shape no longer needs an explicit
        # ``forges`` array (forges are inferred from project URLs)
        # and the project entry is collapsed to ``{name, url}`` for
        # the common single-fork case.
        assert "forges" not in gw_data
        assert len(gw_data["projects"]) == 1
        assert "services" not in gw_data

        proj = gw_data["projects"][0]
        assert proj["name"] == "my-project"
        assert proj["url"] == "https://gitlab.com/group/my-project"

    def test_bootstrap_appends_to_existing_gateway_config(self, tmp_path: Path):
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="first-coord",
            project_name="proj-a",
            project_url="https://github.com/owner/proj-a",
        )
        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="second-coord",
            project_name="proj-b",
            project_url="https://github.com/owner/proj-b",
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
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            project_url="https://github.com/owner/proj",
        )
        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            project_url="https://github.com/owner/proj-v2",
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))
        assert len(gw_data["projects"]) == 1
        assert gw_data["projects"][0]["url"] == "https://github.com/owner/proj-v2"

    def test_bootstrap_custom_token_env(self, tmp_path: Path):
        """Custom access_token_env is written into agent credentials."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="custom",
            project_name="proj",
            project_url="https://github.com/owner/proj",
            access_token_env="MY_TOKEN",
        )

        identity = (
            tmp_path / ".thorn" / "agents" / "custom" / "agent.json"
        )
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"][0]
        assert acct["credentials"]["token"] == "$MY_TOKEN"

    def test_bootstrap_rejects_self_hosted_url(self, tmp_path: Path):
        """An unrecognised forge host is rejected with a clear error.

        The bootstrap helper only supports the canonical GitHub and
        GitLab public hosts; for self-hosted forges, operators are
        expected to write ``gateway.json`` by hand with an explicit
        ``forges:`` entry rather than relying on URL inference.
        """
        from thorn.gateway._bootstrap import bootstrap_coordinator

        with pytest.raises(ValueError, match="well-known forge"):
            bootstrap_coordinator(
                agency_home=tmp_path / ".thorn",
                agency_workspace=tmp_path,
                agent_id="hosted",
                project_name="proj",
                project_url="https://gitlab.example.com/g/proj",
            )

    def test_loads_via_session_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from thorn.gateway._agents import ProjectCoordinator
        from thorn.gateway._bootstrap import bootstrap_coordinator
        from thorn.runtime._store import SessionStore

        monkeypatch.setenv("GITLAB_TOKEN", "fake-token")

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="loadable",
            project_name="proj",
            project_url="https://gitlab.com/group/proj",
        )

        from thorn.runtime._paths import AgencyPaths
        store = SessionStore(
            AgencyPaths.for_gateway(tmp_path / ".thorn", tmp_path),
        )
        agent = store.load_agent(AgentID("loadable"))

        assert isinstance(agent, ProjectCoordinator)
        assert agent.metadata["project"] == "proj"
        assert hasattr(agent, "accounts")
        assert len(agent.accounts.forge_accounts()) == 1

    def test_cli_bootstrap_command(self, tmp_path: Path):
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "cli-test",
            "--project-name", "test-proj",
            "--project-url", "https://gitlab.com/group/test-proj",
            "--agency-home", str(tmp_path / ".thorn"),
            "--agency-workspace", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output
        assert "cli-test" in result.output
        assert (
            tmp_path / ".thorn" / "agents" / "cli-test" / "agent.json"
        ).is_file()
        assert (tmp_path / ".thorn" / "gateway.json").is_file()
        assert "gateway.json" in result.output

    def test_cli_bootstrap_rejects_unknown_forge(self, tmp_path: Path):
        """The CLI rejects URLs whose host isn't a recognised forge.

        Self-hosted forges aren't supported by the bootstrap helper:
        operators are expected to write ``gateway.json`` by hand for
        those cases.  The CLI surfaces this as a clear early error
        rather than producing a half-broken config.
        """
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "needs-known",
            "--project-name", "proj",
            "--project-url", "https://gitlab.example.com/g/proj",
            "--agency-home", str(tmp_path / ".thorn"),
            "--agency-workspace", str(tmp_path),
        ])
        assert result.exit_code != 0
        assert "well-known forge" in result.output

    def test_github_bootstrap_pat_default(self, tmp_path: Path):
        """GitHub bootstrap uses PAT auth (App auth not supported)."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="gh-coord",
            project_name="my-repo",
            project_url="https://github.com/owner/repo",
        )

        gateway_config = tmp_path / ".thorn" / "gateway.json"
        gw_data = json.loads(gateway_config.read_text(encoding="utf-8"))

        # No explicit ``forges`` entry is needed any more; the
        # forge for github.com is inferred at load time.
        assert "forges" not in gw_data
        proj = gw_data["projects"][0]
        assert proj["url"] == "https://github.com/owner/repo"

        identity = (
            tmp_path / ".thorn" / "agents" / "gh-coord" / "agent.json"
        )
        data = json.loads(identity.read_text(encoding="utf-8"))
        assert data["metadata"]["project"] == "my-repo"
        acct = data["accounts"][0]
        assert acct["service"] == "github"
        assert acct["git_user_name"] == "gh-coord"
        assert acct["git_user_email"] == "gh-coord@thorn"
        assert acct["credentials"]["kind"] == "pat"
        assert acct["credentials"]["token"] == "$GITHUB_TOKEN"

    def test_bootstrap_writes_git_identity(self, tmp_path: Path):
        """Bootstrap writes git identity into agent accounts."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="id-test",
            project_name="proj",
            project_url="https://github.com/owner/proj",
        )

        identity = (
            tmp_path / ".thorn" / "agents" / "id-test" / "agent.json"
        )
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"][0]
        assert acct["git_user_name"] == "id-test"
        assert acct["git_user_email"] == "id-test@thorn"

    def test_bootstrap_custom_git_identity(self, tmp_path: Path):
        """Explicit git_user_name/email override the defaults."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="custom-id",
            project_name="proj",
            project_url="https://github.com/owner/proj",
            git_user_name="My Bot",
            git_user_email="bot@example.com",
        )

        identity = (
            tmp_path / ".thorn" / "agents" / "custom-id" / "agent.json"
        )
        data = json.loads(identity.read_text(encoding="utf-8"))
        acct = data["accounts"][0]
        assert acct["git_user_name"] == "My Bot"
        assert acct["git_user_email"] == "bot@example.com"

    def test_cli_bootstrap_github(self, tmp_path: Path):
        """CLI GitHub bootstrap defaults to PAT auth."""
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "bootstrap",
            "--agent-id", "gh-cli-test",
            "--project-name", "test-repo",
            "--project-url", "https://github.com/owner/repo",
            "--agency-home", str(tmp_path / ".thorn"),
            "--agency-workspace", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output
        assert "gh-cli-test" in result.output
        assert "GITHUB_TOKEN" in result.output

    def test_inferred_event_source_uses_inferred_forge_url(
        self, tmp_path: Path,
    ):
        """End-to-end: the GitLab event source inferred from a public
        ``gitlab.com`` project URL polls the public GitLab instance.

        The bootstrap no longer writes an explicit ``forges`` entry
        for the public hosts -- the host is inferred from the
        project URL on load -- so this test exercises that inference
        path end-to-end.
        """
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
            from thorn.gateway._bootstrap import bootstrap_coordinator
            from thorn.gateway._config import (
                infer_event_sources,
                load_gateway_config,
            )

            bootstrap_coordinator(
                agency_home=tmp_path / ".thorn",
                agency_workspace=tmp_path,
                agent_id="bot",
                project_name="proj",
                project_url="https://gitlab.com/g/proj",
            )

            config = load_gateway_config(tmp_path / ".thorn")

            agent = Agent(
                name="bot",
                accounts=AgentAccountsConfig(accounts=[
                    ForgeAccountConfig(
                        service="gitlab",
                        credentials=GitLabCredentials(token="tok"),
                    ),
                ]),
            )
            sources = infer_event_sources(config, [agent])
            assert len(sources) == 1
            assert sources[0]._config.url == "https://gitlab.com"


# ---------------------------------------------------------------------------
# Bootstrap: home/workspace split
# ---------------------------------------------------------------------------


class TestBootstrapHomeWorkspaceSplit:
    """Verify the new agency-home / agency-workspace separation."""

    def test_bootstrap_writes_workspace_into_gateway_config(
        self, tmp_path: Path,
    ):
        """The absolute workspace path is recorded in ``gateway.json``."""
        import json
        from thorn.gateway._bootstrap import bootstrap_coordinator

        agency_home = tmp_path / "agency"
        agency_workspace = tmp_path / "work"

        bootstrap_coordinator(
            agency_home=agency_home,
            agency_workspace=agency_workspace,
            agent_id="ws-coord",
            project_name="proj",
            project_url="https://github.com/owner/proj",
        )

        gw_data = json.loads(
            (agency_home / "gateway.json").read_text(encoding="utf-8"),
        )
        assert gw_data["workspace"] == str(agency_workspace.resolve())

    def test_bootstrap_does_not_nest_dot_thorn(self, tmp_path: Path):
        """Bootstrap treats ``agency_home`` as the home root verbatim.

        Older versions used to nest a ``.thorn/`` subdirectory under
        the supplied path, so we explicitly verify that no such
        directory is created under either the home or the workspace.
        """
        from thorn.gateway._bootstrap import bootstrap_coordinator

        agency_home = tmp_path / "agency"
        agency_workspace = tmp_path / "work"

        bootstrap_coordinator(
            agency_home=agency_home,
            agency_workspace=agency_workspace,
            agent_id="no-nest",
            project_name="proj",
            project_url="https://github.com/owner/proj",
        )

        assert not (agency_home / ".thorn").exists()
        assert not (agency_workspace / ".thorn").exists()
        assert (agency_home / "gateway.json").is_file()
        assert (agency_home / "agents" / "no-nest" / "agent.json").is_file()

    def test_bootstrap_creates_per_agent_workspace_prefix(
        self, tmp_path: Path,
    ):
        """The ``<agency_workspace>/agents/<agent_id>/`` subtree is created eagerly.

        Under the Phase-A layout both the workspace mount and the
        control dir sibling are created up front so ``thorn serve``
        can bind the tool-host socket on first use without racing
        against a missing directory.
        """
        from thorn.gateway._bootstrap import bootstrap_coordinator

        agency_home = tmp_path / "agency"
        agency_workspace = tmp_path / "work"

        bootstrap_coordinator(
            agency_home=agency_home,
            agency_workspace=agency_workspace,
            agent_id="eager",
            project_name="proj",
            project_url="https://github.com/owner/proj",
        )

        agent_ws = agency_workspace / "agents" / "eager"
        assert (agent_ws / "workspace").is_dir()
        assert (agent_ws / "control").is_dir()


# ---------------------------------------------------------------------------
# GatewayConfig: workspace field resolution
# ---------------------------------------------------------------------------


class TestGatewayConfigWorkspace:
    def test_resolve_workspace_returns_none_when_unset(self, tmp_path: Path):
        from thorn.gateway._config import GatewayConfig

        config = GatewayConfig()
        assert config.resolve_workspace(tmp_path) is None

    def test_resolve_workspace_absolute_path_returned_as_is(
        self, tmp_path: Path,
    ):
        from thorn.gateway._config import GatewayConfig

        target = tmp_path / "abs-ws"
        target.mkdir()
        config = GatewayConfig(workspace=str(target))
        # Use a different agency_home to prove the absolute path wins.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        assert config.resolve_workspace(elsewhere) == target.resolve()

    def test_resolve_workspace_relative_resolved_against_agency_home(
        self, tmp_path: Path,
    ):
        from thorn.gateway._config import GatewayConfig

        agency_home = tmp_path / "agency"
        agency_home.mkdir()
        (agency_home / "ws").mkdir()
        config = GatewayConfig(workspace="ws")
        resolved = config.resolve_workspace(agency_home)
        assert resolved == (agency_home / "ws").resolve()

    def test_load_gateway_config_round_trips_workspace(self, tmp_path: Path):
        """The ``workspace`` field survives a write/read cycle."""
        import json
        from thorn.gateway._config import load_gateway_config

        agency_home = tmp_path / "agency"
        agency_home.mkdir()
        (agency_home / "gateway.json").write_text(
            json.dumps({"workspace": "../shared-ws"}),
            encoding="utf-8",
        )
        config = load_gateway_config(agency_home)
        assert config.workspace == "../shared-ws"
        # Relative paths resolve against the agency home.
        expected = (agency_home / "../shared-ws").resolve()
        assert config.resolve_workspace(agency_home) == expected


# ---------------------------------------------------------------------------
# thorn serve: workspace resolution priority
# ---------------------------------------------------------------------------


class TestServeWorkspaceResolution:
    """``thorn serve`` picks workspace via CLI > config > error."""

    def _bootstrap(
        self,
        *,
        agency_home: Path,
        agency_workspace: Path,
    ) -> None:
        from thorn.gateway._bootstrap import bootstrap_coordinator

        bootstrap_coordinator(
            agency_home=agency_home,
            agency_workspace=agency_workspace,
            agent_id="srv-coord",
            project_name="proj",
            project_url="https://gitlab.com/group/proj",
        )

    def _strip_workspace_from_config(self, agency_home: Path) -> None:
        """Mutate gateway.json to remove the workspace field.

        Used to simulate a config that was hand-written without a
        workspace, so that the missing-workspace error branch can be
        exercised.
        """
        import json

        config_path = agency_home / "gateway.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data.pop("workspace", None)
        config_path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8",
        )

    def test_cli_workspace_overrides_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """``--workspace`` on ``serve`` wins over the config value.

        We capture the resolved workspace by patching ``AgencyPaths.for_gateway``
        instead of actually starting the gateway daemon.
        """
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        agency_home = tmp_path / "agency"
        config_workspace = tmp_path / "config-ws"
        cli_workspace = tmp_path / "cli-ws"
        cli_workspace.mkdir()

        self._bootstrap(
            agency_home=agency_home,
            agency_workspace=config_workspace,
        )

        captured: dict[str, Path] = {}

        def fake_for_gateway(*, agency_dir: Path, workspace_dir: Path):
            captured["workspace_dir"] = workspace_dir
            captured["agency_dir"] = agency_dir
            # Bail out before _build_runtime tries to do real work.
            raise SystemExit(0)

        monkeypatch.setattr(
            "thorn.runtime._paths.AgencyPaths.for_gateway", fake_for_gateway,
        )

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve",
            "--agency", str(agency_home),
            "--workspace", str(cli_workspace),
        ])
        # SystemExit(0) from our patch propagates as exit_code=0.
        assert result.exit_code == 0, result.output
        assert captured["workspace_dir"] == cli_workspace.resolve()
        assert captured["agency_dir"] == agency_home.resolve()

    def test_config_workspace_used_when_no_cli_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Without ``--workspace``, the config's ``workspace`` is used."""
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        agency_home = tmp_path / "agency"
        config_workspace = tmp_path / "config-ws"

        self._bootstrap(
            agency_home=agency_home,
            agency_workspace=config_workspace,
        )

        captured: dict[str, Path] = {}

        def fake_for_gateway(*, agency_dir: Path, workspace_dir: Path):
            captured["workspace_dir"] = workspace_dir
            raise SystemExit(0)

        monkeypatch.setattr(
            "thorn.runtime._paths.AgencyPaths.for_gateway", fake_for_gateway,
        )

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve",
            "--agency", str(agency_home),
        ])
        assert result.exit_code == 0, result.output
        assert captured["workspace_dir"] == config_workspace.resolve()

    def test_missing_both_workspaces_errors_clearly(
        self, tmp_path: Path,
    ):
        """No CLI ``--workspace`` and no config ``workspace`` → error."""
        from click.testing import CliRunner
        from thorn._cli import main as cli_main

        agency_home = tmp_path / "agency"
        self._bootstrap(
            agency_home=agency_home,
            agency_workspace=tmp_path / "ws",
        )
        self._strip_workspace_from_config(agency_home)

        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "serve", "--agency", str(agency_home),
        ])
        assert result.exit_code != 0
        assert "workspace" in result.output.lower()


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
# GitHub notifications source
# ---------------------------------------------------------------------------


def _make_notification_thread(
    *,
    thread_id: str = "100",
    repo_id: int = 42,
    repo_full_name: str = "owner/repo",
    subject_type: str = "Issue",
    subject_title: str = "Fix the bug",
    subject_url: str = "https://api.github.com/repos/owner/repo/issues/7",
    latest_comment_url: str = "https://api.github.com/repos/owner/repo/issues/comments/999",
    reason: str = "mention",
    unread: bool = True,
    updated_at: str = "2026-04-15T10:00:00Z",
    clone_url: str = "https://github.com/owner/repo.git",
    default_branch: str = "main",
    html_url: str = "https://github.com/owner/repo",
) -> dict[str, Any]:
    """Build a notification thread dict matching the GitHub API shape."""
    return {
        "id": thread_id,
        "repository": {
            "id": repo_id,
            "full_name": repo_full_name,
            "clone_url": clone_url,
            "default_branch": default_branch,
            "html_url": html_url,
        },
        "subject": {
            "title": subject_title,
            "url": subject_url,
            "latest_comment_url": latest_comment_url,
            "type": subject_type,
        },
        "reason": reason,
        "unread": unread,
        "updated_at": updated_at,
        "last_read_at": None,
        "url": f"https://api.github.com/notifications/threads/{thread_id}",
        "subscription_url": f"https://api.github.com/notifications/threads/{thread_id}/subscription",
    }


class TestExtractNoteableFromNotification:
    def test_issue(self):
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        result = _extract_noteable_from_notification(
            "Issue", "https://api.github.com/repos/owner/repo/issues/7",
        )
        assert result is not None
        assert result.kind == NoteableKind.ISSUE
        assert result.number == 7

    def test_pull_request(self):
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        result = _extract_noteable_from_notification(
            "PullRequest", "https://api.github.com/repos/owner/repo/pulls/3",
        )
        assert result is not None
        assert result.kind == NoteableKind.CHANGE_REQUEST
        assert result.number == 3

    def test_commit_returns_none(self):
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        result = _extract_noteable_from_notification(
            "Commit", "https://api.github.com/repos/owner/repo/commits/abc",
        )
        assert result is None

    def test_bad_url_returns_none(self):
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        result = _extract_noteable_from_notification("Issue", "")
        assert result is None


class TestFormatNotificationContent:
    def test_includes_key_fields(self):
        from thorn.gateway.sources._github import _format_notification_content

        content = _format_notification_content(
            repo_full_name="owner/repo",
            repo_id=42,
            clone_url="https://github.com/owner/repo.git",
            default_branch="main",
            html_url="https://github.com/owner/repo",
            subject_type="Issue",
            subject_title="Fix the bug",
            reason="mention",
            thread_id="100",
            updated_at="2026-04-15T10:00:00Z",
            comment_body="Please fix this ASAP",
        )
        assert "mention" in content
        assert "owner/repo" in content
        assert "Fix the bug" in content
        assert "100" in content
        assert "Please fix this ASAP" in content
        assert "Clone URL:" in content
        assert "marked read on your behalf" in content

    def test_empty_comment(self):
        from thorn.gateway.sources._github import _format_notification_content

        content = _format_notification_content(
            repo_full_name="owner/repo",
            repo_id=42,
            clone_url="",
            default_branch="main",
            html_url="",
            subject_type="PullRequest",
            subject_title="My PR",
            reason="review_requested",
            thread_id="200",
            updated_at="2026-04-15T11:00:00Z",
            comment_body="",
        )
        assert "Comment body:" not in content
        assert "review_requested" in content


class TestMakeIncomingEvent:
    def test_issue_notification(self):
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread()
        event = _make_incoming_event(
            thread=thread,
            comment_body="Hello",
            native_id_to_project_name={"owner/repo": "my-proj"},
        )
        assert event.source == "github"
        assert event.session_key == SessionKey("my-proj/issue/7")
        assert "Hello" in event.content
        assert event.metadata["notification_id"] == "100"
        assert event.metadata["reason"] == "mention"
        assert event.metadata["repo_full_name"] == "owner/repo"
        assert event.metadata["project_name"] == "my-proj"

    def test_pull_request_notification(self):
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(
            subject_type="PullRequest",
            subject_url="https://api.github.com/repos/owner/repo/pulls/3",
        )
        event = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={},
        )
        assert event.session_key == SessionKey("github/42/change-request/3")

    def test_unknown_subject_type_uses_fallback_key(self):
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(
            subject_type="Commit",
            subject_url="https://api.github.com/repos/owner/repo/commits/abc",
        )
        event = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={},
        )
        assert event.session_key == SessionKey("github/42/commit/100")


class TestGitHubNotificationsSourceConfig:
    def test_fields(self):
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        cfg = GitHubNotificationsSourceConfig(
            token="ghp-test",
            base_url="https://api.github.com",
            poll_interval=15,
            native_id_to_project_name={"owner/repo": "proj"},
        )
        assert cfg.token == "ghp-test"
        assert cfg.poll_interval == 15
        assert cfg.native_id_to_project_name == {"owner/repo": "proj"}

class TestGitHubNotificationsSource:
    def test_constructor(self):
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        cfg = GitHubNotificationsSourceConfig(token="ghp-test")
        source = GitHubNotificationsSource(cfg, service_name="test-source")
        assert source.name == "test-source"

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        cfg = GitHubNotificationsSourceConfig(token="ghp-test", poll_interval=5)
        source = GitHubNotificationsSource(cfg, service_name="test")

        events_received: list[IncomingEvent] = []

        mock_user_response = MagicMock()
        mock_user_response.status_code = 200
        mock_user_response.raise_for_status = MagicMock()
        mock_user_response.json.return_value = {
            "login": "bot-user", "name": "Bot", "html_url": "https://github.com/bot-user",
        }

        mock_notifications_response = MagicMock()
        mock_notifications_response.status_code = 200
        mock_notifications_response.raise_for_status = MagicMock()
        mock_notifications_response.json.return_value = []
        mock_notifications_response.headers = {}

        def mock_get(url: str, **kwargs: Any) -> Any:
            if url == "/user":
                return mock_user_response
            return mock_notifications_response

        with patch.object(source._http, "get", side_effect=mock_get):
            async def stop_after_one_poll(event: IncomingEvent) -> None:
                events_received.append(event)

            task = asyncio.create_task(source.start(stop_after_one_poll))
            await asyncio.sleep(0.1)
            await source.stop()
            await task

        assert events_received == []

    @pytest.mark.asyncio
    async def test_delivers_new_notifications(self):
        """The startup drain marks pre-existing unread notifications as
        read (without emitting events), and the next steady-state poll
        delivers any threads that surface as unread after that."""
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        cfg = GitHubNotificationsSourceConfig(
            token="ghp-test", poll_interval=5,
            native_id_to_project_name={"owner/repo": "my-proj"},
        )
        source = GitHubNotificationsSource(cfg, service_name="test")

        existing_unread_thread = _make_notification_thread(thread_id="1")
        new_thread = _make_notification_thread(thread_id="2", reason="assign")

        mock_user_resp = MagicMock()
        mock_user_resp.status_code = 200
        mock_user_resp.raise_for_status = MagicMock()
        mock_user_resp.json.return_value = {"login": "bot", "name": "", "html_url": ""}

        # The first /notifications GET (the drain) sees the
        # pre-existing unread thread; subsequent GETs see the new
        # thread (real GitHub stops returning the first one once we
        # PATCH it to read).
        list_call_count = 0

        def mock_get(url: str, **kwargs: Any) -> Any:
            nonlocal list_call_count
            if url == "/user":
                return mock_user_resp
            if "/issues/comments/" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"body": "comment text"}
                return resp

            list_call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.headers = {}
            if list_call_count == 1:
                resp.json.return_value = [existing_unread_thread]
            elif list_call_count == 2:
                resp.json.return_value = [new_thread]
            else:
                resp.json.return_value = []
            return resp

        patch_resp = MagicMock()
        patch_resp.status_code = 205
        patch_resp.raise_for_status = MagicMock()
        patch_calls: list[str] = []

        def mock_patch(url: str, **kwargs: Any) -> Any:
            patch_calls.append(url)
            return patch_resp

        events_received: list[IncomingEvent] = []

        async def on_event(event: IncomingEvent) -> None:
            events_received.append(event)
            await source.stop()

        with (
            patch.object(source._http, "get", side_effect=mock_get),
            patch.object(source._http, "patch", side_effect=mock_patch),
        ):
            await source.start(on_event)

        # The pre-existing unread thread was drained (PATCHed) and not
        # delivered as an event; the new thread was delivered and then
        # PATCHed at post time.
        assert len(events_received) == 1
        ev = events_received[0]
        assert ev.metadata["notification_id"] == "2"
        assert ev.metadata["reason"] == "assign"
        assert "comment text" in ev.content
        assert patch_calls == [
            "/notifications/threads/1",
            "/notifications/threads/2",
        ]

    @pytest.mark.asyncio
    async def test_304_not_modified_skipped(self):
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        cfg = GitHubNotificationsSourceConfig(token="ghp-test", poll_interval=5)
        source = GitHubNotificationsSource(cfg, service_name="test")

        resp_304 = MagicMock()
        resp_304.status_code = 304
        resp_304.headers = {}

        with patch.object(source._http, "get", return_value=resp_304):
            result = await asyncio.to_thread(source._fetch_new_notifications)

        assert result == []


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

    def test_project_name_based_noteable_key(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_github_event

        key = route_github_event(
            repo_id=42,
            noteable=Noteable(NoteableKind.ISSUE, 7),
            event_type="IssuesEvent",
            event_id="e1",
            project_name="my-proj",
        )
        assert key == SessionKey("my-proj/issue/7")

    def test_project_name_based_non_noteable_key(self):
        from thorn.gateway._routing import route_github_event

        key = route_github_event(
            repo_id=42,
            event_type="PushEvent",
            event_id="abc123",
            project_name="my-proj",
        )
        assert key == SessionKey("my-proj/pushevent/abc123")

    def test_empty_project_name_falls_back_to_legacy(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_github_event

        key = route_github_event(
            repo_id=42,
            noteable=Noteable(NoteableKind.ISSUE, 7),
            event_type="IssuesEvent",
            event_id="e1",
            project_name="",
        )
        assert key == SessionKey("github/42/issue/7")


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

    def test_project_name_based_issue_key(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        key = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.ISSUE, 5),
            project_name="lace",
        )
        assert key == SessionKey("lace/issue/5")

    def test_project_name_based_change_request_key(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        key = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.CHANGE_REQUEST, 2),
            project_name="lace",
        )
        assert key == SessionKey("lace/change-request/2")

    def test_empty_project_name_falls_back_to_legacy(self):
        from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo

        key = route_gitlab_todo(
            project_id=10,
            noteable=Noteable(NoteableKind.ISSUE, 5),
            project_name="",
        )
        assert key == SessionKey("gitlab/10/issue/5")


# ---------------------------------------------------------------------------
# Gateway: centralized workspace derivation
# ---------------------------------------------------------------------------


class TestGatewayWorkspaceRouting:
    """Verify that _handle_event derives the session workspace from
    ``AgencyPaths.session_workspace`` and pre-creates the directory on
    disk.  Under the Phase-A layout this is
    ``<workspace_root>/agents/<safe-agent-id>/workspace/<safe-session-key>/``.
    """

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

        expected_ws = runtime.paths.session_workspace(
            AgentID("default"), SessionKey("github/42/issue/7"),
        )
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
            agency_home=tmp_path / ".thorn",
            agency_workspace=tmp_path,
            agent_id="my-coord",
            project_name="proj",
            project_url="https://gitlab.com/group/proj",
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

        expected_ws = runtime.paths.session_workspace(
            AgentID("my-coord"), SessionKey("gitlab/10/issue/5"),
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

        expected_ws = runtime.paths.session_workspace(
            AgentID("default"), key,
        )
        assert loaded.workspace_root == expected_ws

    @pytest.mark.asyncio
    async def test_flat_session_key_workspace(self, tmp_path: Path):
        """A simple (non-slashed) session key produces a single-level
        directory under the agent's workspace mount."""
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

        expected_ws = runtime.paths.session_workspace(
            AgentID("default"), SessionKey("simple_key"),
        )
        assert expected_ws.is_dir()
        # Flat keys remain single-level under the workspace mount.
        assert expected_ws.parent == runtime.paths.agent_workspace_mount(
            AgentID("default"),
        )

        agent_ids = runtime.sessions.list_agent_ids()
        agent = runtime.sessions.load_agent(agent_ids[0])
        loaded = runtime.sessions.load_session(
            agent, SessionKey("simple_key"),
        )
        assert loaded.workspace_root == expected_ws
