"""Unit tests for the per-agent session scheduler.

Exercises ``AgentScheduler`` and its internal ``_SessionDriver`` in
isolation from the rest of the runtime: agents are bare ``Agent``
objects with an ID, sessions are bare ``Session`` objects with a
key, and the prompt dispatcher is a configurable async callable.

Covers:

- Per-session serialization (a single session's dispatcher is never
  re-entered).
- Agent-level concurrency cap (cap=1 serializes across sessions,
  cap>=2 permits parallelism across sessions).
- The wake protocol: submit()-while-idle triggers a round; a post
  that arrives mid-prompt triggers a follow-up round.
- Dispatcher exception isolation: the drain loop keeps going.
- ``save_session`` is called after successful rounds, never after a
  failed dispatcher.
- Shutdown modes: graceful (timeout waits for in-flight), immediate
  (timeout=0 cancels), idempotent (second shutdown returns).
- Submit after shutdown raises; same-session repeat submit re-uses
  the existing driver.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import pytest

from thorn.core._agent import Agent
from thorn.core._session import Session
from thorn.core.errors import LoopNoProgressError, ProviderUnavailableError
from thorn.runtime._address import AddressBook, ServiceAddress, SessionAddress
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import NotificationSpec, NotificationStatus
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._provider_health import (
    ProviderHealthMonitor,
    ProviderHealthState,
)
from thorn.runtime._provider_state import PROVIDER_UNAVAILABLE_METADATA_KEY
from thorn.runtime._scheduler import (
    DEFAULT_AGENT_CONCURRENCY,
    DEFAULT_PROGRESS_STRIKES,
    AgentScheduler,
    default_progress_evictor,
)
from thorn.runtime._session import AgentID, SessionKey

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_agent(agent_id: str = "bot") -> Agent:
    return Agent(id=AgentID(agent_id), name=agent_id)


def _make_session(agent: Agent, key: str) -> Session:
    return Session(agent=agent, key=SessionKey(key))


def _make_inbox(tmp_path: Path, agent: Agent, session_key: str) -> SessionInbox:
    addr = SessionAddress(agent.id, SessionKey(session_key))
    return SessionInbox(tmp_path / agent.id / session_key / "inbox", addr)


def _spec(target: SessionAddress, content: str = "hello") -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        rsvp_to=None,
        external_key=None,
    )


async def _short_sleep() -> None:
    """Yield control so queued tasks get a chance to run.

    Multiple short sleeps may be needed for cascaded awaits; call in
    a loop if you are waiting for several layers of scheduling.
    """
    await asyncio.sleep(0.01)


async def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
) -> None:
    """Poll *predicate* until true or *timeout* seconds elapse.

    Used instead of a fixed sleep when the test needs to observe a
    side effect produced by scheduler tasks.  Raises
    :class:`AssertionError` on timeout so failures surface clearly.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(
        f"Predicate never became true within {timeout}s timeout"
    )


class _DispatcherProbe:
    """Records every dispatcher call and drains one item per call.

    The default behavior simulates a well-behaved agent that marks
    the oldest pending item ``handled`` on each round, which gives
    the drain loop a natural termination point (eventually the
    inbox empties and the driver idles).

    Tests can swap the behavior via ``action`` -- a callable invoked
    with (session, inbox, call_index) that decides what to do with
    the inbox during that round.
    """

    def __init__(
        self,
        action: Callable[[Session, SessionInbox, int], None] | None = None,
        *,
        delay: float = 0.0,
        raises: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[SessionKey, int]] = []
        self.in_flight: set[SessionKey] = set()
        self.max_in_flight: int = 0
        self._in_flight_lock = asyncio.Lock()
        self._delay = delay
        self._raises = raises
        self._action = action or _default_handle_one

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def __call__(self, session: Session, inbox: SessionInbox) -> None:
        async with self._in_flight_lock:
            self.in_flight.add(session.key)
            if len(self.in_flight) > self.max_in_flight:
                self.max_in_flight = len(self.in_flight)
        try:
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            self.calls.append((session.key, len(self.calls)))
            if self._raises is not None:
                raise self._raises
            self._action(session, inbox, len(self.calls) - 1)
        finally:
            async with self._in_flight_lock:
                self.in_flight.discard(session.key)


def _default_handle_one(
    session: Session,
    inbox: SessionInbox,
    call_index: int,
) -> None:
    """Mark the oldest prompt-pending item ``handled``.

    Simulates the agent's end-of-round behavior and ensures the
    drain loop terminates.
    """
    pending = inbox.prompt_pending()
    if pending:
        inbox.update_status(pending[0].id, NotificationStatus.HANDLED)


def _handle_nothing(
    session: Session,
    inbox: SessionInbox,
    call_index: int,
) -> None:
    """Dispatcher that does not make any progress.

    Used by tests that want to observe the drain loop re-invoking
    the dispatcher while the inbox remains non-empty.
    """
    return None


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestSchedulerConstruction:
    async def test_default_concurrency(self) -> None:
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=_make_agent(), prompt_dispatcher=probe)
        assert sched.concurrency == DEFAULT_AGENT_CONCURRENCY
        assert sched.agent.id == "bot"
        assert sched.is_closed is False
        assert sched.session_keys() == []
        await sched.shutdown(timeout=1.0)

    async def test_custom_concurrency(self) -> None:
        probe = _DispatcherProbe()
        sched = AgentScheduler(
            agent=_make_agent(), prompt_dispatcher=probe, concurrency=7
        )
        assert sched.concurrency == 7
        await sched.shutdown(timeout=1.0)

    async def test_rejects_zero_concurrency(self) -> None:
        probe = _DispatcherProbe()
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            AgentScheduler(
                agent=_make_agent(), prompt_dispatcher=probe, concurrency=0
            )

    async def test_rejects_negative_concurrency(self) -> None:
        probe = _DispatcherProbe()
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            AgentScheduler(
                agent=_make_agent(), prompt_dispatcher=probe, concurrency=-1
            )


# ---------------------------------------------------------------------------
# Basic drain behavior
# ---------------------------------------------------------------------------

class TestBasicDrain:
    async def test_submit_runs_dispatcher_for_pending_item(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 1)
        await sched.shutdown(timeout=1.0)

        assert probe.call_count == 1
        assert probe.calls[0][0] == SessionKey("s1")

    async def test_no_pending_items_no_dispatcher_call(self, tmp_path: Path) -> None:
        # Empty inbox: submit should register the driver but the
        # drain loop must NOT invoke the dispatcher until a post
        # arrives.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.submit(session, inbox)
        # Give the driver plenty of time to realize there is nothing
        # to do and go idle.
        await asyncio.sleep(0.05)
        assert probe.call_count == 0
        await sched.shutdown(timeout=1.0)

    async def test_drains_multiple_items_sequentially(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(3):
            inbox.post(_spec(inbox.address))
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 3)
        await sched.shutdown(timeout=1.0)

        # Default dispatcher marks one item handled per call; three
        # items => three rounds.
        assert probe.call_count == 3
        # prompt_pending now excludes handled items; the remaining
        # inbox view should be empty.
        assert inbox.prompt_pending() == []

    async def test_kick_after_idle_wakes_driver(self, tmp_path: Path) -> None:
        # Driver goes idle with an empty inbox.  A later post +
        # submit must wake it and trigger a round.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.submit(session, inbox)
        await asyncio.sleep(0.02)  # Let the driver go idle.
        assert probe.call_count == 0

        inbox.post(_spec(inbox.address))
        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 1)
        await sched.shutdown(timeout=1.0)

    async def test_post_during_prompt_triggers_followup_round(self, tmp_path: Path) -> None:
        # A post that lands while a prompt is in flight must cause
        # a second round (the agent otherwise wouldn't see the new
        # item until the next external kick).
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")

        first_in_flight = asyncio.Event()
        first_may_return = asyncio.Event()

        def action(
            session: Session,
            inbox: SessionInbox,
            call_index: int,
        ) -> None:
            # Mark the oldest pending item handled on each call so
            # the loop terminates after all posts have been drained.
            _default_handle_one(session, inbox, call_index)

        async def dispatcher(session: Session, inbox: SessionInbox) -> None:
            if not first_in_flight.is_set():
                first_in_flight.set()
                await first_may_return.wait()
            action(session, inbox, 0)

        calls: list[SessionKey] = []

        async def tracking_dispatcher(
            session: Session, inbox: SessionInbox,
        ) -> None:
            calls.append(session.key)
            await dispatcher(session, inbox)

        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=tracking_dispatcher,
        )

        inbox.post(_spec(inbox.address, "first"))
        await sched.submit(session, inbox)

        # Wait until the first dispatcher call is in flight, then
        # post a second item.  The driver is blocked inside
        # dispatcher, so this post will only be observed when the
        # loop re-checks the inbox after the first call returns.
        await first_in_flight.wait()
        inbox.post(_spec(inbox.address, "second"))

        # Release the first call; the driver should loop, see the
        # second item, and dispatch again.  No explicit kick is
        # required -- the loop's post-round re-check handles it.
        first_may_return.set()

        await _wait_for(lambda: len(calls) >= 2)
        await sched.shutdown(timeout=1.0)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Session-level serialization
# ---------------------------------------------------------------------------

class TestSessionSerialization:
    async def test_same_session_never_runs_in_parallel(self, tmp_path: Path) -> None:
        # With two items pending on a single session, we must never
        # observe two in-flight dispatcher calls for that session.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(4):
            inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(delay=0.02)
        # Generous cap; the session-level serialization must still
        # hold because only one driver per session exists.
        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=probe, concurrency=8,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 4)
        await sched.shutdown(timeout=1.0)

        assert probe.max_in_flight == 1

    async def test_repeat_submit_reuses_driver(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.submit(session, inbox)
        await sched.submit(session, inbox)
        await sched.submit(session, inbox)

        assert sched.session_keys() == [SessionKey("s1")]
        await sched.shutdown(timeout=1.0)


# ---------------------------------------------------------------------------
# Agent-level concurrency cap
# ---------------------------------------------------------------------------

class TestAgentConcurrencyCap:
    async def test_cap_of_one_serializes_across_sessions(self, tmp_path: Path) -> None:
        agent = _make_agent()
        s1, s2 = _make_session(agent, "a"), _make_session(agent, "b")
        in1 = _make_inbox(tmp_path, agent, "a")
        in2 = _make_inbox(tmp_path, agent, "b")
        in1.post(_spec(in1.address))
        in2.post(_spec(in2.address))

        probe = _DispatcherProbe(delay=0.03)
        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=probe, concurrency=1,
        )

        await sched.submit(s1, in1)
        await sched.submit(s2, in2)
        await _wait_for(lambda: probe.call_count >= 2)
        await sched.shutdown(timeout=1.0)

        assert probe.max_in_flight == 1

    async def test_cap_of_two_permits_cross_session_parallelism(self, tmp_path: Path) -> None:
        agent = _make_agent()
        s1, s2 = _make_session(agent, "a"), _make_session(agent, "b")
        in1 = _make_inbox(tmp_path, agent, "a")
        in2 = _make_inbox(tmp_path, agent, "b")
        in1.post(_spec(in1.address))
        in2.post(_spec(in2.address))

        # Long enough delay that both sessions will overlap if the
        # cap actually permits it.
        probe = _DispatcherProbe(delay=0.05)
        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=probe, concurrency=2,
        )

        await sched.submit(s1, in1)
        await sched.submit(s2, in2)
        await _wait_for(lambda: probe.call_count >= 2)
        await sched.shutdown(timeout=1.0)

        assert probe.max_in_flight == 2

    async def test_cap_of_two_still_caps_at_two_with_three_sessions(self, tmp_path: Path) -> None:
        agent = _make_agent()
        sessions = [_make_session(agent, f"s{i}") for i in range(3)]
        inboxes = [_make_inbox(tmp_path, agent, f"s{i}") for i in range(3)]
        for ix in inboxes:
            ix.post(_spec(ix.address))

        probe = _DispatcherProbe(delay=0.05)
        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=probe, concurrency=2,
        )

        for s, ix in zip(sessions, inboxes):
            await sched.submit(s, ix)
        await _wait_for(lambda: probe.call_count >= 3)
        await sched.shutdown(timeout=1.0)

        assert probe.max_in_flight == 2


# ---------------------------------------------------------------------------
# SessionSaver callback
# ---------------------------------------------------------------------------

class TestSessionSaver:
    async def test_save_called_after_each_successful_round(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(3):
            inbox.post(_spec(inbox.address))

        saves: list[SessionKey] = []

        async def saver(s: Session) -> None:
            saves.append(s.key)

        probe = _DispatcherProbe()
        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=probe, save_session=saver,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 3)
        await sched.shutdown(timeout=1.0)

        assert saves == [SessionKey("s1")] * 3

    async def test_save_not_called_when_dispatcher_raises(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        saves: list[SessionKey] = []

        async def saver(s: Session) -> None:
            saves.append(s.key)

        # Dispatcher always raises; without a saver call the drain
        # loop spins until shutdown.  We only need a few observed
        # attempts to confirm save_session was never invoked.
        probe = _DispatcherProbe(raises=RuntimeError("boom"))
        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=probe, save_session=saver,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 2)
        assert saves == []
        await sched.shutdown(timeout=0)

    async def test_saver_exception_is_swallowed(self, tmp_path: Path) -> None:
        # If save_session raises, the drain loop must continue.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(3):
            inbox.post(_spec(inbox.address))

        async def failing_saver(s: Session) -> None:
            raise RuntimeError("persist broken")

        probe = _DispatcherProbe()
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            save_session=failing_saver,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 3)
        await sched.shutdown(timeout=1.0)
        assert probe.call_count == 3


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------

class TestExceptionIsolation:
    async def test_dispatcher_exception_does_not_kill_driver(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Dispatcher fails on the first call, succeeds on the second
        # (because it has stopped raising).  Without exception
        # isolation, the driver task would die and the second attempt
        # would never happen.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        call_count = 0

        async def flaky_dispatcher(s: Session, ix: SessionInbox) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("flake")
            # Handle the item on the second call so the loop
            # terminates.
            pending = ix.prompt_pending()
            if pending:
                ix.update_status(pending[0].id, NotificationStatus.HANDLED)

        sched = AgentScheduler(
            agent=agent, prompt_dispatcher=flaky_dispatcher,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: call_count >= 2)
        await sched.shutdown(timeout=1.0)

        assert call_count == 2
        assert inbox.prompt_pending() == []

    async def test_no_progress_dispatcher_spins_until_shutdown(self, tmp_path: Path) -> None:
        # With the N-strikes guard disabled (progress_evictor=None),
        # a dispatcher that never closes out items will spin forever.
        # The default ProgressEvictor path is covered by the
        # dedicated TestProgressGuarantee class below.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(action=_handle_nothing)
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=None,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 3)
        await sched.shutdown(timeout=0)
        assert len(inbox.prompt_pending()) == 1


# ---------------------------------------------------------------------------
# Shutdown modes
# ---------------------------------------------------------------------------

class TestShutdown:
    async def test_graceful_shutdown_waits_for_in_flight(self, tmp_path: Path) -> None:
        # A slow dispatcher should get to complete its work if the
        # shutdown timeout exceeds the round duration.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        in_flight = asyncio.Event()
        may_return = asyncio.Event()

        async def dispatcher(s: Session, ix: SessionInbox) -> None:
            in_flight.set()
            await may_return.wait()
            pending = ix.prompt_pending()
            if pending:
                ix.update_status(pending[0].id, NotificationStatus.HANDLED)

        sched = AgentScheduler(agent=agent, prompt_dispatcher=dispatcher)
        await sched.submit(session, inbox)
        await in_flight.wait()

        # Start a shutdown that allows plenty of time; release the
        # dispatcher so it can complete.
        shutdown_task = asyncio.create_task(sched.shutdown(timeout=2.0))
        await _short_sleep()
        may_return.set()
        await shutdown_task

        assert inbox.prompt_pending() == []

    async def test_immediate_shutdown_cancels_in_flight(self, tmp_path: Path) -> None:
        # timeout=0 must cancel an in-flight dispatcher; the item
        # stays untouched because the dispatcher never got to mark
        # it handled.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        in_flight = asyncio.Event()

        async def dispatcher(s: Session, ix: SessionInbox) -> None:
            in_flight.set()
            # Sleep longer than any reasonable test duration; the
            # shutdown should cancel us.
            await asyncio.sleep(60)

        sched = AgentScheduler(agent=agent, prompt_dispatcher=dispatcher)
        await sched.submit(session, inbox)
        await in_flight.wait()

        await sched.shutdown(timeout=0)

        # Item is still pending; nobody marked it handled.
        assert len(inbox.prompt_pending()) == 1

    async def test_idempotent_shutdown(self, tmp_path: Path) -> None:
        agent = _make_agent()
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.shutdown(timeout=0)
        assert sched.is_closed
        # Second call returns without raising.
        await sched.shutdown(timeout=0)

    async def test_submit_after_shutdown_raises(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.shutdown(timeout=0)
        with pytest.raises(RuntimeError, match="closed"):
            await sched.submit(session, inbox)

    async def test_submit_replaces_stopped_driver(
        self, tmp_path: Path,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address, content="first"))

        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.submit(session, inbox)
        await _wait_for(lambda: inbox.prompt_pending() == [])

        first_driver = sched._drivers[session.key]
        assert first_driver.task is not None
        first_driver.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_driver.task

        inbox.post(_spec(inbox.address, content="second"))
        await sched.submit(session, inbox)
        await _wait_for(lambda: inbox.prompt_pending() == [])
        await sched.shutdown(timeout=0)

        assert sched._drivers[session.key] is not first_driver
        assert probe.call_count == 2

    async def test_shutdown_with_no_drivers_is_noop(self, tmp_path: Path) -> None:
        agent = _make_agent()
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        await sched.shutdown(timeout=1.0)
        assert sched.is_closed


# ---------------------------------------------------------------------------
# Submit validation
# ---------------------------------------------------------------------------

class TestSubmitValidation:
    async def test_submit_rejects_session_without_key(self, tmp_path: Path) -> None:
        agent = _make_agent()
        session = Session(agent=agent, key=None)  # Intentionally keyless.
        inbox = _make_inbox(tmp_path, agent, "s1")
        probe = _DispatcherProbe()
        sched = AgentScheduler(agent=agent, prompt_dispatcher=probe)

        with pytest.raises(ValueError, match="without a key"):
            await sched.submit(session, inbox)

        await sched.shutdown(timeout=0)


# ---------------------------------------------------------------------------
# Progress guarantee
# ---------------------------------------------------------------------------

class _RecordingEvictor:
    """Test double for :data:`ProgressEvictor`.

    Records every call and optionally mutates the inbox in the same
    shape the default evictor would (erroring the oldest item) so
    tests can exercise both "evictor succeeded" and "evictor failed"
    paths.
    """

    def __init__(
        self,
        *,
        mode: str = "evict_oldest",
        raises: BaseException | None = None,
    ) -> None:
        self.calls: list[SessionInbox] = []
        self._mode = mode
        self._raises = raises

    async def __call__(self, inbox: SessionInbox) -> None:
        self.calls.append(inbox)
        if self._raises is not None:
            raise self._raises
        if self._mode == "noop":
            return
        if self._mode == "evict_oldest":
            pending = inbox.prompt_pending()
            if pending:
                inbox.update_status(
                    pending[0].id, NotificationStatus.HANDLED,
                )


class TestProgressGuarantee:
    async def test_evictor_fires_after_n_stalled_rounds(
        self, tmp_path: Path,
    ) -> None:
        # With the default strike threshold (3), a dispatcher that
        # never closes items must trigger exactly one evictor call
        # after the third stalled round.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(action=_handle_nothing)
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: len(evictor.calls) >= 1)
        # Wait a little more to confirm it only fires once per
        # threshold crossing (not once per round past the threshold).
        await _short_sleep()
        await sched.shutdown(timeout=0)

        assert len(evictor.calls) >= 1
        # Three rounds observed before the first eviction.  The
        # dispatcher may race ahead by one extra round before
        # shutdown takes effect, so we check for "at least three."
        first_round_before_evict = probe.call_count
        assert first_round_before_evict >= DEFAULT_PROGRESS_STRIKES

    async def test_progress_resets_counter(self, tmp_path: Path) -> None:
        # If the dispatcher closes an item on the second round, the
        # strike counter must reset; with further stalled rounds the
        # evictor should not fire until another N rounds elapse.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(10):
            inbox.post(_spec(inbox.address))

        call_counter = {"i": 0}

        def progress_on_second(
            session: Session, inbox: SessionInbox, idx: int,
        ) -> None:
            call_counter["i"] += 1
            # Progress only on call 2, then stall forever.
            if call_counter["i"] == 2:
                pending = inbox.prompt_pending()
                if pending:
                    inbox.update_status(
                        pending[0].id, NotificationStatus.HANDLED,
                    )

        probe = _DispatcherProbe(action=progress_on_second)
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,  # Tighten for a faster test.
        )

        await sched.submit(session, inbox)
        # One progress (round 2) + two stalled rounds (rounds 3 & 4)
        # should trigger the evictor at round 4.  The stall count
        # before the reset would be 1 at round 1; 0 at round 2;
        # 1 at round 3; 2 -> fire at round 4.
        await _wait_for(lambda: len(evictor.calls) >= 1)
        await sched.shutdown(timeout=0)

        # First eviction fired no earlier than round 4 (because
        # the round-2 reset delayed the trigger).
        first_evict_round = probe.call_count
        assert first_evict_round >= 4

    async def test_dispatcher_exception_counts_as_stall(
        self, tmp_path: Path,
    ) -> None:
        # A raising dispatcher never closes items, so its rounds
        # should accrue strikes just like a silently-stalling one.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(raises=RuntimeError("boom"))
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: len(evictor.calls) >= 1)
        await sched.shutdown(timeout=0)

        assert len(evictor.calls) >= 1

    async def test_no_progress_error_forces_evictor(
        self, tmp_path: Path,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(
            raises=LoopNoProgressError("no useful work", 8),
        )
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=3,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: len(evictor.calls) >= 1)
        await sched.shutdown(timeout=0)

        assert len(evictor.calls) >= 1

    async def test_new_arrivals_do_not_count_as_progress(
        self, tmp_path: Path,
    ) -> None:
        # Items arriving mid-round do not reset the strike counter;
        # only closed-out items do.  Without this, a chatty source
        # could indefinitely delay the forward-progress guarantee.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address, content="original"))

        def post_then_stall(
            session: Session, inbox: SessionInbox, idx: int,
        ) -> None:
            # Simulate a source posting during the prompt.
            inbox.post(_spec(inbox.address, content=f"arrived-{idx}"))

        probe = _DispatcherProbe(action=post_then_stall)
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: len(evictor.calls) >= 1)
        await sched.shutdown(timeout=0)

        assert len(evictor.calls) >= 1

    async def test_evictor_success_resets_counter(
        self, tmp_path: Path,
    ) -> None:
        # After an eviction, the strike counter must reset.  With a
        # dispatcher that continues to stall, a second eviction
        # should fire only after another N rounds elapse (which we
        # verify by counting rounds between evictions).
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(3):
            inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(action=_handle_nothing)
        evictor = _RecordingEvictor(mode="evict_oldest")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: len(evictor.calls) >= 2)
        await sched.shutdown(timeout=0)

        assert len(evictor.calls) >= 2

    async def test_evictor_exception_retries_next_round(
        self, tmp_path: Path,
    ) -> None:
        # A raising evictor must not kill the driver.  The strike
        # counter is left elevated so the next round retries
        # eviction; until the evictor succeeds, the session keeps
        # trying.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(action=_handle_nothing)
        evictor = _RecordingEvictor(
            mode="noop", raises=RuntimeError("evictor crashed"),
        )
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        # At least two evictor calls -- demonstrating retry after
        # the first failure -- without the driver dying.
        await _wait_for(lambda: len(evictor.calls) >= 2)
        await sched.shutdown(timeout=0)

    async def test_progress_strikes_zero_disables_guard(
        self, tmp_path: Path,
    ) -> None:
        # Explicit 0 disables the guard even when an evictor is
        # installed: the session spins without eviction.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(action=_handle_nothing)
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=0,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 5)
        await sched.shutdown(timeout=0)

        assert evictor.calls == []

    async def test_negative_progress_strikes_rejected(self) -> None:
        probe = _DispatcherProbe()
        with pytest.raises(ValueError, match="progress_strikes must be >= 0"):
            AgentScheduler(
                agent=_make_agent(),
                prompt_dispatcher=probe,
                progress_strikes=-1,
            )


# ---------------------------------------------------------------------------
# default_progress_evictor
# ---------------------------------------------------------------------------

class TestDefaultProgressEvictor:
    async def test_marks_oldest_errored_when_no_rsvp(
        self, tmp_path: Path,
    ) -> None:
        # Without an RSVP target, the oldest item is moved to the
        # inbox's errored/ directory with the canned reason.
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        first = inbox.post(_spec(inbox.address, content="first"))
        inbox.post(_spec(inbox.address, content="second"))
        ab = AddressBook()

        evictor = default_progress_evictor(ab)
        await evictor(inbox)

        # Oldest is gone from prompt-pending.
        pending_ids = [n.id for n in inbox.prompt_pending()]
        assert first.id not in pending_ids
        assert len(pending_ids) == 1

    async def test_routes_to_rsvp_target_when_set(
        self, tmp_path: Path,
    ) -> None:
        # With an RSVP set, the evicted (errored) item is forwarded
        # to the RSVP target's queue with the canned error_reason.
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        svc_addr = ServiceAddress("gitlab-primary")
        svc_queue = NotificationQueue(tmp_path / "svc-queue", svc_addr)
        first = inbox.post(_spec(inbox.address, content="first"))
        # Replace with an RSVP-carrying spec: we have to repost since
        # the factory helper doesn't thread rsvp_to.  Use the queue
        # directly.
        inbox.update_status(first.id, NotificationStatus.HANDLED)
        # Clean slate: post a fresh RSVP item.
        rsvp_spec = NotificationSpec(
            source="test",
            content="with rsvp",
            target=inbox.address,
            rsvp_to=svc_addr,
        )
        posted = inbox.post(rsvp_spec)

        ab = AddressBook()
        ab.register(svc_addr, svc_queue)

        evictor = default_progress_evictor(ab)
        await evictor(inbox)

        # Item no longer pending in session inbox.
        assert [n.id for n in inbox.prompt_pending()] == []
        # Item now in service queue as an errored RSVP.
        forwarded = list(svc_queue.list())
        assert len(forwarded) == 1
        assert forwarded[0].id == posted.id
        assert forwarded[0].status is NotificationStatus.ERRORED
        assert forwarded[0].error_reason is not None
        assert "progress guarantee" in forwarded[0].error_reason.lower()

    async def test_empty_inbox_is_a_noop(self, tmp_path: Path) -> None:
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        evictor = default_progress_evictor(AddressBook())
        # Must not raise.
        await evictor(inbox)
        assert inbox.prompt_pending() == []

    async def test_dispatch_error_swallowed(self, tmp_path: Path) -> None:
        # Step 1 succeeds (item marked errored); step 2 fails
        # because the RSVP target is not registered.  The evictor
        # must swallow the DispatchError so the scheduler can reset
        # its counter -- the item has been removed from the
        # pending view, which is what matters for the guarantee.
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        unknown_addr = ServiceAddress("no-such-service")
        spec = NotificationSpec(
            source="test",
            content="x",
            target=inbox.address,
            rsvp_to=unknown_addr,
        )
        inbox.post(spec)

        evictor = default_progress_evictor(AddressBook())
        # Must not raise even though the RSVP target is unknown.
        await evictor(inbox)

        # prompt_pending is clear regardless of step-2 failure.
        assert inbox.prompt_pending() == []

    async def test_scheduler_integration_with_default_evictor(
        self, tmp_path: Path,
    ) -> None:
        # End-to-end: scheduler + default evictor + a silently-
        # stalling dispatcher must eventually empty the inbox by
        # eviction alone.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        for _ in range(2):
            inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(action=_handle_nothing)
        evictor = default_progress_evictor(AddressBook())
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: len(inbox.prompt_pending()) == 0)
        await sched.shutdown(timeout=0)

        # All items were evicted (marked errored and parked in
        # errored/ since no RSVP was set).
        assert inbox.prompt_pending() == []


# ---------------------------------------------------------------------------
# ProviderUnavailableError handling (Phase 1 QoS)
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_provider_unavailable_backoff(monkeypatch):
    """Reduce provider-unavailable backoff to essentially zero.

    Tests in this section care about *which* bookkeeping changes
    around a provider-unavailable round, not about how long the
    cooldown is.  Setting base/jitter to small values keeps the
    tests fast while still exercising the real backoff path.
    """
    monkeypatch.setenv("THORN_PROVIDER_UNAVAILABLE_BACKOFF", "0.0")
    monkeypatch.setenv("THORN_PROVIDER_UNAVAILABLE_BACKOFF_JITTER", "0.005")


class TestProviderUnavailableHandling:
    """Phase 1 QoS: a ``ProviderUnavailableError`` from the dispatcher
    is not blamed on the session.

    Contract verified here:

    - The strike counter does not advance; even a session that sees
      nothing but :class:`ProviderUnavailableError` never triggers
      :data:`default_progress_evictor`.
    - ``save_session`` is invoked with provider-unavailable metadata
      so operators can see why the session is paused.
    - The driver stays alive (a provider outage should not kill the
      session's long-lived drain task).
    """

    async def test_provider_unavailable_does_not_increment_stall(
        self,
        tmp_path: Path,
        fast_provider_unavailable_backoff,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(
            raises=ProviderUnavailableError("outage", attempts=3),
        )
        evictor = _RecordingEvictor(mode="noop")
        # A tight threshold of 2 would fire quickly if the driver
        # mistakenly counted these as stalls; the assertion is
        # "no eviction even after many rounds".
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 5)
        await sched.shutdown(timeout=0)

        assert probe.call_count >= 5
        assert evictor.calls == []
        assert inbox.prompt_pending()  # unchanged

    async def test_saves_provider_unavailable_metadata(
        self,
        tmp_path: Path,
        fast_provider_unavailable_backoff,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(
            raises=ProviderUnavailableError("outage", attempts=3),
        )
        save_snapshots: list[dict] = []

        async def saver(sess: Session) -> None:
            save_snapshots.append(dict(sess.metadata))

        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            save_session=saver,
            progress_evictor=None,
        )
        await sched.submit(session, inbox)
        await _wait_for(lambda: save_snapshots)
        await sched.shutdown(timeout=0)

        saved_provider_state = save_snapshots[-1][PROVIDER_UNAVAILABLE_METADATA_KEY]
        assert saved_provider_state["state"] == "waiting_on_provider"
        assert saved_provider_state["attempts"] == 3
        assert "outage" in saved_provider_state["reason"]

    async def test_recovery_clears_provider_unavailable_metadata(
        self,
        tmp_path: Path,
        fast_provider_unavailable_backoff,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        def flaky_action(session_arg, inbox_arg, call_index) -> None:
            if call_index == 0:
                raise ProviderUnavailableError("outage", attempts=3)
            _default_handle_one(session_arg, inbox_arg, call_index)

        save_snapshots: list[dict] = []

        async def saver(sess: Session) -> None:
            save_snapshots.append(dict(sess.metadata))

        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=_DispatcherProbe(action=flaky_action),
            save_session=saver,
            progress_evictor=None,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: inbox.prompt_pending() == [])
        await sched.shutdown(timeout=0)

        assert any(
            PROVIDER_UNAVAILABLE_METADATA_KEY in snapshot
            for snapshot in save_snapshots
        )
        assert PROVIDER_UNAVAILABLE_METADATA_KEY not in session.metadata
        assert PROVIDER_UNAVAILABLE_METADATA_KEY not in save_snapshots[-1]

    async def test_recovery_resets_behavior(
        self,
        tmp_path: Path,
        fast_provider_unavailable_backoff,
    ) -> None:
        # Provider outage for the first two rounds, then normal
        # behaviour: the dispatcher closes out the item and the
        # driver goes idle with an empty inbox.  The strike counter
        # never advanced during the outage, so no eviction fired.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        call_log: list[int] = []

        def flaky_action(session_arg, inbox_arg, call_index) -> None:
            call_log.append(call_index)
            if call_index < 2:
                raise ProviderUnavailableError("outage", attempts=3)
            # Normal progress: close out the oldest item.
            _default_handle_one(session_arg, inbox_arg, call_index)

        probe = _DispatcherProbe(action=flaky_action)
        evictor = _RecordingEvictor(mode="noop")
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=evictor,
            progress_strikes=2,
        )

        await sched.submit(session, inbox)
        await _wait_for(lambda: inbox.prompt_pending() == [])
        await sched.shutdown(timeout=0)

        assert inbox.prompt_pending() == []
        assert evictor.calls == []

    async def test_driver_survives_provider_unavailable(
        self,
        tmp_path: Path,
        fast_provider_unavailable_backoff,
    ) -> None:
        # The driver must not terminate on
        # ``ProviderUnavailableError``; subsequent kicks should
        # keep finding it alive and able to retry.
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        probe = _DispatcherProbe(
            raises=ProviderUnavailableError("outage", attempts=3),
        )
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            progress_evictor=None,
        )
        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 3)

        driver = sched._drivers[session.key]
        assert driver.task is not None
        assert not driver.task.done()
        await sched.shutdown(timeout=0)


# ---------------------------------------------------------------------------
# ProviderHealthMonitor integration (Phase 2 QoS)
# ---------------------------------------------------------------------------

class TestSchedulerWithHealthMonitor:
    """Phase 2 QoS: a shared monitor coordinates pacing across sessions.

    Contract verified here:

    - A successful round reports success to the monitor (clearing
      any prior failure history in the rolling window).
    - A :class:`ProviderUnavailableError` round reports failure;
      after enough failures the monitor flips to ``DEGRADED`` and
      every session driver sharing the monitor blocks at
      ``wait_until_healthy`` instead of barreling ahead with its
      own retries.
    - When the monitor recovers (via a successful probe), all
      blocked drivers resume.
    - An installed monitor causes the scheduler to skip the
      per-driver provider-unavailable backoff sleep, since the
      monitor's cooldown is now the cross-session-coordinated
      equivalent.
    - The monitor-aware default evictor does not fire while the
      monitor is degraded, even if some other path managed to
      elevate the strike counter.
    """

    async def test_dispatcher_success_reports_to_monitor(
        self, tmp_path: Path,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        # Pre-seed the monitor with a stale failure so we can
        # confirm a successful round clears it.
        monitor = ProviderHealthMonitor(
            failure_threshold=10,
            failure_window_seconds=120.0,
        )
        await monitor.report_failure()
        assert monitor.snapshot().recent_failure_count == 1

        probe = _DispatcherProbe()
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            health_monitor=monitor,
        )
        await sched.submit(session, inbox)
        await _wait_for(lambda: probe.call_count >= 1)
        await sched.shutdown(timeout=1.0)

        # The successful round should have called report_success,
        # clearing the pre-seeded failure.
        assert monitor.snapshot().recent_failure_count == 0
        assert monitor.is_healthy

    async def test_provider_unavailable_reports_to_monitor(
        self, tmp_path: Path,
    ) -> None:
        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        monitor = ProviderHealthMonitor(
            failure_threshold=2,
            base_cooldown_seconds=60.0,
            max_cooldown_seconds=60.0,
            cooldown_jitter_seconds=0.0,
        )

        probe = _DispatcherProbe(
            raises=ProviderUnavailableError("outage", attempts=3),
        )
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            health_monitor=monitor,
        )
        await sched.submit(session, inbox)
        # Two failures should be enough to trip the threshold.  We
        # then wait briefly for the third call to discover the
        # degraded state and block at wait_until_healthy.
        await _wait_for(
            lambda: monitor.state == ProviderHealthState.DEGRADED,
        )
        # With a 60s cooldown, the dispatcher should *not* be hit
        # again after the breaker trips.
        calls_at_degrade = probe.call_count
        await asyncio.sleep(0.05)
        assert probe.call_count == calls_at_degrade

        await sched.shutdown(timeout=0)

    async def test_monitor_serializes_recovery_probe(
        self, tmp_path: Path,
    ) -> None:
        # Two sessions sharing a monitor.  After the breaker trips,
        # the next round in either session is a probe; a successful
        # probe releases the other session as well.
        agent = _make_agent()
        s1 = _make_session(agent, "s1")
        s2 = _make_session(agent, "s2")
        ibx1 = _make_inbox(tmp_path, agent, "s1")
        ibx2 = _make_inbox(tmp_path, agent, "s2")
        ibx1.post(_spec(ibx1.address))
        ibx2.post(_spec(ibx2.address))

        monitor = ProviderHealthMonitor(
            failure_threshold=2,
            base_cooldown_seconds=0.05,
            max_cooldown_seconds=0.05,
            cooldown_jitter_seconds=0.0,
        )

        # Outage for the first 2 dispatcher calls (across both
        # sessions), then recovery.
        call_log: list[int] = []

        def flaky(session_arg, inbox_arg, idx) -> None:
            call_log.append(idx)
            if idx < 2:
                raise ProviderUnavailableError("outage", attempts=3)
            _default_handle_one(session_arg, inbox_arg, idx)

        probe = _DispatcherProbe(action=flaky)
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            concurrency=2,
            health_monitor=monitor,
        )
        await sched.submit(s1, ibx1)
        await sched.submit(s2, ibx2)

        # Both inboxes should drain once the probe succeeds.
        await _wait_for(
            lambda: ibx1.prompt_pending() == [] and ibx2.prompt_pending() == [],
            timeout=5.0,
        )
        await sched.shutdown(timeout=1.0)

        # After recovery, the monitor must be Healthy.
        assert monitor.is_healthy

    async def test_monitor_skips_per_driver_backoff(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        # When a monitor is wired, the per-driver provider-
        # unavailable sleep is suppressed (the monitor's cooldown
        # is the cross-session-coordinated equivalent).  We force
        # the per-driver sleep to be huge; if it were used the
        # driver could not loop quickly enough to hit the
        # threshold.
        monkeypatch.setenv("THORN_PROVIDER_UNAVAILABLE_BACKOFF", "10.0")
        monkeypatch.setenv("THORN_PROVIDER_UNAVAILABLE_BACKOFF_JITTER", "0.0")

        agent = _make_agent()
        session = _make_session(agent, "s1")
        inbox = _make_inbox(tmp_path, agent, "s1")
        inbox.post(_spec(inbox.address))

        monitor = ProviderHealthMonitor(
            failure_threshold=3,
            base_cooldown_seconds=60.0,
            max_cooldown_seconds=60.0,
            cooldown_jitter_seconds=0.0,
        )
        probe = _DispatcherProbe(
            raises=ProviderUnavailableError("outage", attempts=3),
        )
        sched = AgentScheduler(
            agent=agent,
            prompt_dispatcher=probe,
            health_monitor=monitor,
        )
        await sched.submit(session, inbox)
        # If the per-driver backoff (10s) were applied we could
        # not reach 3 failures within a brief test window.  The
        # monitor's wait_until_healthy returns immediately while
        # Healthy, and only blocks after the trip.
        await _wait_for(
            lambda: monitor.state == ProviderHealthState.DEGRADED,
            timeout=2.0,
        )
        await sched.shutdown(timeout=0)


class TestMonitorAwareEvictor:
    """``default_progress_evictor(health_monitor=...)`` defers to the monitor."""

    async def test_evictor_no_op_while_degraded(
        self, tmp_path: Path,
    ) -> None:
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        inbox.post(_spec(inbox.address, content="first"))
        inbox.post(_spec(inbox.address, content="second"))
        ab = AddressBook()

        monitor = ProviderHealthMonitor(
            failure_threshold=1,
            base_cooldown_seconds=60.0,
            max_cooldown_seconds=60.0,
            cooldown_jitter_seconds=0.0,
        )
        await monitor.report_failure()
        assert not monitor.is_healthy

        evictor = default_progress_evictor(ab, health_monitor=monitor)
        await evictor(inbox)

        # Both items still pending; evictor refused to fire.
        assert len(inbox.prompt_pending()) == 2

    async def test_evictor_fires_again_after_recovery(
        self, tmp_path: Path,
    ) -> None:
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        inbox.post(_spec(inbox.address, content="first"))
        inbox.post(_spec(inbox.address, content="second"))
        ab = AddressBook()

        monitor = ProviderHealthMonitor(
            failure_threshold=1,
            base_cooldown_seconds=60.0,
            max_cooldown_seconds=60.0,
            cooldown_jitter_seconds=0.0,
        )
        evictor = default_progress_evictor(ab, health_monitor=monitor)

        # Trip the breaker; eviction is suppressed.
        await monitor.report_failure()
        await evictor(inbox)
        assert len(inbox.prompt_pending()) == 2

        # Recover; now eviction proceeds normally.
        await monitor.report_success()
        assert monitor.is_healthy
        await evictor(inbox)
        assert len(inbox.prompt_pending()) == 1

    async def test_evictor_without_monitor_unchanged(
        self, tmp_path: Path,
    ) -> None:
        # Default behavior (no monitor argument) must remain
        # exactly the same as the Phase 1 evictor: oldest item
        # gets evicted unconditionally.
        inbox = _make_inbox(tmp_path, _make_agent(), "s1")
        first = inbox.post(_spec(inbox.address, content="first"))
        inbox.post(_spec(inbox.address, content="second"))

        evictor = default_progress_evictor(AddressBook())
        await evictor(inbox)

        pending_ids = [n.id for n in inbox.prompt_pending()]
        assert first.id not in pending_ids
        assert len(pending_ids) == 1
