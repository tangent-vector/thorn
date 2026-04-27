"""Unit tests for :mod:`thorn.runtime._chat_router`.

The chat prompt router is the join point between the
:class:`AgentScheduler`'s fire-and-forget shape and the per-turn
"give me the agent's reply" shape that the ``thorn chat`` REPL
needs.  Phase 4 of the CLI/gateway unification puts it on the same
footing as :func:`make_cli_prompt_dispatcher` for ``thorn run``, so
this test module mirrors :mod:`tests.test_cli_dispatcher` in
structure: a duck-typed fake session lets us exercise the router's
logic without standing up a real :class:`Runtime` /
:class:`ExecutionContext`.

Tests fall into three buckets:

- **Plumbing** (``TestDispatcherPlumbing``, ``TestTurnFlow``):
  the dispatcher wakes up correctly on a posted turn, forwards
  ``extra_system``, deletes the item, and resolves
  the matching future.
- **Error policy** (``TestErrorRouting``): recoverable Thorn
  exceptions are caught so the scheduler still saves the session;
  cancellation and other ``BaseException`` subclasses propagate.
- **Future/item mismatches** (``TestQueueDiscipline``): autonomous
  rounds that arrive without a queued future are processed without
  blowing up; failures inside ``inbox.post`` drop the just-enqueued
  future so the next turn isn't routed to the wrong round.

Integration with the real :class:`AgentScheduler` (drain loop,
save_session callback, multi-turn ordering) is covered by a small
``TestSchedulerIntegration`` class that uses the real scheduler with
the fake session.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from thorn.core.errors import SkillError, ThornError
from thorn.runtime._address import SessionAddress
from thorn.runtime._chat_router import ChatPromptRouter
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import NotificationSpec
from thorn.runtime._scheduler import AgentScheduler
from thorn.runtime._session import AgentID, SessionKey


pytestmark = pytest.mark.asyncio


class _FakeAgent:
    """Stand-in for :class:`Agent` carrying just ``id``.

    The router and scheduler only access ``session.agent.id`` for
    log lines and driver-task naming, so a bare object suffices.
    """

    def __init__(self, agent_id: str = "test-agent") -> None:
        self.id = AgentID(agent_id)


class _FakeSession:
    """Duck-typed :class:`Session` with a configurable prompt stub.

    *prompt_impl* is the async callable invoked when the dispatcher
    awaits ``session.prompt(content, ...)``.  Tests can replace it
    between turns to simulate different per-round outcomes.
    """

    def __init__(self, prompt_impl) -> None:
        self.agent = _FakeAgent()
        self.key = SessionKey("test-session")
        self._prompt_impl = prompt_impl
        self.prompt_calls: list[tuple[str, dict]] = []

    @property
    def prompt(self):
        async def _call(text, **kwargs):
            self.prompt_calls.append((text, dict(kwargs)))
            return await self._prompt_impl(text, **kwargs)

        return _call


def _make_inbox(tmp_path: Path) -> tuple[SessionInbox, SessionAddress]:
    address = SessionAddress(AgentID("test-agent"), SessionKey("test-session"))
    inbox_dir = tmp_path / "inbox"
    inbox = SessionInbox(inbox_dir, address)
    return inbox, address


def _spec(target: SessionAddress, content: str = "hello") -> NotificationSpec:
    return NotificationSpec(
        source="user",
        content=content,
        target=target,
        rsvp_to=None,
        external_key=None,
    )


# ---------------------------------------------------------------------------
# Dispatcher plumbing
# ---------------------------------------------------------------------------

class TestDispatcherPlumbing:
    """Calls into ``router.dispatcher`` directly (no scheduler).

    Verifies the dispatcher's local contract -- empty-inbox no-op,
    extra_system propagation, item deletion -- without the additional
    moving parts a real scheduler brings.
    """

    async def test_empty_inbox_returns_without_calling_prompt(
        self, tmp_path: Path,
    ):
        async def prompt_impl(text, **kwargs):
            raise AssertionError("prompt should not be called for empty inbox")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        await router.dispatcher(session, inbox)

        assert session.prompt_calls == []
        assert router.pending_count == 0

    async def test_forwards_extra_system(self, tmp_path: Path):
        sentinel_system = "you are a helpful assistant"

        async def prompt_impl(text, **kwargs):
            return "ok"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(
            target=address,
            extra_system=sentinel_system,
        )

        # Pre-seed the queue and inbox so the dispatcher has work.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        router._pending.append(fut)
        inbox.post(_spec(address))

        await router.dispatcher(session, inbox)

        text, kwargs = session.prompt_calls[0]
        assert kwargs["system"] == sentinel_system
        assert "tools" not in kwargs
        assert fut.result() == "ok"

    async def test_deletes_item_after_success(self, tmp_path: Path):
        """Otherwise the scheduler's progress guarantee would stall."""
        async def prompt_impl(text, **kwargs):
            return "done"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        router._pending.append(fut)
        inbox.post(_spec(address))

        await router.dispatcher(session, inbox)

        assert inbox.prompt_pending() == []


# ---------------------------------------------------------------------------
# Turn flow (router-only, no scheduler)
# ---------------------------------------------------------------------------

class TestTurnFlow:
    """``turn`` enqueues a future, posts, kicks the scheduler, and awaits.

    The scheduler is a stub here -- ``turn``'s only contract with the
    scheduler is calling ``submit``, so we don't need a real drain
    loop to verify the turn ordering.
    """

    async def test_turn_enqueues_future_then_posts(self, tmp_path: Path):
        """The future *must* land on the queue before the post.

        Otherwise an instantly-running dispatcher (e.g. an idle drain
        loop kicked by a prior submit) could pop nothing on its first
        round and discard a result the awaiter is about to wait for.
        """

        async def prompt_impl(text, **kwargs):
            return "answer"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        observed_pending_at_post: list[int] = []
        original_post = inbox.post

        def spy_post(spec):
            observed_pending_at_post.append(router.pending_count)
            return original_post(spec)

        inbox.post = spy_post  # type: ignore[method-assign]

        scheduler_calls: list[tuple] = []

        class _StubScheduler:
            async def submit(self, sess, ibx):
                scheduler_calls.append((sess, ibx))
                # Immediately drive one round so the turn's future
                # resolves; the real scheduler would do this via its
                # drain task.
                await router.dispatcher(sess, ibx)

        result = await router.turn(
            scheduler=_StubScheduler(),  # type: ignore[arg-type]
            session=session,
            inbox=inbox,
            prompt_text="hi",
        )

        assert result == "answer"
        assert scheduler_calls == [(session, inbox)]
        assert observed_pending_at_post == [1], (
            "the future must be enqueued before inbox.post is called"
        )

    async def test_post_failure_drops_queued_future(self, tmp_path: Path):
        """``turn`` must clean up a future whose post never landed."""

        class _BadInbox:
            address = SessionAddress(
                AgentID("a"), SessionKey("s"),
            )

            def post(self, spec):
                raise RuntimeError("disk is full")

        session = _FakeSession(lambda *a, **k: None)
        bad_inbox = _BadInbox()
        router = ChatPromptRouter(target=bad_inbox.address)

        class _StubScheduler:
            async def submit(self, sess, ibx):
                raise AssertionError(
                    "submit must not be called when post failed"
                )

        with pytest.raises(RuntimeError, match="disk is full"):
            await router.turn(
                scheduler=_StubScheduler(),  # type: ignore[arg-type]
                session=session,
                inbox=bad_inbox,  # type: ignore[arg-type]
                prompt_text="anything",
            )

        assert router.pending_count == 0, (
            "post failed, so the just-enqueued future has no chance "
            "of being resolved and must be dropped to keep the queue "
            "aligned with future inbox state"
        )


# ---------------------------------------------------------------------------
# Error policy
# ---------------------------------------------------------------------------

class TestErrorRouting:
    """How the dispatcher reports per-round failures.

    Recoverable Thorn exceptions are routed to the awaiting future and
    swallowed by the dispatcher, so the scheduler runs ``save_session``
    afterwards (history was mutated even though the prompt raised).
    Other ``BaseException`` subclasses propagate to the scheduler so
    its own logging fires.  In every error case the inbox item is
    still deleted so the progress guarantee does not treat the round
    as a stall.
    """

    async def test_skill_error_is_caught_and_routed_to_future(
        self, tmp_path: Path,
    ):
        async def prompt_impl(text, **kwargs):
            raise SkillError("the skill blew up")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        router._pending.append(fut)
        inbox.post(_spec(address))

        await router.dispatcher(session, inbox)

        assert fut.done()
        with pytest.raises(SkillError):
            fut.result()
        assert inbox.prompt_pending() == []

    async def test_thorn_error_is_caught_and_routed_to_future(
        self, tmp_path: Path,
    ):
        async def prompt_impl(text, **kwargs):
            raise ThornError("transient provider hiccup")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        router._pending.append(fut)
        inbox.post(_spec(address))

        await router.dispatcher(session, inbox)

        assert fut.done()
        with pytest.raises(ThornError):
            fut.result()
        assert inbox.prompt_pending() == []

    async def test_runtime_error_propagates_and_routes_to_future(
        self, tmp_path: Path,
    ):
        """Non-Thorn errors propagate so scheduler logging fires.

        The future also carries the exception so any awaiting caller
        unblocks rather than hanging forever.
        """

        class BoomError(RuntimeError):
            pass

        async def prompt_impl(text, **kwargs):
            raise BoomError("kapow")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        router._pending.append(fut)
        inbox.post(_spec(address))

        with pytest.raises(BoomError):
            await router.dispatcher(session, inbox)

        assert fut.done()
        with pytest.raises(BoomError, match="kapow"):
            fut.result()
        assert inbox.prompt_pending() == []

    async def test_cancellation_propagates_and_routes_to_future(
        self, tmp_path: Path,
    ):
        async def prompt_impl(text, **kwargs):
            raise asyncio.CancelledError()

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        router._pending.append(fut)
        inbox.post(_spec(address))

        with pytest.raises(asyncio.CancelledError):
            await router.dispatcher(session, inbox)

        assert fut.cancelled() or (
            fut.done() and isinstance(
                fut.exception(), asyncio.CancelledError,
            )
        )
        assert inbox.prompt_pending() == []


# ---------------------------------------------------------------------------
# Queue discipline
# ---------------------------------------------------------------------------

class TestQueueDiscipline:
    """Behaviour when futures and items get out of sync.

    The router treats "item without future" as a benign autonomous
    round (process it, discard the result) rather than a programming
    error, because the inbox model permits it even though the chat
    REPL never produces it today.
    """

    async def test_autonomous_round_runs_prompt_and_discards_result(
        self, tmp_path: Path,
    ):
        async def prompt_impl(text, **kwargs):
            return "auto-result"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        # No future enqueued; item appears in the inbox by some other
        # path.
        inbox.post(_spec(address, content="autonomous"))

        await router.dispatcher(session, inbox)

        assert len(session.prompt_calls) == 1
        assert session.prompt_calls[0][0] == "autonomous"
        assert inbox.prompt_pending() == []
        assert router.pending_count == 0

    async def test_two_turns_resolve_in_post_order(self, tmp_path: Path):
        """The FIFO queue must pair futures with rounds in order."""
        results = iter(["first-answer", "second-answer"])

        async def prompt_impl(text, **kwargs):
            return next(results)

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        loop = asyncio.get_running_loop()
        fut_a: asyncio.Future = loop.create_future()
        fut_b: asyncio.Future = loop.create_future()
        router._pending.append(fut_a)
        router._pending.append(fut_b)
        inbox.post(_spec(address, content="q1"))
        inbox.post(_spec(address, content="q2"))

        await router.dispatcher(session, inbox)
        await router.dispatcher(session, inbox)

        assert fut_a.result() == "first-answer"
        assert fut_b.result() == "second-answer"


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    """End-to-end ``router.turn`` over a real :class:`AgentScheduler`.

    Exercises the multi-turn flow that ``thorn chat`` will rely on:
    each ``turn`` call returns when the matching round completes, and
    ``save_session`` runs on every successful (or recoverable-failure)
    round.
    """

    async def test_turn_returns_prompt_result(self, tmp_path: Path):
        async def prompt_impl(text, **kwargs):
            return f"echo: {text}"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        save_calls: list[_FakeSession] = []

        async def save(sess):
            save_calls.append(sess)

        scheduler = AgentScheduler(
            agent=session.agent,  # type: ignore[arg-type]
            prompt_dispatcher=router.dispatcher,
            save_session=save,
        )

        try:
            r1 = await router.turn(
                scheduler=scheduler,
                session=session,  # type: ignore[arg-type]
                inbox=inbox,
                prompt_text="ping",
            )
            r2 = await router.turn(
                scheduler=scheduler,
                session=session,  # type: ignore[arg-type]
                inbox=inbox,
                prompt_text="pong",
            )
        finally:
            await scheduler.shutdown(timeout=2.0)

        assert r1 == "echo: ping"
        assert r2 == "echo: pong"
        assert len(save_calls) == 2, (
            "save_session must run after every successful round"
        )

    async def test_recoverable_error_still_runs_save(self, tmp_path: Path):
        """SkillError mid-turn must still trigger ``save_session``.

        The dispatcher catches the exception and returns normally so
        the scheduler reaches its save callback; the future raises so
        the REPL prints the error and continues.
        """

        async def prompt_impl(text, **kwargs):
            raise SkillError("agent failed")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        save_calls: list[_FakeSession] = []

        async def save(sess):
            save_calls.append(sess)

        scheduler = AgentScheduler(
            agent=session.agent,  # type: ignore[arg-type]
            prompt_dispatcher=router.dispatcher,
            save_session=save,
        )

        try:
            with pytest.raises(SkillError):
                await router.turn(
                    scheduler=scheduler,
                    session=session,  # type: ignore[arg-type]
                    inbox=inbox,
                    prompt_text="please fail",
                )
        finally:
            await scheduler.shutdown(timeout=2.0)

        assert len(save_calls) == 1, (
            "scheduler must save after a recoverable failure so the "
            "partial history is not lost"
        )

    async def test_runtime_error_skips_save(self, tmp_path: Path):
        """Non-Thorn errors propagate from the dispatcher.

        The scheduler observes the raise, logs it, and skips the
        ``save_session`` callback (matching its own contract for
        dispatcher exceptions).
        """

        async def prompt_impl(text, **kwargs):
            raise RuntimeError("kapow")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        router = ChatPromptRouter(target=address)

        save_calls: list[_FakeSession] = []

        async def save(sess):
            save_calls.append(sess)

        scheduler = AgentScheduler(
            agent=session.agent,  # type: ignore[arg-type]
            prompt_dispatcher=router.dispatcher,
            save_session=save,
        )

        try:
            with pytest.raises(RuntimeError, match="kapow"):
                await router.turn(
                    scheduler=scheduler,
                    session=session,  # type: ignore[arg-type]
                    inbox=inbox,
                    prompt_text="please crash",
                )
        finally:
            await scheduler.shutdown(timeout=2.0)

        assert save_calls == [], (
            "non-Thorn dispatcher errors signal a state we do not "
            "want to persist; scheduler skips save and the REPL "
            "exits with the exception"
        )
