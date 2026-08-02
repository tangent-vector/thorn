"""Unit tests for :mod:`thorn.runtime._cli_dispatcher`.

The CLI prompt dispatcher is short on logic but is the join point
between the scheduler's fire-and-forget shape and the synchronous
"give me the answer" shape that ``thorn run`` (and eventually
``thorn chat`` after Phase 4 of the CLI/gateway unification) needs.
That makes it worth a focused unit test, distinct from the broader
end-to-end coverage in ``tests/test_cli.py`` which exercises the
dispatcher only as a side effect of the ``thorn run`` command path.

Tests use a duck-typed fake session whose ``prompt`` is a stub
async function, so a real :class:`Runtime` / :class:`ExecutionContext`
is not needed.  The dispatcher's only real dependency is
``await session.prompt(...)``; everything else (inbox shape,
notification semantics) is shared with the durable queue layer and
already covered by ``test_inbox.py`` / ``test_queue.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from thorn.runtime._address import AddressBook, SessionAddress
from thorn.runtime._cli_dispatcher import (
    establish_fresh_cli_direct_focus,
    make_cli_prompt_dispatcher,
)
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import NotificationSpec, NotificationStatus
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._working_set import HandlingPhase, WorkingSet

pytestmark = pytest.mark.asyncio


class _FakeAgent:
    """Stand-in for :class:`Agent` carrying just an ``id``.

    The dispatcher's only access to the agent is via
    ``session.agent.id`` for log lines, so a bare object suffices.
    """

    def __init__(self, agent_id: str = "test-agent") -> None:
        self.id = AgentID(agent_id)


class _FakeSession:
    """Duck-typed :class:`Session` with a configurable prompt stub.

    *prompt_impl* is the async callable invoked when the dispatcher
    awaits ``session.prompt(content, ...)``.  It receives the same
    keyword arguments the real ``Session.prompt`` would, so tests can
    assert on tool/system propagation.
    """

    def __init__(self, prompt_impl) -> None:
        self.agent = _FakeAgent()
        self.key = SessionKey("test-session")
        self._prompt_impl = prompt_impl
        self.prompt_calls: list[tuple[str, dict]] = []
        self.working_set = WorkingSet()

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


def _spec(target: SessionAddress, content: str = "do the thing") -> NotificationSpec:
    return NotificationSpec(
        source="user",
        content=content,
        target=target,
        rsvp_to=None,
        external_key=None,
    )


class TestEmptyInbox:
    """Dispatcher must no-op cleanly when the inbox is empty.

    Mirrors :func:`inbox_prompt_dispatcher`'s behaviour: the scheduler
    may invoke a dispatcher on a stale wake (e.g. an item transitioned
    to terminal between the inbox check and the dispatcher call).
    """

    async def test_returns_without_calling_prompt(self, tmp_path: Path):
        async def prompt_impl(text, **kwargs):
            raise AssertionError("prompt should not be called for an empty inbox")

        session = _FakeSession(prompt_impl)
        inbox, _ = _make_inbox(tmp_path)
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(result_future=future)

        await dispatcher(session, inbox)

        assert session.prompt_calls == []
        assert not future.done(), "future must not resolve when no work was found"


class TestSingleItem:
    """One pending notification triggers one prompt and resolves the future."""

    async def test_resolves_future_with_prompt_result(self, tmp_path: Path):
        async def prompt_impl(text, **kwargs):
            return "agent-response"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        inbox.post(_spec(address, content="please respond"))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(result_future=future)

        await dispatcher(session, inbox)

        assert future.done()
        assert future.result() == "agent-response"
        assert len(session.prompt_calls) == 1
        text, kwargs = session.prompt_calls[0]
        assert text == "please respond"

    async def test_deletes_notification_after_success(self, tmp_path: Path):
        """Prevents the scheduler's drain loop from re-presenting it."""
        async def prompt_impl(text, **kwargs):
            return "ok"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        inbox.post(_spec(address))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(result_future=future)

        await dispatcher(session, inbox)

        assert inbox.prompt_pending() == [], (
            "the processed notification must be removed so the driver "
            "does not loop on it"
        )

    async def test_forwards_extra_system(self, tmp_path: Path):
        sentinel_system = "you are a strict pirate"

        async def prompt_impl(text, **kwargs):
            return "arr"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        inbox.post(_spec(address))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(
            result_future=future,
            extra_system=sentinel_system,
        )

        await dispatcher(session, inbox)

        text, kwargs = session.prompt_calls[0]
        assert kwargs["system"] == sentinel_system
        assert "tools" not in kwargs


class TestFrameworkFocus:
    """Fresh direct delivery may establish focus without model round-trips."""

    async def test_focus_is_visible_during_prompt_and_reset_after_return(
        self, tmp_path: Path,
    ) -> None:
        seen_working_sets: list[WorkingSet] = []

        async def prompt_impl(text, **kwargs):
            seen_working_sets.append(session.working_set)
            return "ok"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        posted = inbox.post(_spec(address, content="repair the parser"))

        focused_item_id = establish_fresh_cli_direct_focus(
            session=session,
            inbox=inbox,
            notification=posted,
            address_book=AddressBook(),
        )

        assert focused_item_id == posted.id
        assert inbox.get(posted.id).status is NotificationStatus.IN_PROGRESS
        assert session.working_set == WorkingSet(
            phase=HandlingPhase.INSPECT,
            focused_inbox_item_id=posted.id,
            objective="repair the parser",
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(
            result_future=future,
            framework_focused_item_id=focused_item_id,
        )
        await dispatcher(session, inbox)

        assert seen_working_sets[0].focused_inbox_item_id == posted.id
        assert session.working_set == WorkingSet()

    async def test_multiple_pending_items_preserve_intake(
        self, tmp_path: Path,
    ) -> None:
        async def prompt_impl(text, **kwargs):
            return "unused"

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        first = inbox.post(_spec(address, content="first"))
        inbox.post(_spec(address, content="second"))

        focused_item_id = establish_fresh_cli_direct_focus(
            session=session,
            inbox=inbox,
            notification=first,
            address_book=AddressBook(),
        )

        assert focused_item_id is None
        assert session.working_set == WorkingSet()
        assert inbox.get(first.id).status is NotificationStatus.PENDING


class TestErrors:
    """Errors raised by the prompt must be visible to the awaiting caller."""

    async def test_prompt_exception_resolves_future_with_exception(
        self, tmp_path: Path,
    ):
        class BoomError(RuntimeError):
            pass

        async def prompt_impl(text, **kwargs):
            raise BoomError("the prompt blew up")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        inbox.post(_spec(address))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(result_future=future)

        with pytest.raises(BoomError):
            # The dispatcher re-raises so the scheduler's per-round
            # logging fires; the future also carries the exception so
            # the awaiting caller learns about it.
            await dispatcher(session, inbox)

        assert future.done()
        with pytest.raises(BoomError, match="the prompt blew up"):
            future.result()

    async def test_prompt_exception_still_deletes_notification(
        self, tmp_path: Path,
    ):
        """Otherwise the progress guarantee would treat the round as a stall."""
        async def prompt_impl(text, **kwargs):
            raise RuntimeError("nope")

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        inbox.post(_spec(address))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(result_future=future)

        with pytest.raises(RuntimeError):
            await dispatcher(session, inbox)

        assert inbox.prompt_pending() == []


class TestFutureResolutionIsOneShot:
    """A second invocation must not re-resolve an already-resolved future.

    For ``thorn run`` exactly one notification is ever posted, so the
    dispatcher runs exactly once.  But the scheduler is free to call
    the dispatcher again on a follow-up wake; the second call must
    not call ``set_result``/``set_exception`` on a future the awaiting
    caller has already consumed (that would itself raise).
    """

    async def test_second_invocation_does_not_touch_resolved_future(
        self, tmp_path: Path,
    ):
        prompt_results = iter(["first", "second"])

        async def prompt_impl(text, **kwargs):
            return next(prompt_results)

        session = _FakeSession(prompt_impl)
        inbox, address = _make_inbox(tmp_path)
        inbox.post(_spec(address, content="first item"))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        dispatcher = make_cli_prompt_dispatcher(result_future=future)

        await dispatcher(session, inbox)
        assert future.result() == "first"

        # Post a second notification and run the dispatcher again.
        # The future is already done; the dispatcher must not crash
        # trying to set_result on it.
        inbox.post(_spec(address, content="second item"))
        await dispatcher(session, inbox)

        assert future.result() == "first", (
            "future result must be the value the awaiting caller "
            "already consumed; second-round result is discarded"
        )
        assert len(session.prompt_calls) == 2
