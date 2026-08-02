"""Unit tests for SessionInbox and NotificationQueue.

Covers the Phase 1 session-inbox and notification-queue contracts:

- SessionInbox slices its contents by lifecycle state:
  prompt-pending, awaiting-dispatch, awaiting-cleanup.
- NotificationQueue classifies each arrival as fresh or RSVP and
  rejects confirmed items.
- drain() invokes the appropriate registered handler, completes the
  two-step confirmation ceremony on success, captures handler
  exceptions without propagating, and recovers stuck-confirmed items
  without invoking any handler.
- drain() skips ``in_progress`` items so concurrent handler work is
  not double-triggered.
- Missing-handler case returns NO_HANDLER without deleting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.runtime._address import (
    Address,
    AddressBook,
    ServiceAddress,
    SessionAddress,
)
from thorn.runtime._dispatch import apply_handling_transition
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    Notification,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import (
    ArrivalKind,
    DrainResult,
    NotificationQueue,
)
from thorn.runtime._session import AgentID, SessionKey

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _session_addr(key: str = "proj/mr/42") -> SessionAddress:
    return SessionAddress(AgentID("coord"), SessionKey(key))


def _service_addr(name: str = "gitlab-primary") -> ServiceAddress:
    return ServiceAddress(name)


def _fresh_spec(
    *,
    target: Address,
    rsvp_to: Address | None = None,
    external_key: str | None = None,
    content: str = "hello",
) -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        metadata={},
        rsvp_to=rsvp_to,
        external_key=external_key,
    )


# ---------------------------------------------------------------------------
# SessionInbox
# ---------------------------------------------------------------------------

class TestSessionInbox:
    def test_address_is_recorded(self, tmp_path: Path) -> None:
        addr = _session_addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)
        assert inbox.address == addr

    def test_prompt_pending_filters_to_pending_and_in_progress(
        self, tmp_path: Path
    ) -> None:
        addr = _session_addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)

        n_pending = inbox.post(_fresh_spec(target=addr))
        n_in_prog = inbox.post(_fresh_spec(target=addr))
        n_handled = inbox.post(_fresh_spec(target=addr))
        n_errored = inbox.post(_fresh_spec(target=addr))
        n_confirmed = inbox.post(_fresh_spec(target=addr))

        inbox.update_status(n_in_prog.id, NotificationStatus.IN_PROGRESS)
        inbox.update_status(n_handled.id, NotificationStatus.HANDLED, notes="x")
        inbox.update_status(
            n_errored.id, NotificationStatus.ERRORED, error_reason="boom",
        )
        inbox.update_status(n_confirmed.id, NotificationStatus.CONFIRMED)

        prompt = inbox.prompt_pending()
        assert {n.id for n in prompt} == {n_pending.id, n_in_prog.id}
        # Order is ULID-sorted, which equals post order here.
        assert [n.id for n in prompt] == [n_pending.id, n_in_prog.id]

    def test_awaiting_dispatch_filters_to_handled_and_errored(
        self, tmp_path: Path
    ) -> None:
        addr = _session_addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)

        n1 = inbox.post(_fresh_spec(target=addr))
        n2 = inbox.post(_fresh_spec(target=addr))
        n3 = inbox.post(_fresh_spec(target=addr))
        inbox.update_status(n2.id, NotificationStatus.HANDLED, notes="ok")
        inbox.update_status(n3.id, NotificationStatus.ERRORED, error_reason="bad")

        awaiting = inbox.awaiting_dispatch()
        assert {n.id for n in awaiting} == {n2.id, n3.id}
        assert n1.id not in {n.id for n in awaiting}

    def test_awaiting_cleanup_filters_to_confirmed(self, tmp_path: Path) -> None:
        addr = _session_addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)

        inbox.post(_fresh_spec(target=addr))
        n2 = inbox.post(_fresh_spec(target=addr))
        inbox.update_status(n2.id, NotificationStatus.CONFIRMED)

        cleanup = inbox.awaiting_cleanup()
        assert [n.id for n in cleanup] == [n2.id]

    def test_requeue_errored_restores_prompt_pending_work(
        self, tmp_path: Path
    ) -> None:
        addr = _session_addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)
        posted = inbox.post(
            _fresh_spec(
                target=addr,
                external_key="gitlab:https://gitlab.example.com:todo:123",
                content="please handle this issue",
            )
        )

        apply_handling_transition(
            inbox,
            posted.id,
            NotificationStatus.ERRORED,
            address_book=AddressBook(),
            error_reason="provider key was invalid",
        )

        parked = inbox.errored_items()
        assert [item.id for item in parked] == [posted.id]
        assert parked[0].status is NotificationStatus.ERRORED
        assert parked[0].error_reason == "provider key was invalid"
        assert inbox.prompt_pending() == []

        requeued = inbox.requeue_errored(posted.id)

        assert requeued.id == posted.id
        assert requeued.status is NotificationStatus.PENDING
        assert requeued.content == "please handle this issue"
        assert (
            requeued.external_key
            == "gitlab:https://gitlab.example.com:todo:123"
        )
        assert requeued.error_reason is None
        assert requeued.notes is None
        assert inbox.errored_items() == []
        assert [item.id for item in inbox.prompt_pending()] == [posted.id]

    def test_requeue_missing_errored_item_raises(
        self, tmp_path: Path
    ) -> None:
        inbox = SessionInbox(tmp_path / "inbox", _session_addr())

        with pytest.raises(KeyError):
            inbox.requeue_errored("missing-item")


# ---------------------------------------------------------------------------
# NotificationQueue: classification
# ---------------------------------------------------------------------------

class TestNotificationQueueClassify:
    def test_target_match_is_fresh(self, tmp_path: Path) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        n = Notification.from_spec(_fresh_spec(target=addr))
        assert q.classify(n) is ArrivalKind.FRESH

    def test_rsvp_match_is_rsvp(self, tmp_path: Path) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        n = Notification.from_spec(
            _fresh_spec(target=_session_addr(), rsvp_to=addr),
        )
        assert q.classify(n) is ArrivalKind.RSVP

    def test_rejects_unrelated_notification(self, tmp_path: Path) -> None:
        q = NotificationQueue(tmp_path / "q", _service_addr())
        n = Notification.from_spec(_fresh_spec(target=_session_addr()))
        with pytest.raises(ValueError):
            q.classify(n)

    def test_rejects_confirmed_notification(self, tmp_path: Path) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        n = Notification.from_spec(_fresh_spec(target=addr)).with_updates(
            status=NotificationStatus.CONFIRMED,
        )
        with pytest.raises(ValueError):
            q.classify(n)


# ---------------------------------------------------------------------------
# NotificationQueue: handler registration
# ---------------------------------------------------------------------------

class TestNotificationQueueHandlers:
    def test_duplicate_fresh_registration_raises(self, tmp_path: Path) -> None:
        q = NotificationQueue(tmp_path / "q", _service_addr())

        async def h(_n: Notification) -> None:
            pass

        q.on_fresh(h)
        with pytest.raises(ValueError):
            q.on_fresh(h)

    def test_duplicate_rsvp_registration_raises(self, tmp_path: Path) -> None:
        q = NotificationQueue(tmp_path / "q", _service_addr())

        async def h(_n: Notification) -> None:
            pass

        q.on_rsvp(h)
        with pytest.raises(ValueError):
            q.on_rsvp(h)

    def test_clear_allows_reregistration(self, tmp_path: Path) -> None:
        q = NotificationQueue(tmp_path / "q", _service_addr())

        async def h(_n: Notification) -> None:
            pass

        q.on_fresh(h)
        q.on_rsvp(h)
        q.clear_handlers()
        q.on_fresh(h)
        q.on_rsvp(h)  # Should not raise.


# ---------------------------------------------------------------------------
# NotificationQueue: drain
# ---------------------------------------------------------------------------

class _RecordingHandler:
    """Test helper: captures every notification passed to it and can
    optionally raise on the n-th call."""

    def __init__(self, *, raise_on: set[int] | None = None) -> None:
        self.calls: list[Notification] = []
        self._raise_on = raise_on or set()

    async def __call__(self, notification: Notification) -> None:
        self.calls.append(notification)
        idx = len(self.calls)
        if idx in self._raise_on:
            raise RuntimeError(f"boom at call #{idx}")


class TestNotificationQueueDrain:
    async def test_drain_fresh_happy_path(self, tmp_path: Path) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        handler = _RecordingHandler()
        q.on_fresh(handler)

        posted = q.post(_fresh_spec(target=addr))
        outcomes = await q.drain()

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.notification_id == posted.id
        assert outcome.kind is ArrivalKind.FRESH
        assert outcome.result is DrainResult.HANDLED
        assert outcome.error is None

        assert [n.id for n in handler.calls] == [posted.id]
        # File is gone after successful drain.
        assert posted.id not in q
        assert q.list() == []

    async def test_drain_rsvp_happy_path(self, tmp_path: Path) -> None:
        service_addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", service_addr)
        handler = _RecordingHandler()
        q.on_rsvp(handler)

        # Simulate an RSVP landing in this queue: target is a session,
        # rsvp_to is this service, status is handled.
        notification = Notification.from_spec(
            _fresh_spec(target=_session_addr(), rsvp_to=service_addr),
        ).with_updates(status=NotificationStatus.HANDLED, notes="done")
        # Post as-is via a direct write, bypassing post() so we don't
        # reset status back to pending.
        q._write_atomic(notification)  # type: ignore[attr-defined]

        outcomes = await q.drain()
        assert len(outcomes) == 1
        assert outcomes[0].kind is ArrivalKind.RSVP
        assert outcomes[0].result is DrainResult.HANDLED
        assert [n.id for n in handler.calls] == [notification.id]
        assert notification.id not in q

    async def test_handler_failure_leaves_notification_in_place(
        self, tmp_path: Path
    ) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        handler = _RecordingHandler(raise_on={1})
        q.on_fresh(handler)

        posted = q.post(_fresh_spec(target=addr))
        outcomes = await q.drain()

        assert len(outcomes) == 1
        assert outcomes[0].result is DrainResult.HANDLER_FAILED
        assert isinstance(outcomes[0].error, RuntimeError)
        # Notification still in place, at its original status.
        still_there = q.get(posted.id)
        assert still_there.status is NotificationStatus.PENDING

    async def test_no_handler_registered_returns_no_handler_outcome(
        self, tmp_path: Path
    ) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        posted = q.post(_fresh_spec(target=addr))

        outcomes = await q.drain()
        assert len(outcomes) == 1
        assert outcomes[0].result is DrainResult.NO_HANDLER
        # File untouched.
        assert q.get(posted.id).status is NotificationStatus.PENDING

    async def test_confirmed_item_is_recovered_without_handler(
        self, tmp_path: Path
    ) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        handler = _RecordingHandler()
        q.on_fresh(handler)

        posted = q.post(_fresh_spec(target=addr))
        # Simulate: step 1 mutated to confirmed, step 2 (delete) never ran.
        q.update_status(posted.id, NotificationStatus.CONFIRMED)

        outcomes = await q.drain()
        assert len(outcomes) == 1
        assert outcomes[0].result is DrainResult.RECOVERED
        assert outcomes[0].kind is ArrivalKind.FRESH
        # Handler must not have been invoked -- the work is presumed
        # already done by a prior run.
        assert handler.calls == []
        # File is gone.
        assert posted.id not in q

    async def test_confirmed_rsvp_recovered_with_rsvp_kind(
        self, tmp_path: Path
    ) -> None:
        service_addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", service_addr)
        handler = _RecordingHandler()
        q.on_rsvp(handler)

        rsvp_item = Notification.from_spec(
            _fresh_spec(target=_session_addr(), rsvp_to=service_addr),
        ).with_updates(status=NotificationStatus.CONFIRMED)
        q._write_atomic(rsvp_item)  # type: ignore[attr-defined]

        outcomes = await q.drain()
        assert len(outcomes) == 1
        assert outcomes[0].kind is ArrivalKind.RSVP
        assert outcomes[0].result is DrainResult.RECOVERED
        assert handler.calls == []
        assert rsvp_item.id not in q

    async def test_drain_skips_in_progress_items(self, tmp_path: Path) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        handler = _RecordingHandler()
        q.on_fresh(handler)

        busy = q.post(_fresh_spec(target=addr))
        q.update_status(busy.id, NotificationStatus.IN_PROGRESS)

        outcomes = await q.drain()
        # Nothing processed: the sole item is in_progress.
        assert outcomes == []
        # File still there, still in_progress.
        assert q.get(busy.id).status is NotificationStatus.IN_PROGRESS
        assert handler.calls == []

    async def test_drain_processes_multiple_in_post_order(
        self, tmp_path: Path
    ) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        handler = _RecordingHandler()
        q.on_fresh(handler)

        posted = [q.post(_fresh_spec(target=addr)) for _ in range(4)]
        outcomes = await q.drain()

        assert [o.notification_id for o in outcomes] == [n.id for n in posted]
        assert [n.id for n in handler.calls] == [n.id for n in posted]
        assert q.list() == []

    async def test_handler_failure_does_not_stop_subsequent_items(
        self, tmp_path: Path
    ) -> None:
        addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", addr)
        handler = _RecordingHandler(raise_on={2})  # Second call fails.
        q.on_fresh(handler)

        a, b, c = [q.post(_fresh_spec(target=addr)) for _ in range(3)]
        outcomes = await q.drain()

        assert [o.result for o in outcomes] == [
            DrainResult.HANDLED,
            DrainResult.HANDLER_FAILED,
            DrainResult.HANDLED,
        ]
        # a and c were completed; b is still in the queue.
        assert a.id not in q
        assert b.id in q
        assert c.id not in q

    async def test_mixed_fresh_and_rsvp_dispatch_by_kind(
        self, tmp_path: Path
    ) -> None:
        service_addr = _service_addr()
        q = NotificationQueue(tmp_path / "q", service_addr)

        fresh_handler = _RecordingHandler()
        rsvp_handler = _RecordingHandler()
        q.on_fresh(fresh_handler)
        q.on_rsvp(rsvp_handler)

        # Fresh arrival.
        fresh = q.post(_fresh_spec(target=service_addr))
        # RSVP arrival (synthetic: handled status from another queue).
        rsvp = Notification.from_spec(
            _fresh_spec(target=_session_addr(), rsvp_to=service_addr),
        ).with_updates(status=NotificationStatus.HANDLED, notes="ok")
        q._write_atomic(rsvp)  # type: ignore[attr-defined]

        outcomes = await q.drain()
        assert {(o.notification_id, o.kind) for o in outcomes} == {
            (fresh.id, ArrivalKind.FRESH),
            (rsvp.id, ArrivalKind.RSVP),
        }
        assert [n.id for n in fresh_handler.calls] == [fresh.id]
        assert [n.id for n in rsvp_handler.calls] == [rsvp.id]
