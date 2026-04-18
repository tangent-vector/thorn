"""Unit tests for ``apply_handling_transition`` and ``dispatch_step_two``.

Covers every (status x rsvp_to-present) combination plus the
step-1-only ``IN_PROGRESS`` transition and the guardrail errors for
invalid inputs.  All tests use a real filesystem so they exercise
the atomic-rename behavior end-to-end.
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
from thorn.runtime._dispatch import (
    DispatchError,
    apply_handling_transition,
    dispatch_step_two,
)
from thorn.runtime._in_flight_index import InFlightIndex
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._queue import DurableQueue
from thorn.runtime._session import AgentID, SessionKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_addr(key: str = "proj/1") -> SessionAddress:
    return SessionAddress(AgentID("coord"), SessionKey(key))


def _service_addr(name: str = "forge") -> ServiceAddress:
    return ServiceAddress(name)


def _spec(
    *,
    target: Address,
    rsvp_to: Address | None = None,
    external_key: str | None = None,
    content: str = "body",
) -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        metadata={},
        rsvp_to=rsvp_to,
        external_key=external_key,
    )


@pytest.fixture
def book() -> AddressBook:
    return AddressBook()


@pytest.fixture
def index() -> InFlightIndex:
    return InFlightIndex()


@pytest.fixture
def inbox(tmp_path: Path, index: InFlightIndex) -> SessionInbox:
    return SessionInbox(
        tmp_path / "inbox", _session_addr(), in_flight_index=index,
    )


@pytest.fixture
def forge_queue(
    tmp_path: Path, index: InFlightIndex, book: AddressBook,
) -> NotificationQueue:
    addr = _service_addr()
    queue = NotificationQueue(
        tmp_path / "forge_queue", addr, in_flight_index=index,
    )
    book.register(addr, queue)
    return queue


# ---------------------------------------------------------------------------
# Step 1 only (IN_PROGRESS)
# ---------------------------------------------------------------------------

class TestInProgress:
    def test_in_progress_updates_status_in_place(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        updated = apply_handling_transition(
            inbox, posted.id, NotificationStatus.IN_PROGRESS,
            address_book=book, notes="working on it",
        )
        assert updated.status is NotificationStatus.IN_PROGRESS
        assert updated.notes == "working on it"
        # File stays in the inbox (prompt_pending still lists it).
        pending = inbox.prompt_pending()
        assert [n.id for n in pending] == [posted.id]

    def test_in_progress_without_notes_preserves_existing_notes(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        # Notes carried across status transitions is explicitly part
        # of the design: callers leaving notes=None do not wipe a
        # previously-recorded annotation.
        posted = inbox.post(_spec(target=inbox.address))
        inbox.update_status(posted.id, NotificationStatus.PENDING, notes="earlier")
        updated = apply_handling_transition(
            inbox, posted.id, NotificationStatus.IN_PROGRESS,
            address_book=book,
        )
        assert updated.notes == "earlier"

    def test_in_progress_rejects_error_reason(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        with pytest.raises(ValueError, match="error_reason is only valid"):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.IN_PROGRESS,
                address_book=book, error_reason="boom",
            )


# ---------------------------------------------------------------------------
# HANDLED transitions
# ---------------------------------------------------------------------------

class TestHandled:
    def test_handled_without_rsvp_deletes_file(
        self, inbox: SessionInbox, index: InFlightIndex, book: AddressBook,
    ) -> None:
        posted = inbox.post(
            _spec(target=inbox.address, external_key="ext-1")
        )
        assert index.contains("ext-1")
        updated = apply_handling_transition(
            inbox, posted.id, NotificationStatus.HANDLED,
            address_book=book, notes="done",
        )
        assert updated.status is NotificationStatus.HANDLED
        assert updated.notes == "done"
        # Step 2 deleted the file and de-indexed the key.
        assert inbox.list() == []
        assert not index.contains("ext-1")

    def test_handled_with_rsvp_moves_to_target_queue(
        self,
        inbox: SessionInbox,
        index: InFlightIndex,
        book: AddressBook,
        forge_queue: NotificationQueue,
    ) -> None:
        posted = inbox.post(
            _spec(
                target=inbox.address,
                rsvp_to=forge_queue.address,
                external_key="ext-2",
            )
        )
        apply_handling_transition(
            inbox, posted.id, NotificationStatus.HANDLED,
            address_book=book, notes="done-rsvp",
        )
        # File has moved; inbox is empty, forge queue holds it with
        # its terminal status preserved.
        assert inbox.list() == []
        forge_items = forge_queue.list()
        assert [n.id for n in forge_items] == [posted.id]
        assert forge_items[0].status is NotificationStatus.HANDLED
        assert forge_items[0].notes == "done-rsvp"
        # Key stays in the index -- the forge queue still has it.
        assert index.contains("ext-2")

    def test_handled_with_unregistered_rsvp_raises(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        missing = _service_addr("no-such-service")
        posted = inbox.post(
            _spec(target=inbox.address, rsvp_to=missing)
        )
        with pytest.raises(DispatchError):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.HANDLED,
                address_book=book,
            )
        # Step 1 landed even though step 2 failed -- the item is
        # still in the inbox with the terminal status.
        leftover = inbox.get(posted.id)
        assert leftover.status is NotificationStatus.HANDLED


# ---------------------------------------------------------------------------
# ERRORED transitions
# ---------------------------------------------------------------------------

class TestErrored:
    def test_errored_requires_reason(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        with pytest.raises(ValueError, match="error_reason is required"):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.ERRORED,
                address_book=book,
            )

    def test_errored_rejects_notes(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        with pytest.raises(ValueError, match="notes"):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.ERRORED,
                address_book=book, notes="x", error_reason="y",
            )

    def test_errored_without_rsvp_moves_to_errored_dir(
        self, inbox: SessionInbox, index: InFlightIndex, book: AddressBook,
        tmp_path: Path,
    ) -> None:
        posted = inbox.post(
            _spec(target=inbox.address, external_key="ext-3")
        )
        apply_handling_transition(
            inbox, posted.id, NotificationStatus.ERRORED,
            address_book=book, error_reason="could not do it",
        )
        assert inbox.list() == []
        # File should now be in inbox/errored/.
        errored_root = tmp_path / "inbox" / "errored"
        assert errored_root.is_dir()
        errored_queue = DurableQueue(errored_root)
        items = errored_queue.list()
        assert [n.id for n in items] == [posted.id]
        assert items[0].status is NotificationStatus.ERRORED
        assert items[0].error_reason == "could not do it"
        # Index retains the key until an operator clears the errored
        # file manually.
        assert index.contains("ext-3")

    def test_errored_with_rsvp_moves_to_target_queue(
        self,
        inbox: SessionInbox,
        index: InFlightIndex,
        book: AddressBook,
        forge_queue: NotificationQueue,
    ) -> None:
        posted = inbox.post(
            _spec(
                target=inbox.address,
                rsvp_to=forge_queue.address,
                external_key="ext-4",
            )
        )
        apply_handling_transition(
            inbox, posted.id, NotificationStatus.ERRORED,
            address_book=book, error_reason="gave up",
        )
        assert inbox.list() == []
        items = forge_queue.list()
        assert [n.id for n in items] == [posted.id]
        assert items[0].status is NotificationStatus.ERRORED
        assert items[0].error_reason == "gave up"
        assert index.contains("ext-4")


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class TestGuards:
    def test_rejects_pending_target_status(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        with pytest.raises(ValueError, match="does not accept"):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.PENDING,
                address_book=book,
            )

    def test_rejects_confirmed_target_status(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        with pytest.raises(ValueError, match="does not accept"):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.CONFIRMED,
                address_book=book,
            )

    def test_rejects_second_terminal_transition(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        # Pre-mark the item handled via raw queue operations so that
        # apply_handling_transition sees it in a terminal state.  We
        # do this instead of first calling apply_handling_transition
        # because step 2 of the first call deletes the file.
        posted = inbox.post(_spec(target=inbox.address))
        inbox.update_status(posted.id, NotificationStatus.HANDLED, notes="x")
        with pytest.raises(ValueError, match="already"):
            apply_handling_transition(
                inbox, posted.id, NotificationStatus.HANDLED,
                address_book=book,
            )

    def test_unknown_notification_raises_key_error(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        with pytest.raises(KeyError):
            apply_handling_transition(
                inbox, "NO-SUCH-ID", NotificationStatus.IN_PROGRESS,
                address_book=book,
            )


# ---------------------------------------------------------------------------
# dispatch_step_two on its own (sweep-style use)
# ---------------------------------------------------------------------------

class TestDispatchStepTwoAlone:
    def test_requires_terminal_status(
        self, inbox: SessionInbox, book: AddressBook,
    ) -> None:
        posted = inbox.post(_spec(target=inbox.address))
        with pytest.raises(ValueError, match="terminal status"):
            dispatch_step_two(inbox, posted, address_book=book)

    def test_recovers_stuck_handled_no_rsvp(
        self, inbox: SessionInbox, index: InFlightIndex, book: AddressBook,
    ) -> None:
        # Simulate a crash between step 1 and step 2: step 1 landed
        # but the dispatch never ran.  The sweep discovers the stuck
        # handled item and calls dispatch_step_two directly.
        posted = inbox.post(
            _spec(target=inbox.address, external_key="ext-5")
        )
        stuck = inbox.update_status(
            posted.id, NotificationStatus.HANDLED, notes="done",
        )
        dispatch_step_two(inbox, stuck, address_book=book)
        assert inbox.list() == []
        assert not index.contains("ext-5")

    def test_recovers_stuck_handled_with_rsvp(
        self,
        inbox: SessionInbox,
        book: AddressBook,
        forge_queue: NotificationQueue,
    ) -> None:
        posted = inbox.post(
            _spec(
                target=inbox.address,
                rsvp_to=forge_queue.address,
            )
        )
        stuck = inbox.update_status(
            posted.id, NotificationStatus.HANDLED, notes="done",
        )
        dispatch_step_two(inbox, stuck, address_book=book)
        assert inbox.list() == []
        assert [n.id for n in forge_queue.list()] == [posted.id]
