"""Unit tests for ``run_startup_sweep`` and :class:`SweepReport`.

Each test sets up a specific post-crash filesystem state and asserts
that the sweep transitions the filesystem to the expected
steady-state while tallying the work correctly.
"""

from __future__ import annotations

from pathlib import Path

from thorn.runtime._address import (
    Address,
    AddressBook,
    ServiceAddress,
    SessionAddress,
)
from thorn.runtime._in_flight_index import InFlightIndex
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._queue import DurableQueue
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._sweep import run_startup_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paths(tmp_path: Path) -> AgencyPaths:
    return AgencyPaths(
        home_root=tmp_path / "home",
        workspace_root=tmp_path / "ws",
    )


def _spec(
    *,
    target: Address,
    rsvp_to: Address | None = None,
    external_key: str | None = None,
) -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content="body",
        target=target,
        metadata={},
        rsvp_to=rsvp_to,
        external_key=external_key,
    )


def _make_session_inbox(
    paths: AgencyPaths,
    agent_id: str,
    session_key: str,
    *,
    index: InFlightIndex | None = None,
) -> SessionInbox:
    addr = SessionAddress(AgentID(agent_id), SessionKey(session_key))
    return SessionInbox(
        paths.session_inbox_dir(addr.agent_id, addr.session_key),
        addr,
        in_flight_index=index,
    )


def _make_service_queue(
    paths: AgencyPaths,
    service_name: str,
    *,
    index: InFlightIndex | None = None,
) -> NotificationQueue:
    addr = ServiceAddress(service_name)
    return NotificationQueue(
        paths.service_queue_dir(service_name),
        addr,
        in_flight_index=index,
    )


# ---------------------------------------------------------------------------
# Session inbox recovery
# ---------------------------------------------------------------------------

class TestSessionInboxSweep:
    def test_in_progress_is_left_untouched(self, tmp_path: Path) -> None:
        """The sweep intentionally leaves ``in_progress`` items alone.

        ``SessionInbox.prompt_pending`` treats ``pending`` and
        ``in_progress`` identically, so nothing in the scheduling
        machinery depends on a revert.  The gateway's session
        activation pass (not the sweep) is what re-wakes the driver.
        Keeping the status means we also preserve any ``notes`` the
        prior agent incarnation wrote for its future self.
        """
        paths = _paths(tmp_path)
        book = AddressBook()
        inbox = _make_session_inbox(paths, "coord", "proj/1")
        posted = inbox.post(_spec(target=inbox.address))
        inbox.update_status(posted.id, NotificationStatus.IN_PROGRESS)

        run_startup_sweep(paths, book)

        surviving = inbox.get(posted.id)
        assert surviving.status is NotificationStatus.IN_PROGRESS

    def test_stuck_handled_no_rsvp_is_deleted(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        index = InFlightIndex()
        inbox = _make_session_inbox(paths, "coord", "proj/2", index=index)
        posted = inbox.post(
            _spec(target=inbox.address, external_key="ek-1")
        )
        inbox.update_status(posted.id, NotificationStatus.HANDLED, notes="x")
        assert index.contains("ek-1")

        report = run_startup_sweep(paths, book, in_flight_index=index)

        assert report.session_handled_dispatched == 1
        assert inbox.list() == []
        assert not index.contains("ek-1")

    def test_stuck_handled_with_rsvp_is_moved_to_target(
        self, tmp_path: Path,
    ) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        forge = _make_service_queue(paths, "forge")
        book.register(forge.address, forge)

        inbox = _make_session_inbox(paths, "coord", "proj/3")
        posted = inbox.post(
            _spec(target=inbox.address, rsvp_to=forge.address)
        )
        inbox.update_status(posted.id, NotificationStatus.HANDLED, notes="done")

        report = run_startup_sweep(paths, book)

        assert report.session_handled_dispatched == 1
        assert inbox.list() == []
        assert [n.id for n in forge.list()] == [posted.id]

    def test_stuck_errored_no_rsvp_goes_to_errored_dir(
        self, tmp_path: Path,
    ) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        inbox = _make_session_inbox(paths, "coord", "proj/4")
        posted = inbox.post(_spec(target=inbox.address))
        inbox.update_status(
            posted.id, NotificationStatus.ERRORED, error_reason="boom",
        )

        report = run_startup_sweep(paths, book)

        assert report.session_errored_dispatched == 1
        assert inbox.list() == []
        errored_dir = paths.session_inbox_errored_dir(
            AgentID("coord"), SessionKey("proj/4"),
        )
        errored_q = DurableQueue(errored_dir)
        assert [n.id for n in errored_q.list()] == [posted.id]

    def test_stuck_handled_with_unresolved_rsvp_is_skipped(
        self, tmp_path: Path,
    ) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()  # nothing registered
        inbox = _make_session_inbox(paths, "coord", "proj/5")
        unregistered = ServiceAddress("ghost")
        posted = inbox.post(
            _spec(target=inbox.address, rsvp_to=unregistered)
        )
        inbox.update_status(posted.id, NotificationStatus.HANDLED, notes="n")

        report = run_startup_sweep(paths, book)

        # Dispatch failed; item left in place, report records the skip.
        assert report.session_handled_dispatched == 0
        assert posted.id in report.dispatch_skipped_unresolved
        surviving = inbox.get(posted.id)
        assert surviving.status is NotificationStatus.HANDLED

    def test_confirmed_in_session_inbox_is_cleaned(
        self, tmp_path: Path,
    ) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        inbox = _make_session_inbox(paths, "coord", "proj/6")
        posted = inbox.post(_spec(target=inbox.address))
        inbox.update_status(posted.id, NotificationStatus.CONFIRMED)

        report = run_startup_sweep(paths, book)

        assert report.session_confirmed_cleaned == 1
        assert inbox.list() == []

    def test_temp_sidecars_are_removed(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        inbox = _make_session_inbox(paths, "coord", "proj/7")
        inbox_dir = paths.session_inbox_dir(
            AgentID("coord"), SessionKey("proj/7"),
        )
        inbox_dir.mkdir(parents=True, exist_ok=True)
        (inbox_dir / ".tmp-ORPHAN.json").write_text("{}", encoding="utf-8")

        report = run_startup_sweep(paths, book)

        assert report.temp_files_removed == 1
        assert not (inbox_dir / ".tmp-ORPHAN.json").exists()

    def test_pending_items_are_left_alone(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        inbox = _make_session_inbox(paths, "coord", "proj/8")
        posted = inbox.post(_spec(target=inbox.address))

        report = run_startup_sweep(paths, book)

        # A pending item triggers no sweep action.
        assert report.session_handled_dispatched == 0
        surviving = inbox.get(posted.id)
        assert surviving.status is NotificationStatus.PENDING


# ---------------------------------------------------------------------------
# Service queue recovery
# ---------------------------------------------------------------------------

class TestServiceQueueSweep:
    def test_in_progress_is_reverted_to_pending(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        queue = _make_service_queue(paths, "forge")
        book.register(queue.address, queue)
        posted = queue.post(_spec(target=queue.address))
        queue.update_status(posted.id, NotificationStatus.IN_PROGRESS)

        report = run_startup_sweep(paths, book)

        assert report.service_in_progress_reverted == 1
        assert queue.get(posted.id).status is NotificationStatus.PENDING

    def test_confirmed_is_deleted(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        index = InFlightIndex()
        queue = _make_service_queue(paths, "forge", index=index)
        book.register(queue.address, queue)
        posted = queue.post(
            _spec(target=queue.address, external_key="ek-svc")
        )
        queue.update_status(posted.id, NotificationStatus.CONFIRMED)

        report = run_startup_sweep(paths, book, in_flight_index=index)

        assert report.service_confirmed_cleaned == 1
        assert queue.list() == []
        assert not index.contains("ek-svc")

    def test_misplaced_handled_is_counted_but_left(
        self, tmp_path: Path,
    ) -> None:
        paths = _paths(tmp_path)
        book = AddressBook()
        queue = _make_service_queue(paths, "forge")
        book.register(queue.address, queue)
        posted = queue.post(_spec(target=queue.address))
        queue.update_status(posted.id, NotificationStatus.HANDLED, notes="x")

        report = run_startup_sweep(paths, book)

        assert report.service_misplaced == 1
        # File stays put for manual triage.
        assert queue.get(posted.id).status is NotificationStatus.HANDLED

    def test_ephemeral_queue_used_when_not_registered(
        self, tmp_path: Path,
    ) -> None:
        # Even if the service queue was never registered in the
        # address book, the sweep should still find its directory
        # and process it.
        paths = _paths(tmp_path)
        book = AddressBook()
        queue = _make_service_queue(paths, "stray")
        posted = queue.post(_spec(target=queue.address))
        queue.update_status(posted.id, NotificationStatus.IN_PROGRESS)

        report = run_startup_sweep(paths, book)

        assert report.service_in_progress_reverted == 1


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_second_sweep_is_noop(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    book = AddressBook()
    forge = _make_service_queue(paths, "forge")
    book.register(forge.address, forge)

    inbox = _make_session_inbox(paths, "coord", "proj/1")
    posted = inbox.post(
        _spec(target=inbox.address, rsvp_to=forge.address)
    )
    inbox.update_status(posted.id, NotificationStatus.HANDLED, notes="x")

    first = run_startup_sweep(paths, book)
    assert first.session_handled_dispatched == 1

    second = run_startup_sweep(paths, book)
    assert second.session_handled_dispatched == 0
    assert second.dispatch_skipped_unresolved == []
