"""Startup sweep: crash-recovery across every queue in the agency.

The sweep runs once on runtime entry, before any scheduler starts or
any source begins polling.  Its job is to bring the filesystem into a
state that matches the invariants the running system expects.  For
each notification file found in any queue, it handles every category
of "left-over" state that a previous process could have abandoned:

Session inboxes:

- **Temp sidecars (``.tmp-*.json``)** -- deleted.  These are partial
  writes from a crash during ``DurableQueue._write_atomic``; no live
  state depends on them.
- **``status == in_progress``** -- left alone.  On the session-inbox
  side, ``prompt_pending`` already treats ``pending`` and
  ``in_progress`` identically, so reverting would have no effect on
  scheduling.  The alternative -- presenting the agent with a fresh
  item on restart that it never saw in-progress -- would be just as
  confusing as the status surviving the crash (and arguably more so,
  since the notification's ``notes`` field may contain context the
  prior agent incarnation left for itself).  We deliberately let the
  on-disk status stand and rely on the session activation pass in
  :meth:`Gateway._startup` to ensure the driver still runs.
- **``status == handled`` or ``status == errored``** -- step 2
  re-driven.  The previous process landed step 1 (status update)
  but crashed before step 2 (move or delete).  The sweep calls
  :func:`thorn.runtime._dispatch.dispatch_step_two` which either
  deletes (handled, no RSVP), moves to the RSVP target (either
  terminal status with RSVP), or moves to ``errored/`` (errored, no
  RSVP).  RSVP-target resolution failures are logged and skipped;
  the item stays where it is for a subsequent sweep.
- **``status == confirmed``** -- a ``confirmed`` item in a session
  inbox is not expected to occur under normal operation; it is
  treated like a stuck handled item (deleted) and counted in a
  dedicated field so operators notice.

Service queues (:class:`~thorn.runtime._notification_queue.NotificationQueue`):

- **Temp sidecars** -- deleted.
- **``status == in_progress``** -- reverted to ``pending`` so the
  next drain picks the item up again.
- **``status == confirmed``** -- handler completed step 1 (mutate
  to confirmed) but crashed before step 2 (delete).  The sweep
  just runs step 2.
- **``status == handled`` / ``errored``** -- never expected in a
  receiver queue (those live in session inboxes during the
  mid-dispatch window and get *moved* to the receiver by step 2);
  if seen, they are counted and left alone because the sweep has
  no sensible action to take.

The sweep returns a :class:`SweepReport` summarizing exactly what it
did.  Idempotence: running the sweep a second time on the same
filesystem state is a no-op (all fixups converge on steady-state).

All queue access goes through the same :class:`AddressBook` that the
runtime hands to the dispatcher, so queues receive their registered
:class:`InFlightIndex` (if any) and the index stays consistent with
the filesystem across recovery operations.  The sweep does not build
its own queue instances except for an ephemeral
:class:`~thorn.runtime._inbox.SessionInbox` per discovered session
directory -- session inboxes live under per-session paths that may
not yet have been registered in the book at sweep time.  The
:class:`~thorn.runtime._in_flight_index.InFlightIndex` is optional
and, when supplied, is shared with those ephemeral inbox instances
so that delete/post updates land in the same index the rest of the
runtime uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from thorn.runtime._address import (
    AddressBook,
    ServiceAddress,
    SessionAddress,
)
from thorn.runtime._dispatch import DispatchError, dispatch_step_two
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import NotificationStatus
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._paths import AgencyPaths

if TYPE_CHECKING:
    from thorn.runtime._in_flight_index import InFlightIndex


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep report
# ---------------------------------------------------------------------------

@dataclass
class SweepReport:
    """Tally of everything the startup sweep did.

    Counters are per-category and always non-negative.  Aggregating
    them across multiple sweep runs is meaningful (a second sweep
    over a clean filesystem yields all zeros).  Separate fields for
    session-inbox work versus service-queue work make operator-facing
    logs easy to skim.
    """

    temp_files_removed: int = 0
    """``.tmp-*.json`` sidecars cleaned up across all queues."""

    session_handled_dispatched: int = 0
    """``handled`` session-inbox items whose step 2 the sweep ran."""

    session_errored_dispatched: int = 0
    """``errored`` session-inbox items whose step 2 the sweep ran."""

    session_confirmed_cleaned: int = 0
    """``confirmed``-in-session-inbox items the sweep deleted.

    Treated as stuck ``handled`` equivalents; not expected under
    normal operation.
    """

    service_in_progress_reverted: int = 0
    """``in_progress`` service-queue items reverted to ``pending``."""

    service_confirmed_cleaned: int = 0
    """``confirmed`` service-queue items whose step 2 (delete) the
    sweep ran."""

    service_misplaced: int = 0
    """``handled``/``errored`` items observed in a service queue.

    These should not occur during normal operation and are left
    untouched; surfaced as a counter so monitoring can notice.
    """

    dispatch_skipped_unresolved: list[str] = field(default_factory=list)
    """IDs of ``handled``/``errored`` items whose ``rsvp_to`` address
    did not resolve in the address book.  Items stay on disk for a
    later sweep to pick up once the target is registered."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_startup_sweep(
    paths: AgencyPaths,
    address_book: AddressBook,
    *,
    in_flight_index: "InFlightIndex | None" = None,
) -> SweepReport:
    """Walk every inbox and service queue; fix any crash-leftover state.

    See the module docstring for the exhaustive list of states and
    their fixups.

    Arguments:
        paths: agency layout used to discover queue directories.
        address_book: used for RSVP target resolution during step 2.
            Queues for sessions are re-registered transparently by
            this sweep (because session inboxes may not yet be in
            the book on early startup); service queues are expected
            to have been registered by the runtime's service-load
            step before the sweep runs.
        in_flight_index: optional shared index.  When supplied, the
            sweep constructs its ephemeral inbox/queue instances
            with this index, so that deletes during step 2 update
            the same index the rest of the runtime uses.  Callers
            that run :func:`rebuild_in_flight_index` *after* the
            sweep can leave this as ``None`` -- the index rebuild
            will regenerate from whatever the sweep left on disk.

    Returns a :class:`SweepReport` summarizing the work done.
    """
    report = SweepReport()
    _sweep_session_inboxes(paths, address_book, in_flight_index, report)
    _sweep_service_queues(paths, address_book, in_flight_index, report)
    return report


# ---------------------------------------------------------------------------
# Session inbox sweep
# ---------------------------------------------------------------------------

def _sweep_session_inboxes(
    paths: AgencyPaths,
    address_book: AddressBook,
    in_flight_index: "InFlightIndex | None",
    report: SweepReport,
) -> None:
    for agent_id, session_key, inbox_dir in paths.iter_session_inbox_locations():
        address = SessionAddress(agent_id, session_key)
        inbox = SessionInbox(
            inbox_dir, address, in_flight_index=in_flight_index,
        )
        report.temp_files_removed += inbox.cleanup_temp_files()

        for notification in inbox.list():
            status = notification.status

            if status is NotificationStatus.IN_PROGRESS:
                # Deliberately left as-is; see module docstring.  The
                # session activation pass in Gateway._startup is
                # responsible for making sure the driver runs.
                continue

            if status in (
                NotificationStatus.HANDLED,
                NotificationStatus.ERRORED,
            ):
                # Load current on-disk notification (not the one from
                # the initial listing, which is fine here too since
                # nothing else mutates during a sweep).
                try:
                    dispatch_step_two(
                        inbox, notification, address_book=address_book,
                    )
                except DispatchError as exc:
                    log.warning(
                        "Sweep leaving notification %s in session inbox "
                        "%s: %s", notification.id, address, exc,
                    )
                    report.dispatch_skipped_unresolved.append(
                        notification.id,
                    )
                    continue
                if status is NotificationStatus.HANDLED:
                    report.session_handled_dispatched += 1
                else:
                    report.session_errored_dispatched += 1
                continue

            if status is NotificationStatus.CONFIRMED:
                # Shouldn't occur in a session inbox under normal
                # operation.  Clean it up -- the file represents a
                # notification whose lifecycle has already completed
                # elsewhere; keeping it here serves no purpose.
                inbox.delete(notification.id)
                report.session_confirmed_cleaned += 1
                continue

            # PENDING: healthy steady-state; no sweep action needed.


# ---------------------------------------------------------------------------
# Service queue sweep
# ---------------------------------------------------------------------------

def _sweep_service_queues(
    paths: AgencyPaths,
    address_book: AddressBook,
    in_flight_index: "InFlightIndex | None",
    report: SweepReport,
) -> None:
    for service_name, queue_dir in paths.iter_service_queue_locations():
        service_address = ServiceAddress(service_name)
        # Try to reuse the registered queue so that any handlers
        # attached to it stay with us; fall back to an ephemeral
        # instance (no handlers registered, which is fine for a
        # mechanical sweep since we only perform status/delete ops).
        queue = address_book.get(service_address)
        if not isinstance(queue, NotificationQueue):
            queue = NotificationQueue(
                queue_dir,
                service_address,
                in_flight_index=in_flight_index,
            )
        report.temp_files_removed += queue.cleanup_temp_files()

        for notification in queue.list():
            status = notification.status

            if status is NotificationStatus.IN_PROGRESS:
                queue.update_status(
                    notification.id, NotificationStatus.PENDING,
                )
                report.service_in_progress_reverted += 1
                continue

            if status is NotificationStatus.CONFIRMED:
                # Handler's step 1 landed before crash; just run the
                # step-2 delete.  The delete uses the queue's index
                # wiring to de-index the external key.
                queue.delete(notification.id)
                report.service_confirmed_cleaned += 1
                continue

            if status in (
                NotificationStatus.HANDLED,
                NotificationStatus.ERRORED,
            ):
                # Not expected: handled/errored live on the session
                # inbox side, not here.  Leave alone; count for
                # visibility.
                log.warning(
                    "Sweep found misplaced %s notification %s in "
                    "service queue %s; leaving in place for manual "
                    "triage", status.value, notification.id,
                    service_address,
                )
                report.service_misplaced += 1
                continue

            # PENDING: healthy steady-state.


__all__ = [
    "SweepReport",
    "run_startup_sweep",
]
