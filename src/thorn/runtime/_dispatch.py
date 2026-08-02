"""Two-step handling dispatch for session inbox items.

The session inbox abstraction splits the lifecycle of a handled or
errored notification into two individually-atomic filesystem steps:

1. **Mutate in place** -- write a new JSON representation of the
   notification with the updated ``status`` (and ``notes`` /
   ``error_reason``) to a temp sidecar, then ``rename(2)`` over the
   live file.  After this step the notification is durably recorded
   as handled or errored, but the file still lives in the session's
   inbox directory.

2. **Dispatch** -- move or delete the file depending on whether the
   notification requested an RSVP and which terminal status it
   reached:

   - ``status == HANDLED``, no RSVP: delete the file; the inbox's
     ``in_flight_index`` (if present) is decremented.
   - ``status == HANDLED``, with RSVP: atomically rename the file
     into the RSVP target's queue directory.
   - ``status == ERRORED``, no RSVP: atomically rename into the
     inbox's ``errored/`` sibling for operator inspection.  The
     in-flight key is **not** removed: an errored item without RSVP
     is not "done in flight" from the originator's perspective.
   - ``status == ERRORED``, with RSVP: atomically rename into the
     RSVP target's queue directory (same as handled+RSVP; the
     recipient decides what to do with the error).

Both steps are individually atomic via ``rename(2)``; the
in-between state (a ``handled``/``errored`` file still sitting in a
session inbox) is explicitly tolerated and recovered from by the
startup sweep.  See :mod:`thorn.runtime._sweep`.

:func:`apply_handling_transition` is the shared transition entry point
used by the focused inbox tools (:mod:`thorn.runtime._inbox_tools`).
:func:`dispatch_step_two` is factored out so the sweep can re-drive
step 2 on items left stuck after a crash without re-running step 1.

The module also handles the ``IN_PROGRESS`` transition, which is
intentionally step-1-only: the agent claiming an item just updates
status in place; the item stays in the inbox and is still visible to
the next prompt (via :meth:`SessionInbox.prompt_pending`) with its
new status.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from thorn.runtime._address import AddressBook
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    InboxCompletionRationale,
    Notification,
    NotificationStatus,
)
from thorn.runtime._queue import DurableQueue

if TYPE_CHECKING:
    pass


log = logging.getLogger(__name__)


# Terminal statuses that trigger step 2 dispatch after step 1 lands.
_TERMINAL_STATUSES: frozenset[NotificationStatus] = frozenset({
    NotificationStatus.HANDLED,
    NotificationStatus.ERRORED,
})

# Statuses the runtime tool layer may transition into on an agent's behalf.
_AGENT_VISIBLE_TRANSITIONS: frozenset[NotificationStatus] = frozenset({
    NotificationStatus.IN_PROGRESS,
    NotificationStatus.HANDLED,
    NotificationStatus.ERRORED,
})


class DispatchError(Exception):
    """Raised by step 2 when the dispatch target cannot be resolved.

    Typical causes: the notification has an ``rsvp_to`` address that
    is not registered in the supplied :class:`AddressBook`.  Callers
    that want best-effort behavior (e.g. the startup sweep) catch
    this, log, and leave the item in place for a later sweep to pick
    up after the missing queue is registered.
    """


# ---------------------------------------------------------------------------
# Step 1 + step 2 combined (agent-triggered path)
# ---------------------------------------------------------------------------

def apply_handling_transition(
    inbox: SessionInbox,
    notification_id: str,
    status: NotificationStatus,
    *,
    address_book: AddressBook,
    notes: str | None = None,
    completion_rationale: InboxCompletionRationale | None = None,
    error_reason: str | None = None,
) -> Notification:
    """Transition a session-inbox item into *status* and dispatch.

    Always performs step 1 (atomic in-place status + note update).
    For terminal statuses (``HANDLED``, ``ERRORED``) additionally
    performs step 2 (move or delete) using *address_book* to resolve
    any RSVP target.

    Arguments:
        inbox: the session inbox the item currently lives in.
        notification_id: ULID of the notification to transition.
        status: the target status.  Must be one of ``IN_PROGRESS``,
            ``HANDLED``, or ``ERRORED`` -- these are the only
            transitions the agent drives; ``PENDING`` and
            ``CONFIRMED`` are framework-owned.
        address_book: registry used to resolve ``rsvp_to`` addresses
            for terminal transitions.  Unused for ``IN_PROGRESS``.
        notes: free-form annotation.  For ``IN_PROGRESS`` and
            ``HANDLED`` it populates :attr:`Notification.notes`; for
            ``ERRORED`` the caller should use *error_reason* instead
            (passing *notes* with ``ERRORED`` is an error).
        completion_rationale: structured completion evidence required
            for ``HANDLED`` transitions.
        error_reason: required for ``ERRORED``; rejected otherwise.

    Returns the updated :class:`Notification` as written to disk.

    Raises:
        ValueError: if *status* is not an agent-visible transition,
            if *error_reason* usage is inconsistent with *status*, or
            if *completion_rationale* is missing for ``HANDLED`` or
            otherwise inconsistent with *status*, or
            if the notification is already in a terminal status (the
            agent's tool layer is expected to normalize this, but a
            defensive check here keeps bad transitions out of the
            filesystem).
        KeyError: if *notification_id* is not present in *inbox*.
        DispatchError: from step 2, if the RSVP target cannot be
            resolved via *address_book*.  When this happens, step 1
            has already landed on disk; the item is in the inbox
            with a terminal status and will be picked up by the next
            startup sweep (which shares the same step-2 logic).
    """
    if status not in _AGENT_VISIBLE_TRANSITIONS:
        raise ValueError(
            f"apply_handling_transition does not accept {status}; "
            f"only {sorted(s.value for s in _AGENT_VISIBLE_TRANSITIONS)}"
        )
    if status is NotificationStatus.ERRORED:
        if not error_reason:
            raise ValueError(
                "error_reason is required when transitioning to ERRORED"
            )
        if notes is not None:
            raise ValueError(
                "Pass error explanation via error_reason, not notes, "
                "for ERRORED transitions"
            )
        if completion_rationale is not None:
            raise ValueError(
                "completion_rationale is only valid for HANDLED transitions"
            )
    else:
        if error_reason is not None:
            raise ValueError(
                f"error_reason is only valid for ERRORED; got status={status}"
            )
        if status is NotificationStatus.HANDLED and completion_rationale is None:
            raise ValueError(
                "completion_rationale is required when transitioning to HANDLED"
            )
        if (
            status is not NotificationStatus.HANDLED
            and completion_rationale is not None
        ):
            raise ValueError(
                "completion_rationale is only valid for HANDLED transitions"
            )
        if completion_rationale is not None:
            validation_errors = completion_rationale.validation_errors()
            if validation_errors:
                raise ValueError(
                    "completion_rationale is incomplete: "
                    + "; ".join(validation_errors)
                )

    current = inbox.get(notification_id)
    if current.status in _TERMINAL_STATUSES:
        raise ValueError(
            f"Cannot transition notification {notification_id}: already "
            f"in terminal status {current.status.value}"
        )
    if current.status is NotificationStatus.CONFIRMED:
        raise ValueError(
            f"Cannot transition notification {notification_id}: already "
            f"confirmed (framework-owned status)"
        )

    # Step 1: atomic in-place mutation.
    update_fields: dict[str, object] = {}
    if status is NotificationStatus.ERRORED:
        update_fields["error_reason"] = error_reason
        # ``notes`` was rejected above for ERRORED; keep whatever was
        # already set (typically None).
    else:
        # IN_PROGRESS or HANDLED: record the optional note.  Leave
        # the existing value untouched when the caller passes None so
        # a later round can re-use a previously-attached note.
        if notes is not None:
            update_fields["notes"] = notes
        if completion_rationale is not None:
            update_fields["completion_rationale"] = completion_rationale

    updated = inbox.update_status(notification_id, status, **update_fields)

    # Step 2 only applies to terminal statuses.
    if status not in _TERMINAL_STATUSES:
        return updated

    dispatch_step_two(inbox, updated, address_book=address_book)
    return updated


# ---------------------------------------------------------------------------
# Step 2 alone (also used by the startup sweep)
# ---------------------------------------------------------------------------

def dispatch_step_two(
    inbox: SessionInbox,
    notification: Notification,
    *,
    address_book: AddressBook,
) -> None:
    """Execute step 2 for *notification*, which must be terminal.

    The caller is expected to have either just completed step 1 (the
    agent-triggered path) or discovered a mid-crash item in the
    sweep.  Behavior:

    - ``HANDLED`` with no RSVP: ``inbox.delete(id)``.  The inbox's
      in-flight index (if any) removes the notification's external
      key as part of the delete.
    - ``HANDLED`` or ``ERRORED`` with RSVP: resolve the target queue
      via *address_book* and ``move_to`` the notification there.
      The in-flight key stays in the index because the item is
      still in flight (just on the RSVP recipient's side).
    - ``ERRORED`` with no RSVP: move the notification into the
      inbox's ``errored/`` sibling directory for operator
      inspection.  The in-flight key is **not** removed -- an
      RSVP-less errored item is stuck until an operator clears it.

    Idempotence: calling this twice on the same notification is not
    defined; callers should only call it once per notification per
    step-1 landing.  The sweep loads the current on-disk state
    before calling to ensure it hasn't already been dispatched.

    Raises:
        ValueError: if *notification*'s status is not terminal.
        DispatchError: if the notification has an ``rsvp_to`` that
            cannot be resolved via *address_book*.
    """
    if notification.status not in _TERMINAL_STATUSES:
        raise ValueError(
            f"dispatch_step_two requires a terminal status, got "
            f"{notification.status.value} for {notification.id}"
        )

    if notification.rsvp_to is not None:
        target = address_book.get(notification.rsvp_to)
        if target is None:
            raise DispatchError(
                f"Cannot dispatch notification {notification.id}: "
                f"rsvp_to={notification.rsvp_to} is not registered in "
                f"the address book"
            )
        inbox.move_to(notification.id, target)
        return

    # No RSVP -- the terminal outcome determines local disposition.
    if notification.status is NotificationStatus.HANDLED:
        inbox.delete(notification.id)
        return

    # ERRORED without RSVP: park in errored/ for operator inspection.
    # We intentionally do NOT decrement the in-flight index here; the
    # external key stays in flight until somebody clears the errored
    # item manually.
    errored_root = inbox.root_dir / "errored"
    # Use a bare DurableQueue (no in-flight index wiring) for the
    # errored sibling: move_to is a pure rename and never touches the
    # index regardless, and constructing a full SessionInbox here
    # would wrongly advertise a new address for the same session.
    errored_queue = DurableQueue(errored_root)
    inbox.move_to(notification.id, errored_queue)


__all__ = [
    "DispatchError",
    "apply_handling_transition",
    "dispatch_step_two",
]
