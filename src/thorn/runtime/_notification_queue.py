"""NotificationQueue: a DurableQueue drained by a registered handler.

While :class:`~thorn.runtime._inbox.SessionInbox` is drained by the
LLM prompt loop (via the scheduler), a
:class:`NotificationQueue` is drained by plain Python: the owning
service or source registers one or both of a fresh-notification
handler and an RSVP handler, and calls :meth:`NotificationQueue.drain`
on whatever cadence is natural for its existing loop (e.g. as a step
in a source's poll cycle).

The drain flow classifies each notification by inspecting its fields:

- **Fresh** (``target == self.address``, ``status == pending``): a
  direct post from somewhere in the agency.  Dispatched to the fresh
  handler.
- **RSVP** (``target != self.address``, ``rsvp_to == self.address``,
  ``status in {handled, errored}``): a previously-posted notification
  that the original target has finished with and is returning for
  post-completion processing.  Dispatched to the RSVP handler.
- **Confirmed** (``status == confirmed``): step 1 of the recipient's
  two-step completion ran in an earlier pass but step 2 (delete)
  didn't land.  The queue just completes step 2; no handler is
  invoked.

Completion ceremony for fresh and RSVP arrivals is owned by the
framework, not the handler: on handler return-normally, the queue
writes ``status = confirmed`` and then deletes the file.  This
departs from the original plan wording ("handler internally does step
1 + step 2") on deliberate grounds -- see the explanation in the
Session Inbox Abstraction plan's discussion of completion ownership.
The short version: putting the ceremony in the framework makes
handlers simpler, keeps the two-step recovery semantics uniform
across handlers, and relies only on the idempotence property we
already need for handler retry anyway.

Handler errors are captured in :class:`DrainOutcome` and do **not**
propagate out of :meth:`drain`.  The queue itself has no retry /
backoff policy in Phase 1: on failure, the notification stays in
place and the caller decides whether to drain again.  Higher layers
(scheduler progress guarantee, source-specific retry) add structure
on top of this.

The queue does not spawn background tasks; the owner controls the
cadence.  Concurrency within a single queue is the caller's
responsibility: if two tasks call :meth:`drain` at the same time
they may both invoke a handler for the same notification.  Typical
usage serializes drain calls within a single service loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from thorn.runtime._address import Address
from thorn.runtime._notification import Notification, NotificationStatus
from thorn.runtime._queue import DurableQueue


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler types
# ---------------------------------------------------------------------------

NotificationHandler = Callable[[Notification], Awaitable[None]]
"""Callable signature for fresh and RSVP handlers.

Return-normally indicates success; the queue completes the two-step
cleanup (mutate to ``confirmed``, then delete) automatically.  Raising
any exception indicates failure; the queue leaves the notification in
place and reports the failure in the :class:`DrainOutcome`.
"""


class ArrivalKind(Enum):
    """How a notification arrived at its current :class:`NotificationQueue`.

    - :attr:`FRESH` -- the notification was addressed directly here.
    - :attr:`RSVP` -- the notification was originally sent elsewhere,
      and is returning because its ``rsvp_to`` field points here.
    """

    FRESH = "fresh"
    RSVP = "rsvp"


class DrainResult(Enum):
    """High-level outcome category for a single drain step."""

    HANDLED = "handled"
    """Handler ran, returned normally; file has been removed."""

    RECOVERED = "recovered"
    """Status was already ``confirmed`` on entry; step-2 delete was run
    without invoking any handler."""

    HANDLER_FAILED = "handler_failed"
    """Handler raised; file is still in the queue at its prior status."""

    NO_HANDLER = "no_handler"
    """No handler is registered for this arrival kind; file is still
    in the queue."""


@dataclass(frozen=True)
class DrainOutcome:
    """Result of processing one notification during a drain pass.

    Collected by :meth:`NotificationQueue.drain` and returned to the
    caller so that services can log, trigger retries, or surface
    errors.
    """

    notification_id: str
    kind: ArrivalKind
    result: DrainResult
    error: BaseException | None = None


# ---------------------------------------------------------------------------
# NotificationQueue
# ---------------------------------------------------------------------------

class NotificationQueue(DurableQueue):
    """DurableQueue drained by registered Python handlers.

    Construct with a *root_dir* (where notifications live) and an
    *address* (this queue's canonical identity, used to distinguish
    fresh vs RSVP arrivals).  Register at least one handler via
    :meth:`on_fresh` and/or :meth:`on_rsvp` before calling
    :meth:`drain`.

    The queue does not spawn background tasks; callers invoke
    :meth:`drain` explicitly.
    """

    def __init__(self, root_dir: Path, address: Address) -> None:
        super().__init__(root_dir)
        self._address = address
        self._fresh_handler: NotificationHandler | None = None
        self._rsvp_handler: NotificationHandler | None = None

    @property
    def address(self) -> Address:
        """The address that identifies this queue."""
        return self._address

    # ------------------------------------------------------------------
    # Handler registration

    def on_fresh(self, handler: NotificationHandler) -> None:
        """Register the handler for fresh arrivals.

        Raises ``ValueError`` if a fresh handler is already registered.
        Use :meth:`clear_handlers` to replace deliberately.
        """
        if self._fresh_handler is not None:
            raise ValueError(
                f"Fresh handler already registered for {self._address}"
            )
        self._fresh_handler = handler

    def on_rsvp(self, handler: NotificationHandler) -> None:
        """Register the handler for RSVP arrivals.

        Raises ``ValueError`` if an RSVP handler is already registered.
        Use :meth:`clear_handlers` to replace deliberately.
        """
        if self._rsvp_handler is not None:
            raise ValueError(
                f"RSVP handler already registered for {self._address}"
            )
        self._rsvp_handler = handler

    def clear_handlers(self) -> None:
        """Forget all registered handlers.

        Intended for tests and for deliberate handler replacement; not
        expected on a steady-state operational path.
        """
        self._fresh_handler = None
        self._rsvp_handler = None

    # ------------------------------------------------------------------
    # Classification

    def classify(self, notification: Notification) -> ArrivalKind:
        """Return the arrival kind for *notification* at this queue.

        Raises ``ValueError`` if the notification is neither addressed
        to this queue nor RSVP'd to it, or if its status is
        ``confirmed`` (confirmed items are handled by the recovery
        path, not classification).
        """
        if notification.status is NotificationStatus.CONFIRMED:
            raise ValueError(
                f"Notification {notification.id} is already confirmed; "
                f"classification does not apply"
            )
        if notification.target == self._address:
            return ArrivalKind.FRESH
        if notification.rsvp_to == self._address:
            return ArrivalKind.RSVP
        raise ValueError(
            f"Notification {notification.id} is neither addressed "
            f"(target={notification.target}) nor RSVP'd "
            f"(rsvp_to={notification.rsvp_to}) to {self._address}"
        )

    # ------------------------------------------------------------------
    # Drain

    async def drain(self) -> list[DrainOutcome]:
        """Process every notification currently in the queue.

        Returns a list of :class:`DrainOutcome`, one per item
        encountered.  Handler exceptions are captured in the outcomes
        and do not propagate.

        Notifications whose status is ``in_progress`` are left alone
        on this code path -- they are assumed to be owned by a concurrent
        in-flight handler call; the startup sweep is responsible for
        reverting any orphaned ``in_progress`` items to ``pending``.
        """
        outcomes: list[DrainOutcome] = []
        for notification in self.list():
            if notification.status is NotificationStatus.IN_PROGRESS:
                # Skip; a concurrent handler owns this. The sweep
                # recovers from genuinely-orphaned in_progress state.
                continue
            outcomes.append(await self._drain_one(notification))
        return outcomes

    async def _drain_one(self, notification: Notification) -> DrainOutcome:
        # Recovery path: a prior handler completed step 1 but not
        # step 2 before a crash.  Finish step 2; no handler invocation.
        if notification.status is NotificationStatus.CONFIRMED:
            kind = (
                ArrivalKind.RSVP
                if notification.rsvp_to == self._address
                else ArrivalKind.FRESH
            )
            self.delete(notification.id)
            return DrainOutcome(
                notification_id=notification.id,
                kind=kind,
                result=DrainResult.RECOVERED,
            )

        kind = self.classify(notification)
        handler = (
            self._fresh_handler if kind is ArrivalKind.FRESH
            else self._rsvp_handler
        )
        if handler is None:
            log.warning(
                "No %s handler registered for %s; leaving notification %s in place",
                kind.value, self._address, notification.id,
            )
            return DrainOutcome(
                notification_id=notification.id,
                kind=kind,
                result=DrainResult.NO_HANDLER,
            )

        try:
            await handler(notification)
        except Exception as exc:
            log.exception(
                "Handler for %s at %s raised on notification %s",
                kind.value, self._address, notification.id,
            )
            return DrainOutcome(
                notification_id=notification.id,
                kind=kind,
                result=DrainResult.HANDLER_FAILED,
                error=exc,
            )

        # Framework-owned two-step completion.  If the mutate succeeds
        # but the delete crashes (or the mutate itself crashes after
        # the rename), the startup sweep takes over: a confirmed item
        # in this directory is the next-start signal for "run step 2".
        self.update_status(notification.id, NotificationStatus.CONFIRMED)
        self.delete(notification.id)
        return DrainOutcome(
            notification_id=notification.id,
            kind=kind,
            result=DrainResult.HANDLED,
        )


__all__ = [
    "ArrivalKind",
    "DrainOutcome",
    "DrainResult",
    "NotificationHandler",
    "NotificationQueue",
]
