"""Session inbox: the DurableQueue that drives a session's prompt loop.

A :class:`SessionInbox` is a thin specialization of
:class:`~thorn.runtime._queue.DurableQueue` attached to a specific
:class:`~thorn.runtime._address.SessionAddress`.  Its additional role
over the base queue is categorizing its contents by where the
notification is in the two-step handling lifecycle:

- **Prompt-ready** (``pending`` or ``in_progress``): items the agent
  should see on its next prompt.
- **Awaiting dispatch** (``handled`` or ``errored``): items the agent
  has already closed out but whose step-2 move/delete hasn't yet
  run -- typically only observed during the startup recovery sweep.
- **Awaiting cleanup** (``confirmed``): items that finished their
  RSVP round-trip but whose delete didn't land.  Session inboxes
  should never produce these directly, but the primitive makes them
  representable so tests and sweeps can exercise the full lifecycle
  uniformly.

Scheduling, prompt construction, and the actual mechanics of
transitioning items through their lifecycle all live at higher
layers.  :class:`SessionInbox` owns only the "which notifications
belong in which slice of the lifecycle" question.
"""

from __future__ import annotations

from pathlib import Path

from thorn.runtime._address import SessionAddress
from thorn.runtime._notification import Notification, NotificationStatus
from thorn.runtime._queue import DurableQueue


class SessionInbox(DurableQueue):
    """A session's durable inbox.

    Attached to a :class:`SessionAddress` so the rest of the runtime
    can identify which session a given queue belongs to without
    round-tripping through the address book.
    """

    def __init__(self, root_dir: Path, address: SessionAddress) -> None:
        super().__init__(root_dir)
        self._address = address

    @property
    def address(self) -> SessionAddress:
        """The session address this inbox belongs to."""
        return self._address

    # ------------------------------------------------------------------
    # Lifecycle-aware views

    def prompt_pending(self) -> list[Notification]:
        """Return items the scheduler should present to the agent.

        Includes everything in ``pending`` or ``in_progress`` status,
        in post order.  Items in ``handled``, ``errored``, or
        ``confirmed`` status are deliberately excluded -- they have
        already been closed out by the agent and are awaiting
        downstream dispatch / cleanup.
        """
        return self.list(status=(
            NotificationStatus.PENDING,
            NotificationStatus.IN_PROGRESS,
        ))

    def awaiting_dispatch(self) -> list[Notification]:
        """Return items the agent has closed out but not yet dispatched.

        These are items with ``status`` ``handled`` or ``errored``
        still physically present in the inbox directory -- step 1 of
        the two-step handling flow ran, but step 2 (move to RSVP
        target / delete / move to errored) did not.  The startup
        sweep uses this to recover from a crash between the two
        steps.
        """
        return self.list(status=(
            NotificationStatus.HANDLED,
            NotificationStatus.ERRORED,
        ))

    def awaiting_cleanup(self) -> list[Notification]:
        """Return ``confirmed``-status items.

        A session inbox should never contain these under normal
        operation, because confirmed only happens on the RSVP
        recipient's side.  The method exists so the startup sweep
        can enumerate every pathological state uniformly.
        """
        return self.list(status=NotificationStatus.CONFIRMED)


__all__ = [
    "SessionInbox",
]
