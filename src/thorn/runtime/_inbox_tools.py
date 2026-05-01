"""Agent-facing tools for working with the current session's inbox.

These tools let an agent inspect and close out notifications that have
been posted to its :class:`~thorn.runtime._inbox.SessionInbox`.  Three
operations are exposed:

- :func:`list_inbox_items` -- summary of every item currently in the
  inbox (pending and in-progress).  Handled, errored, and confirmed
  items are filtered out because they represent closed-out work.
- :func:`read_inbox_item` -- full content and metadata of one item by
  its ULID.
- :func:`update_inbox_item` -- claim (``in_progress``), complete
  (``handled``), or give up on (``errored``) an item.  The status
  parameter is :class:`~typing.Literal`-typed so the LLM sees a
  genuine enum rather than a free-form string.

Resolution of the current session follows the journal-tool pattern:

- :data:`ExecutionContext.agent` gives the agent ID.
- ``scope.metadata["session_key"]`` (walking outermost-first) gives
  the session key.
- ``ExecutionContext.runtime.address_book`` resolves
  ``SessionAddress(agent_id, session_key)`` to a
  :class:`SessionInbox`.

When any part of this resolution fails (no ambient runtime, no
session scope, no registered inbox), the tool returns a clear
error-string rather than raising -- consistent with the journal
tools' behavior and easier for an LLM to react to.

These tools are automatically included in every agent's toolset via
``Agent._collect_tools`` (alongside the journal tools).
"""

from __future__ import annotations

import logging
from typing import Literal

from thorn.core._context import get_context
from thorn.core._executor import ToolVenue
from thorn.core._func import tool
from thorn.runtime._address import SessionAddress
from thorn.runtime._dispatch import (
    DispatchError,
    apply_handling_transition,
)
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import Notification, NotificationStatus


log = logging.getLogger(__name__)


# Allowed values for the agent-facing status parameter of
# ``update_inbox_item``.  Defined once as a module-level Literal so the
# tool signature and the docstring stay in sync; the concrete list is
# a subset of :class:`NotificationStatus` (pending and confirmed are
# framework-owned).
_AgentStatus = Literal["in_progress", "handled", "errored"]


# ---------------------------------------------------------------------------
# Current-session resolution helpers
# ---------------------------------------------------------------------------

def _current_session_inbox() -> SessionInbox | str:
    """Resolve the current scope's :class:`SessionInbox`.

    Returns the inbox on success, or a human-readable error string on
    failure.  The string form is designed to be returned directly
    from a tool body so the LLM sees a clear, actionable diagnostic
    instead of a stack trace.
    """
    try:
        ctx = get_context()
    except RuntimeError:
        return "Error: no active execution context. Inbox tools must be called from within an agent prompt."

    runtime = ctx.runtime
    if runtime is None:
        return "Error: no runtime is available. Inbox tools require a Runtime."

    agent = ctx.agent
    if agent is None or agent.id is None:
        return "Error: no agent is bound to the current scope. Inbox tools require an agent."

    # session_key lives in the scope metadata chain (see ``_run_session_prompt``).
    session_key: str | None = None
    scope = ctx.scope
    while scope is not None:
        key = scope.metadata.get("session_key")
        if key is not None:
            session_key = str(key)
            break
        scope = scope.outer
    if session_key is None:
        return "Error: no session is active. Inbox tools can only be used inside a session prompt."

    from thorn.runtime._session import SessionKey
    address = SessionAddress(agent.id, SessionKey(session_key))
    queue = runtime.address_book.get(address)
    if queue is None:
        return (
            f"Error: no inbox is registered for {address}. "
            "The runtime has not wired this session's inbox into the address book."
        )
    if not isinstance(queue, SessionInbox):
        return (
            f"Error: queue registered for {address} is not a SessionInbox "
            f"(got {type(queue).__name__})."
        )
    return queue


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

_SUMMARY_CHARS = 80
"""Maximum number of characters shown in a single-item summary."""


def _summarize(notification: Notification) -> str:
    """Return a compact, single-line summary of *notification*."""
    first_line = notification.content.split("\n", 1)[0].strip()
    if len(first_line) > _SUMMARY_CHARS:
        first_line = first_line[: _SUMMARY_CHARS - 1].rstrip() + "\u2026"
    return first_line or "(empty content)"


# ---------------------------------------------------------------------------
# Tool: list_inbox_items
# ---------------------------------------------------------------------------

@tool(venue=ToolVenue.IN_PROCESS)
async def list_inbox_items() -> str:
    """List notifications currently awaiting your attention in this session.

    Shows every item in ``pending`` or ``in_progress`` status.
    Items you have already marked ``handled`` or ``errored`` are
    filtered out -- they represent closed-out work.

    Each line carries enough information (ID, status, source, and a
    short summary) for you to decide which item to read in full.
    Use :func:`read_inbox_item` with an ID to see the complete
    content and metadata for an item.
    """
    inbox = _current_session_inbox()
    if isinstance(inbox, str):
        return inbox

    items = inbox.prompt_pending()
    if not items:
        return "Your inbox is empty."

    lines = [f"You have {len(items)} inbox item(s):"]
    for item in items:
        lines.append(
            f"- [{item.id}] status={item.status.value} "
            f"source={item.source}: {_summarize(item)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: read_inbox_item
# ---------------------------------------------------------------------------

@tool(venue=ToolVenue.IN_PROCESS)
async def read_inbox_item(item_id: str) -> str:
    """Read the full content and metadata of a single inbox item.

    Returns the item's ID, source, status, posted-at timestamp, any
    ``notes`` or ``error_reason`` attached to it, and its full
    textual content.  Use :func:`list_inbox_items` first to see
    which IDs are available.

    Args:
        item_id: The ULID of the item to read.  These are surfaced
            by :func:`list_inbox_items` in the form shown between
            ``[`` and ``]``.
    """
    inbox = _current_session_inbox()
    if isinstance(inbox, str):
        return inbox

    try:
        item = inbox.get(item_id)
    except KeyError:
        return f"Error: no inbox item with id {item_id!r}."

    lines = [
        f"Item: {item.id}",
        f"Source: {item.source}",
        f"Status: {item.status.value}",
        f"Posted: {item.posted_at.isoformat()}",
    ]
    if item.attempt_count > 0:
        lines.append(f"Attempts: {item.attempt_count}")
    if item.notes:
        lines.append(f"Notes: {item.notes}")
    if item.error_reason:
        lines.append(f"Error reason: {item.error_reason}")
    if item.metadata:
        lines.append(f"Metadata: {dict(item.metadata)!r}")
    lines.append("")
    lines.append("Content:")
    lines.append(item.content)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: update_inbox_item
# ---------------------------------------------------------------------------

@tool(venue=ToolVenue.IN_PROCESS)
async def update_inbox_item(
    item_id: str,
    status: _AgentStatus,
    notes: str = "",
) -> str:
    """Transition an inbox item into a new status.

    Status values:

    - ``in_progress``: you are actively working on this item.  The
      item stays in your inbox; future prompts will still show it,
      but marked as in-progress.  Useful when you will need multiple
      turns to finish.
    - ``handled``: you have completed this item.  It will be
      removed from your inbox (or forwarded to its original sender
      if an RSVP was requested).  Optionally attach *notes* to
      describe what you did; the notes travel to any RSVP
      recipient.
    - ``errored``: you cannot handle this item.  *notes* is
      **required** and should explain why.  The item will be moved
      to an ``errored/`` parking area for operator inspection, or
      forwarded to its original sender (with your explanation) if
      an RSVP was requested.

    Args:
        item_id: The ULID of the item.  See
            :func:`list_inbox_items` for available IDs.
        status: New status.  Must be one of ``in_progress``,
            ``handled``, or ``errored``.
        notes: Free-form explanation.  Required for ``errored``
            (and rejected if empty); optional for the other
            statuses.  When present, carried through to any RSVP
            recipient of this item.
    """
    inbox = _current_session_inbox()
    if isinstance(inbox, str):
        return inbox

    try:
        target_status = NotificationStatus(status)
    except ValueError:
        return (
            f"Error: invalid status {status!r}. "
            "Must be one of 'in_progress', 'handled', 'errored'."
        )

    try:
        ctx = get_context()
    except RuntimeError:
        return "Error: no active execution context."
    address_book = ctx.runtime.address_book

    try:
        if target_status is NotificationStatus.ERRORED:
            if not notes:
                return "Error: notes is required when marking an item as 'errored'; include a brief reason."
            updated = apply_handling_transition(
                inbox,
                item_id,
                target_status,
                address_book=address_book,
                error_reason=notes,
            )
        else:
            updated = apply_handling_transition(
                inbox,
                item_id,
                target_status,
                address_book=address_book,
                notes=notes or None,
            )
    except KeyError:
        return f"Error: no inbox item with id {item_id!r}."
    except ValueError as exc:
        return f"Error: {exc}"
    except DispatchError as exc:
        # Step 1 landed, step 2 did not.  The sweep will reconcile on
        # the next startup, but we should tell the agent that the
        # status change is visible while dispatch is stuck.
        return (
            f"Warning: item {item_id} is now marked {target_status.value}, "
            f"but dispatch to its RSVP target failed: {exc}. "
            "The item will be retried on next runtime start."
        )

    return f"Item {updated.id} is now {updated.status.value}."


INBOX_TOOLS: list = [list_inbox_items, read_inbox_item, update_inbox_item]
"""Default inbox tools added to every agent via ``Agent._collect_tools``."""


__all__ = [
    "INBOX_TOOLS",
    "list_inbox_items",
    "read_inbox_item",
    "update_inbox_item",
]
