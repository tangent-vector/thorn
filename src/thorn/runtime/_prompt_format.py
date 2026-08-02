"""Inbox-aware prompt construction for session prompt rounds.

This module ships the default :data:`PromptDispatcher` the scheduler
uses for ordinary session work:

- :func:`build_inbox_prompt` is the pure formatting function that
  turns a list of :class:`Notification` items (in post order, oldest
  first) into the prompt text to send to the agent.  Kept pure so
  that alternative dispatchers -- or test scaffolding -- can reuse
  the formatting without pulling in a full Session/asyncio setup.

- :func:`inbox_prompt_dispatcher` is the actual ``PromptDispatcher``:
  it reads :meth:`SessionInbox.prompt_pending`, builds the text with
  :func:`build_inbox_prompt`, and awaits ``session.prompt(...)``.
  A plain module-level async function is sufficient because the
  dispatcher has no configuration state.

Design notes
------------

The prompt has two shapes, determined by the number of pending items:

- **Single item (prettified).**  When ``len(items) == 1`` we render
  the item's full content inline, preceded by a compact metadata
  block.  This matches today's gateway behaviour (agent sees the
  notification content as its prompt) while adding the focused
  close-out contract the inbox tools need.

- **Multi item (summary).**  When ``len(items) > 1`` we render a
  per-item summary line carrying ID, status, source, and a one-line
  content summary.  ``in_progress`` items additionally carry any
  ``notes`` the agent left on a previous round, so the agent can
  resume work without a mandatory :func:`read_inbox_item`
  round-trip.

The prompt text is deterministic given a fixed list of items, so it
is straightforward to snapshot-test without committing to exact
wording (tests check structural properties rather than byte
equality).

No mutation
-----------

The dispatcher never transitions notifications.  ``attempt_count`` is
not incremented here; the progress-guarantee work item owns that
bookkeeping.  The only side effect is the ``session.prompt(...)``
await, which the scheduler wraps under the agent-level concurrency
cap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

from thorn.core._context import get_context
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    Notification,
    NotificationID,
    NotificationStatus,
)
from thorn.runtime._todo import LinkedTodoSummary, SessionTodoList

if TYPE_CHECKING:
    from thorn.core._session import Session


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

_SUMMARY_CHARS = 80
"""Maximum number of characters shown in a single-item summary line.

Matches the constant used by :mod:`thorn.runtime._inbox_tools` so that
the list-style view the agent constructs on demand and the list-style
view the scheduler hands it at prompt time look the same."""


def summarize_notification_content(notification: Notification) -> str:
    """Return a compact, single-line summary of *notification*.

    Takes the first non-empty line of ``notification.content``,
    strips it, and truncates to :data:`_SUMMARY_CHARS` characters
    with an ellipsis on overrun.  Empty content renders as a
    visible placeholder so the summary line never collapses to
    empty space.
    """
    first_line = notification.content.split("\n", 1)[0].strip()
    if len(first_line) > _SUMMARY_CHARS:
        first_line = first_line[: _SUMMARY_CHARS - 1].rstrip() + "\u2026"
    return first_line or "(empty content)"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_CLOSE_OUT_INSTRUCTIONS_SINGLE = (
    "As soon as you start acting on this notification, use "
    "update_focus(phase=\"inspect\", item_id=\"{item_id}\", "
    "objective=\"...\", notes=\"...\") to claim it as your focused work. "
    "Move the focused work through inspect/act/validate/closeout with "
    "update_focus as you learn, act, and validate. Before close-out, "
    "self-review against this notification and any linked issue, comment, "
    "or change request, then record validation evidence with "
    "update_focus(phase=\"closeout\", validation_outcome=\"...\", "
    "validation_summary=\"...\") or provide no_validation_rationale. "
    "After closeout gates pass, use complete_focused_work(completion={{...}}). "
    "The completion object must include completed_actions, request_coverage, "
    "remaining_work, self_review, and external_follow_up. Complete focused "
    "work only after every action requested by this notification is complete. "
    "Partial, first-pass, exploratory, or incremental work stays focused and "
    "in_progress unless the human explicitly asked only for that limited pass. "
    "`handled` means this notification/request is done, not necessarily that "
    "an external issue or change request is closed. If you cannot handle it, "
    "use park_focused_work(notes=\"...\") with a brief reason."
)

_CLOSE_OUT_INSTRUCTIONS_MULTI = (
    "Items are listed in post order (oldest first). Handle them in order "
    "unless one clearly warrants out-of-order attention. Use "
    "read_inbox_item(id) to see an item's full content. As soon as you start "
    "acting on an item, claim it with update_focus(phase=\"inspect\", "
    "item_id=id, objective=\"...\", notes=\"...\"). Move the focused work "
    "through inspect/act/validate/closeout with update_focus as you learn, "
    "act, and validate. Before completing focused work, self-review against "
    "the inbox item and any linked issue, comment, or change request, then "
    "record validation evidence with update_focus(phase=\"closeout\", "
    "validation_outcome=\"...\", validation_summary=\"...\") or provide "
    "no_validation_rationale. After closeout gates pass, call "
    "complete_focused_work(completion={...}) with completed_actions, "
    "request_coverage, remaining_work, self_review, and external_follow_up. "
    "`handled` means every action requested by that notification is complete, "
    "not necessarily that an external issue or change request is closed. "
    "Partial, first-pass, exploratory, or incremental work stays focused and "
    "in_progress unless the human explicitly asked only for that limited pass. "
    "If the focused item cannot be handled, use park_focused_work(notes=\"...\")."
)


def build_inbox_prompt(
    items: list[Notification],
    *,
    todo_summaries: Mapping[NotificationID | str, LinkedTodoSummary] | None = None,
) -> str:
    """Construct the prompt text for a scheduler-driven prompt round.

    Branches on ``len(items)``:

    - Empty list raises :class:`ValueError`.  The scheduler never
      calls the dispatcher with an empty inbox, so this is a
      programming-error guard rather than a supported path.
    - One item: prettified single-item form.
    - More than one item: summary list with close-out instructions.

    The result is a plain ``str``; callers hand it to
    ``session.prompt(...)``.
    """
    if not items:
        raise ValueError(
            "build_inbox_prompt requires at least one notification"
        )
    if len(items) == 1:
        return _build_single_item_prompt(items[0], todo_summaries=todo_summaries)
    return _build_multi_item_prompt(items, todo_summaries=todo_summaries)


def _build_single_item_prompt(
    item: Notification,
    *,
    todo_summaries: Mapping[NotificationID | str, LinkedTodoSummary] | None,
) -> str:
    """Render a single-item prompt with full content inline.

    The header names the item and its provenance; the body is the
    notification's full content; the footer is the close-out
    instruction referencing the item's ID by value so the agent
    need not hunt for it.
    """
    lines = [
        f"Incoming notification (id: {item.id}, source: {item.source}, "
        f"status: {item.status.value}):",
    ]
    if item.status is NotificationStatus.IN_PROGRESS and item.notes:
        # Surface the agent's own prior notes so it can pick up where
        # it left off without a round trip through read_inbox_item.
        lines.append(f"Your prior notes on this item: {item.notes}")
    linked_todos = _todo_summary_for(item, todo_summaries)
    if linked_todos is not None:
        lines.extend(_render_linked_todo_lines(linked_todos))
    lines.append("")
    lines.append(item.content)
    lines.append("")
    lines.append(_CLOSE_OUT_INSTRUCTIONS_SINGLE.format(item_id=item.id))
    return "\n".join(lines)


def _build_multi_item_prompt(
    items: list[Notification],
    *,
    todo_summaries: Mapping[NotificationID | str, LinkedTodoSummary] | None,
) -> str:
    """Render a summary list prompt for multiple items.

    The header reports counts; the body is one summary line per
    item; the footer points at the inbox tools.
    """
    pending = sum(
        1 for item in items if item.status is NotificationStatus.PENDING
    )
    in_progress = sum(
        1 for item in items
        if item.status is NotificationStatus.IN_PROGRESS
    )

    header = (
        f"You have {len(items)} inbox item(s) "
        f"({pending} pending, {in_progress} in progress)."
    )
    lines: list[str] = [header, ""]
    for item in items:
        lines.append(_render_summary_line(item, todo_summaries=todo_summaries))
    lines.append("")
    lines.append(_CLOSE_OUT_INSTRUCTIONS_MULTI)
    return "\n".join(lines)


def _render_summary_line(
    item: Notification,
    *,
    todo_summaries: Mapping[NotificationID | str, LinkedTodoSummary] | None = None,
) -> str:
    """Render one summary line for the multi-item prompt.

    For ``in_progress`` items with non-empty ``notes``, the notes
    are appended as a ``[notes: ...]`` annotation so the agent can
    see its own prior context at a glance.  Notes are truncated the
    same way as content summaries to keep the list compact.
    """
    summary = summarize_notification_content(item)
    line = (
        f"- [{item.id}] status={item.status.value} "
        f"source={item.source}: {summary}"
    )
    if item.status is NotificationStatus.IN_PROGRESS and item.notes:
        notes = item.notes.strip().replace("\n", " ")
        if len(notes) > _SUMMARY_CHARS:
            notes = notes[: _SUMMARY_CHARS - 1].rstrip() + "\u2026"
        line = f"{line}  [notes: {notes}]"
    linked_todos = _todo_summary_for(item, todo_summaries)
    if linked_todos is not None:
        line = f"{line}  [{_render_linked_todo_summary(linked_todos)}]"
    return line


def _todo_summary_for(
    item: Notification,
    todo_summaries: Mapping[NotificationID | str, LinkedTodoSummary] | None,
) -> LinkedTodoSummary | None:
    if todo_summaries is None:
        return None
    summary = todo_summaries.get(item.id) or todo_summaries.get(str(item.id))
    if summary is None or summary.total_count == 0:
        return None
    return summary


def _render_linked_todo_lines(summary: LinkedTodoSummary) -> list[str]:
    lines = [f"Linked TODOs: {_render_linked_todo_counts(summary)}."]
    if summary.open_titles:
        lines.append(
            "Open TODOs: " + _truncate_summary("; ".join(summary.open_titles))
        )
    return lines


def _render_linked_todo_summary(summary: LinkedTodoSummary) -> str:
    rendered = f"TODOs: {_render_linked_todo_counts(summary)}"
    if summary.open_titles:
        rendered += ": " + _truncate_summary("; ".join(summary.open_titles))
    return rendered


def _render_linked_todo_counts(summary: LinkedTodoSummary) -> str:
    return f"{summary.open_count} open, {summary.resolved_count} resolved"


def _truncate_summary(text: str) -> str:
    if len(text) <= _SUMMARY_CHARS:
        return text
    return text[: _SUMMARY_CHARS - 1].rstrip() + "\u2026"


def _todo_summaries_for_session(
    session: "Session",
) -> dict[NotificationID, LinkedTodoSummary]:
    if session.agent.id is None or session.key is None:
        return {}
    try:
        ctx = get_context()
    except RuntimeError:
        return {}
    if ctx.runtime is None:
        return {}
    todo_file = ctx.runtime.paths.session_todo_file(
        session.agent.id,
        session.key,
    )
    try:
        return SessionTodoList(todo_file).linked_summary_by_inbox_item()
    except ValueError:
        log.warning(
            "Could not load session TODOs for agent=%r session=%r",
            session.agent.id,
            session.key,
            exc_info=True,
        )
        return {}


# ---------------------------------------------------------------------------
# PromptDispatcher implementation
# ---------------------------------------------------------------------------

async def inbox_prompt_dispatcher(
    session: "Session",
    inbox: SessionInbox,
) -> None:
    """Default :data:`PromptDispatcher` for scheduler-driven sessions.

    Reads the inbox's prompt-pending view, builds the prompt text
    via :func:`build_inbox_prompt`, and awaits ``session.prompt``.

    If the inbox is empty when this runs the dispatcher returns
    without calling ``session.prompt``.  This happens in two benign
    cases:

    - The scheduler's re-check-before-idle timing let an item get
      transitioned to terminal between the scheduler's own pending
      check and the dispatcher invocation.
    - A handler on another thread/task drained items we were about
      to process.

    Neither is a bug.  The scheduler's drain loop will simply idle
    until the next kick.
    """
    items = inbox.prompt_pending()
    if not items:
        log.debug(
            "inbox_prompt_dispatcher: no items pending for agent=%r session=%r",
            session.agent.id, session.key,
        )
        return
    text = build_inbox_prompt(
        items,
        todo_summaries=_todo_summaries_for_session(session),
    )
    await session.prompt(text)


__all__ = [
    "build_inbox_prompt",
    "inbox_prompt_dispatcher",
    "summarize_notification_content",
]
