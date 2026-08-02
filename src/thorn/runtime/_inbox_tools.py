"""Agent-facing tools for working with the current session's inbox.

These tools let an agent inspect, focus, and close out notifications
that have been posted to its :class:`~thorn.runtime._inbox.SessionInbox`.
Five operations are exposed:

- :func:`list_inbox_items` -- summary of every item currently in the
  inbox (pending and in-progress).  Handled, errored, and confirmed
  items are filtered out because they represent closed-out work.
- :func:`read_inbox_item` -- full content and metadata of one item by
  its ULID.
- :func:`update_focus` -- claim/resume a single focused inbox item and
  update the session working-set phase.
- :func:`complete_focused_work` -- mark the currently focused item
  ``handled`` after close-out gates pass.
- :func:`park_focused_work` -- mark the currently focused item
  ``errored`` with an operator-readable reason.

The older :func:`update_inbox_item` helper remains available for
compatibility and recovery code, but is intentionally not part of the
default agent-facing toolset.  Ordinary agent workflow should move
through :func:`update_focus`, then one of the focused terminal tools,
instead of setting arbitrary lifecycle statuses directly.

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
from typing import TYPE_CHECKING, Literal

from thorn.core._context import get_context
from thorn.core._executor import ToolVenue
from thorn.core._func import tool
from thorn.runtime._address import SessionAddress
from thorn.runtime._current_session import (
    CurrentSessionRuntime,
    current_session_runtime,
)
from thorn.runtime._dispatch import (
    DispatchError,
    apply_handling_transition,
)
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    InboxCompletionRationale,
    Notification,
    NotificationID,
    NotificationStatus,
)
from thorn.runtime._todo import LinkedTodoSummary, SessionTodoList, TodoItem
from thorn.runtime._working_set import (
    HandlingPhase,
    LastValidation,
    ValidationOutcome,
    WorkingSet,
    WorkingSetBlocker,
)

if TYPE_CHECKING:
    from thorn.core._session import Session
    from thorn.runtime._working_set_telemetry import (
        WorkingSetGateTelemetry,
        WorkingSetTelemetryKind,
    )

log = logging.getLogger(__name__)


# Allowed values for the compatibility status parameter of
# ``update_inbox_item``.  Defined once as a module-level Literal so the
# tool signature and the docstring stay in sync; the concrete list is
# a subset of :class:`NotificationStatus` (pending, in_progress, and
# confirmed are framework-owned in ordinary agent workflow).
_AgentTerminalStatus = Literal["handled", "errored"]
_AgentFocusPhase = Literal[
    "intake",
    "inspect",
    "act",
    "validate",
    "closeout",
    "blocked",
]
_AgentValidationOutcome = Literal["passed", "failed", "skipped", "unknown"]


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
    resolved = current_session_runtime("Inbox tools")
    if isinstance(resolved, str):
        return resolved

    address = SessionAddress(resolved.agent_id, resolved.session_key)
    queue = resolved.runtime.address_book.get(address)
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


def _current_session_todos() -> SessionTodoList | str:
    """Resolve the current scope's session TODO list."""
    resolved = current_session_runtime("Inbox tools")
    if isinstance(resolved, str):
        return resolved

    todo_file = resolved.runtime.paths.session_todo_file(
        resolved.agent_id,
        resolved.session_key,
    )
    return SessionTodoList(todo_file)


def _current_session_object(resolved: CurrentSessionRuntime) -> "Session":
    if resolved.session is not None:
        return resolved.session
    return resolved.runtime.get_or_create_session(
        resolved.agent,
        resolved.session_key,
    )


def _optional_tool_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


async def _emit_focus_diagnostic(message: str) -> None:
    try:
        ctx = get_context()
    except RuntimeError:
        return
    await ctx.event_sink.on_status(
        f"working-set diagnostic: {message}",
        scope=ctx.scope,
    )


async def _emit_working_set_telemetry(
    kind: "WorkingSetTelemetryKind",
    working_set: WorkingSet,
    *,
    gate: "WorkingSetGateTelemetry | None" = None,
) -> None:
    try:
        ctx = get_context()
    except RuntimeError:
        return

    from thorn.runtime._working_set_telemetry import (
        build_working_set_telemetry,
    )

    await ctx.event_sink.on_working_set_telemetry(
        build_working_set_telemetry(
            kind=kind,
            working_set=working_set,
            gate=gate,
        ),
        scope=ctx.scope,
    )


async def _focus_gate_error(
    working_set: WorkingSet,
    *,
    gate_name: str,
    message: str,
) -> str:
    from thorn.runtime._working_set_telemetry import (
        WorkingSetGateTelemetry,
        WorkingSetTelemetryKind,
    )

    await _emit_working_set_telemetry(
        WorkingSetTelemetryKind.GATE_INTERVENTION,
        working_set,
        gate=WorkingSetGateTelemetry(name=gate_name, reason=message),
    )
    return f"Error: {message}"


async def _emit_unfocused_in_progress_diagnostic(
    inbox: SessionInbox,
    working_set: WorkingSet,
) -> None:
    focused_id = working_set.focused_inbox_item_id
    unfocused_ids = [
        str(item.id) for item in inbox.prompt_pending()
        if (
            item.status is NotificationStatus.IN_PROGRESS
            and item.id != focused_id
        )
    ]
    if not unfocused_ids:
        return
    await _emit_focus_diagnostic(
        "in-progress inbox item(s) outside working-set focus: "
        + ", ".join(unfocused_ids),
    )


async def _save_session_after_focus_change(
    resolved: CurrentSessionRuntime,
    working_set: WorkingSet,
) -> str | None:
    session = _current_session_object(resolved)
    session.working_set = working_set
    try:
        resolved.runtime.save_session(session)
    except (OSError, ValueError) as exc:
        message = f"working-set persistence failed: {exc}"
        await _emit_focus_diagnostic(message)
        return message
    return None


async def _reset_focused_work_after_terminal_inbox_update(item_id: str) -> None:
    resolved = current_session_runtime("Inbox tools")
    if isinstance(resolved, str):
        return

    session = _current_session_object(resolved)
    focused_id = session.working_set.focused_inbox_item_id
    if focused_id != NotificationID(item_id):
        return

    save_error = await _save_session_after_focus_change(
        resolved,
        WorkingSet(),
    )
    if save_error is not None:
        await _emit_focus_diagnostic(
            f"inbox item {item_id} reached a terminal status, "
            "but focused work was not durably reset",
        )


def _focused_item_id_or_error() -> NotificationID | str:
    resolved = current_session_runtime("Inbox tools")
    if isinstance(resolved, str):
        return resolved

    session = _current_session_object(resolved)
    working_set = session.working_set
    if working_set.focused_inbox_item_id is None:
        return (
            "Error: no inbox item is focused. Use update_focus(...) to "
            "claim or resume a specific item before closing or parking work."
        )
    return working_set.focused_inbox_item_id


def _handled_gate_error(item_id: str) -> str | None:
    resolved = current_session_runtime("Inbox tools")
    if isinstance(resolved, str):
        return resolved

    session = _current_session_object(resolved)
    working_set = session.working_set
    focused_id = working_set.focused_inbox_item_id
    if focused_id != NotificationID(item_id):
        if focused_id is None:
            return (
                "Error: inbox item cannot be marked handled until "
                "update_focus establishes it as the focused work."
            )
        return (
            f"Error: inbox item {item_id!r} cannot be marked handled "
            f"while focused work is {focused_id}. Use update_focus(...) "
            "to switch focus deliberately before close-out."
        )
    if working_set.phase is not HandlingPhase.CLOSEOUT:
        return (
            "Error: focused inbox item cannot be marked handled until "
            "update_focus moves it to phase 'closeout'."
        )
    if (
        working_set.last_validation is None
        and working_set.no_validation_rationale is None
    ):
        return (
            "Error: focused inbox item cannot be marked handled without "
            "validation evidence or no_validation_rationale in the "
            "working set."
        )
    return None


async def _transition_terminal_inbox_item(
    item_id: str,
    target_status: NotificationStatus,
    *,
    notes: str = "",
    completion: InboxCompletionRationale | None = None,
) -> str:
    inbox = _current_session_inbox()
    if isinstance(inbox, str):
        return inbox

    resolved = current_session_runtime("Inbox tools")
    if isinstance(resolved, str):
        return resolved

    cleaned_notes = notes.strip()

    try:
        if target_status is NotificationStatus.ERRORED:
            if not cleaned_notes:
                return (
                    "Error: notes is required when parking focused work; "
                    "include a brief reason."
                )
            if completion is not None:
                return (
                    "Error: completion is only valid when marking an item "
                    "as 'handled'."
                )
            updated = apply_handling_transition(
                inbox,
                item_id,
                target_status,
                address_book=resolved.runtime.address_book,
                error_reason=cleaned_notes,
            )
        elif target_status is NotificationStatus.HANDLED:
            if completion is None:
                return (
                    "Error: completion is required when marking an item "
                    "as 'handled'; include completed_actions, "
                    "request_coverage, remaining_work, self_review, and "
                    "external_follow_up."
                )
            validation_errors = completion.validation_errors()
            if validation_errors:
                return (
                    "Error: completion rationale is incomplete: "
                    + "; ".join(validation_errors)
                )
            try:
                inbox.get(item_id)
            except KeyError:
                return f"Error: no inbox item with id {item_id!r}."
            closeout_gate_error = _handled_gate_error(item_id)
            if closeout_gate_error is not None:
                return closeout_gate_error
            open_linked_todos = _open_linked_todos_for_current_session(item_id)
            if isinstance(open_linked_todos, str):
                return open_linked_todos
            if open_linked_todos:
                return _linked_todo_blocking_error(item_id, open_linked_todos)
            updated = apply_handling_transition(
                inbox,
                item_id,
                target_status,
                address_book=resolved.runtime.address_book,
                notes=cleaned_notes or None,
                completion_rationale=completion,
            )
        else:
            return (
                f"Error: invalid terminal status {target_status.value!r}. "
                "Use update_focus(...) to claim active work."
            )
    except KeyError:
        return f"Error: no inbox item with id {item_id!r}."
    except ValueError as exc:
        return f"Error: {exc}"
    except DispatchError as exc:
        # Step 1 landed, step 2 did not.  The sweep will reconcile on
        # the next startup, but we should tell the agent that the
        # status change is visible while dispatch is stuck.
        await _reset_focused_work_after_terminal_inbox_update(item_id)
        return (
            f"Warning: item {item_id} is now marked {target_status.value}, "
            f"but dispatch to its RSVP target failed: {exc}. "
            "The item will be retried on next runtime start."
        )

    await _reset_focused_work_after_terminal_inbox_update(item_id)
    return f"Item {updated.id} is now {updated.status.value}."


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


def _todo_summaries_for_current_session() -> dict[NotificationID, LinkedTodoSummary]:
    todos = _current_session_todos()
    if isinstance(todos, str):
        return {}
    try:
        return todos.linked_summary_by_inbox_item()
    except ValueError:
        return {}


def _todo_summary_for(
    item: Notification,
    summaries: dict[NotificationID, LinkedTodoSummary],
) -> LinkedTodoSummary | None:
    summary = summaries.get(item.id)
    if summary is None or summary.total_count == 0:
        return None
    return summary


def _render_todo_summary(summary: LinkedTodoSummary) -> str:
    text = f"TODOs: {summary.open_count} open, {summary.resolved_count} resolved"
    if summary.open_titles:
        text += ": " + "; ".join(summary.open_titles)
    return text


def _open_linked_todos_for_current_session(
    item_id: str,
) -> list[TodoItem] | str:
    todos = _current_session_todos()
    if isinstance(todos, str):
        return todos
    try:
        return todos.unresolved_linked_to(NotificationID(item_id))
    except ValueError as exc:
        return f"Error: session TODO list is unreadable: {exc}"


def _linked_todo_blocking_error(
    item_id: str,
    open_todos: list[TodoItem],
) -> str:
    lines = [
        f"Error: cannot mark inbox item {item_id!r} handled while "
        "linked session TODOs remain open TODO items:",
    ]
    for todo_item in open_todos:
        lines.append(f"- `{todo_item.id}` {todo_item.title}")
    lines.append(
        "Complete or abandon each linked TODO with rationale before "
        "marking the inbox item handled."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: update_focus
# ---------------------------------------------------------------------------

@tool(venue=ToolVenue.IN_PROCESS)
async def update_focus(
    phase: _AgentFocusPhase,
    item_id: str | None = None,
    objective: str | None = None,
    action_summary: str | None = None,
    validation_outcome: _AgentValidationOutcome | None = None,
    validation_summary: str | None = None,
    validation_command: str | None = None,
    notes: str | None = None,
    override_rationale: str | None = None,
    blocker_summary: str | None = None,
    unblock_condition: str | None = None,
    no_validation_rationale: str | None = None,
) -> str:
    """Update the current session's focused work and handling phase.

    Use this when you start or resume work on an inbox item, or when
    you move the focused work between ``inspect``, ``act``,
    ``validate``, ``closeout``, and ``blocked``.  Starting work on a
    pending item first claims it as ``in_progress``, then updates the
    session working set.  Re-running the same operation for an
    already-``in_progress`` item is allowed and repairs missing
    working-set focus.

    ``item_id`` and a clear ``objective`` are required when moving
    from ``intake`` into active work.  After a focus is established,
    omit ``item_id`` for phase changes that continue the same work.
    Switching to a different item while work is already focused
    requires ``override_rationale``.

    Moving to ``validate`` requires ``action_summary`` unless
    ``override_rationale`` explains a validation-only pass.  Moving
    to ``closeout`` requires validation evidence
    (``validation_outcome`` plus ``validation_summary``) or an
    explicit ``no_validation_rationale``.

    This tool does not mark an item ``handled`` or ``errored``.  Use
    :func:`complete_focused_work` after ``closeout`` gates pass, or
    :func:`park_focused_work` when the focused item cannot be handled.
    """
    resolved = current_session_runtime("Focus tools")
    if isinstance(resolved, str):
        return resolved

    inbox = _current_session_inbox()
    if isinstance(inbox, str):
        return inbox

    try:
        target_phase = HandlingPhase(str(phase))
    except ValueError:
        return (
            f"Error: invalid phase {phase!r}. Must be one of "
            "'intake', 'inspect', 'act', 'validate', 'closeout', or 'blocked'."
        )

    session = _current_session_object(resolved)
    current_working_set = session.working_set
    cleaned_item_id = _optional_tool_text(item_id)
    cleaned_objective = _optional_tool_text(objective)
    cleaned_action_summary = _optional_tool_text(action_summary)
    cleaned_notes = _optional_tool_text(notes)
    cleaned_override = _optional_tool_text(override_rationale)
    cleaned_blocker_summary = _optional_tool_text(blocker_summary)
    cleaned_unblock_condition = _optional_tool_text(unblock_condition)
    cleaned_validation_summary = _optional_tool_text(validation_summary)
    cleaned_validation_command = _optional_tool_text(validation_command)
    cleaned_no_validation_rationale = _optional_tool_text(
        no_validation_rationale,
    )

    if target_phase is HandlingPhase.INTAKE:
        if cleaned_item_id is not None:
            return "Error: item_id is not valid when returning focus to intake."
        save_error = await _save_session_after_focus_change(
            resolved,
            WorkingSet(),
        )
        if save_error is not None:
            return (
                "Warning: focus reset was applied in memory, but could not be "
                f"persisted: {save_error}"
            )
        await _emit_unfocused_in_progress_diagnostic(inbox, WorkingSet())
        from thorn.runtime._working_set_telemetry import WorkingSetTelemetryKind

        await _emit_working_set_telemetry(
            WorkingSetTelemetryKind.FOCUS_UPDATED,
            WorkingSet(),
        )
        return "Focus cleared. Phase is intake."

    if (
        target_phase is not HandlingPhase.BLOCKED
        and (
            cleaned_blocker_summary is not None
            or cleaned_unblock_condition is not None
        )
    ):
        return "Error: blocker fields are only valid when phase is 'blocked'."
    if (
        target_phase is HandlingPhase.BLOCKED
        and (
            cleaned_blocker_summary is None
            or cleaned_unblock_condition is None
        )
    ):
        return (
            "Error: blocker_summary and unblock_condition must be provided "
            "together when phase is 'blocked'."
        )
    if (
        target_phase is not HandlingPhase.CLOSEOUT
        and (
            validation_outcome is not None
            or cleaned_validation_summary is not None
            or cleaned_validation_command is not None
            or cleaned_no_validation_rationale is not None
        )
    ):
        return (
            "Error: validation evidence is only recorded when moving "
            "to phase 'closeout'."
        )
    if (
        validation_outcome is None
        and (
            cleaned_validation_summary is not None
            or cleaned_validation_command is not None
        )
    ):
        return (
            "Error: validation_outcome is required when recording "
            "validation summary or command."
        )
    if validation_outcome is not None and cleaned_validation_summary is None:
        return (
            "Error: validation_summary is required when recording "
            "validation_outcome."
        )
    if (
        validation_outcome is not None
        and cleaned_no_validation_rationale is not None
    ):
        return (
            "Error: provide validation evidence or no_validation_rationale, "
            "not both."
        )

    validation_record: LastValidation | None = None
    if validation_outcome is not None:
        try:
            outcome = ValidationOutcome(str(validation_outcome))
        except ValueError:
            return (
                f"Error: invalid validation_outcome {validation_outcome!r}. "
                "Must be one of 'passed', 'failed', 'skipped', or 'unknown'."
            )
        validation_record = LastValidation(
            outcome=outcome,
            summary=cleaned_validation_summary or "",
            command=cleaned_validation_command,
        )

    if (
        cleaned_item_id is None
        and current_working_set.phase is HandlingPhase.INTAKE
    ):
        return await _focus_gate_error(
            current_working_set,
            gate_name="item_id_required_from_intake",
            message=(
                "item_id is required when moving from intake into "
                "focused work."
            ),
        )

    target_item_id = (
        NotificationID(cleaned_item_id)
        if cleaned_item_id is not None
        else current_working_set.focused_inbox_item_id
    )
    if target_item_id is None:
        return await _focus_gate_error(
            current_working_set,
            gate_name="item_id_required_without_focus",
            message="item_id is required because no focused item is recorded.",
        )

    current_focus_id = current_working_set.focused_inbox_item_id
    switching_focus = (
        current_focus_id is not None
        and current_focus_id != target_item_id
    )
    if (
        switching_focus
        and current_working_set.phase is not HandlingPhase.INTAKE
        and cleaned_override is None
    ):
        return await _focus_gate_error(
            current_working_set,
            gate_name="override_required_to_switch_focus",
            message=(
                f"focused work is already on {current_focus_id}. "
                "Provide override_rationale to switch focus before close-out."
            ),
        )

    try:
        item = inbox.get(target_item_id)
    except KeyError:
        return f"Error: no inbox item with id {str(target_item_id)!r}."

    if item.status not in (
        NotificationStatus.PENDING,
        NotificationStatus.IN_PROGRESS,
    ):
        return (
            f"Error: inbox item {target_item_id} is {item.status.value}; "
            "focus can only target pending or in-progress items."
        )

    carry_existing = (
        current_focus_id == target_item_id
        and current_working_set.phase is not HandlingPhase.INTAKE
    )
    objective_value = (
        cleaned_objective
        if cleaned_objective is not None
        else current_working_set.objective if carry_existing else None
    )
    if objective_value is None and cleaned_override is None:
        return await _focus_gate_error(
            current_working_set,
            gate_name="objective_required",
            message=(
                "objective is required for focused work. Provide a "
                "clear objective or an override_rationale explaining why "
                "the transition cannot record one yet."
            ),
        )

    action_value = (
        cleaned_action_summary
        if cleaned_action_summary is not None
        else (
            current_working_set.last_action_summary
            if carry_existing else None
        )
    )
    if (
        target_phase is HandlingPhase.VALIDATE
        and action_value is None
        and cleaned_override is None
    ):
        return await _focus_gate_error(
            current_working_set,
            gate_name="action_summary_required_for_validate",
            message=(
                "action_summary is required before moving to validate. "
                "Provide action_summary for the effectful work already done, "
                "or override_rationale for validation-only work."
            ),
        )

    blocker = None
    if target_phase is HandlingPhase.BLOCKED:
        blocker = WorkingSetBlocker(
            summary=cleaned_blocker_summary or "",
            unblock_condition=cleaned_unblock_condition or "",
        )

    last_validation = (
        validation_record
        if validation_record is not None
        else current_working_set.last_validation if carry_existing else None
    )
    no_validation = (
        cleaned_no_validation_rationale
        if cleaned_no_validation_rationale is not None
        else (
            current_working_set.no_validation_rationale
            if carry_existing else None
        )
    )
    if cleaned_no_validation_rationale is not None:
        last_validation = None
    if target_phase is HandlingPhase.CLOSEOUT and (
        last_validation is None and no_validation is None
    ):
        return await _focus_gate_error(
            current_working_set,
            gate_name="validation_required_for_closeout",
            message=(
                "closeout requires validation evidence or "
                "no_validation_rationale."
            ),
        )

    try:
        if item.status is NotificationStatus.PENDING or cleaned_notes is not None:
            item = apply_handling_transition(
                inbox,
                target_item_id,
                NotificationStatus.IN_PROGRESS,
                address_book=resolved.runtime.address_book,
                notes=cleaned_notes,
            )
    except KeyError:
        return f"Error: no inbox item with id {str(target_item_id)!r}."
    except ValueError as exc:
        return f"Error: {exc}"

    next_working_set = WorkingSet(
        phase=target_phase,
        focused_inbox_item_id=target_item_id,
        objective=objective_value,
        last_action_summary=action_value,
        active_context=(
            current_working_set.active_context if carry_existing else ()
        ),
        last_validation=last_validation,
        no_validation_rationale=no_validation,
        blocker=blocker,
        override_rationale=cleaned_override,
    )
    save_error = await _save_session_after_focus_change(
        resolved,
        next_working_set,
    )
    await _emit_unfocused_in_progress_diagnostic(inbox, next_working_set)
    if save_error is not None:
        return (
            f"Warning: item {item.id} is now {item.status.value}, but "
            f"working-set persistence failed: {save_error}"
        )

    from thorn.runtime._working_set_telemetry import WorkingSetTelemetryKind

    await _emit_working_set_telemetry(
        WorkingSetTelemetryKind.FOCUS_UPDATED,
        next_working_set,
    )

    lines = [
        f"Focus updated: item {item.id}; phase={target_phase.value}.",
        f"Inbox item {item.id} is {item.status.value}.",
    ]
    return "\n".join(lines)


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

    todo_summaries = _todo_summaries_for_current_session()
    lines = [f"You have {len(items)} inbox item(s):"]
    for item in items:
        line = (
            f"- [{item.id}] status={item.status.value} "
            f"source={item.source}: {_summarize(item)}"
        )
        todo_summary = _todo_summary_for(item, todo_summaries)
        if todo_summary is not None:
            line = f"{line}  [{_render_todo_summary(todo_summary)}]"
        lines.append(line)
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
    if item.completion_rationale:
        lines.extend(item.completion_rationale.to_display_text().splitlines())
    if item.error_reason:
        lines.append(f"Error reason: {item.error_reason}")
    if item.metadata:
        lines.append(f"Metadata: {dict(item.metadata)!r}")
    lines.append("")
    lines.append("Content:")
    lines.append(item.content)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools: focused terminal actions
# ---------------------------------------------------------------------------

@tool(venue=ToolVenue.IN_PROCESS)
async def complete_focused_work(
    completion: InboxCompletionRationale,
    notes: str = "",
) -> str:
    """Mark the currently focused inbox item handled.

    Use this only after :func:`update_focus` has established a focused
    inbox item and moved it to phase ``closeout`` with validation
    evidence or ``no_validation_rationale``.  Self-review the original
    request before calling it.  This tool records a structured
    completion rationale, verifies that linked session TODOs are
    resolved, performs the durable ``handled`` transition, and clears
    the session working set back to ``intake``.

    Args:
        completion: Required close-out evidence with
            completed_actions, request_coverage, remaining_work,
            self_review, and external_follow_up.  Leave
            remaining_work empty when no follow-up remains.
        notes: Optional extra context carried through to any RSVP
            recipient of this item.
    """
    focused_item_id = _focused_item_id_or_error()
    if isinstance(focused_item_id, str) and focused_item_id.startswith("Error:"):
        return focused_item_id
    return await _transition_terminal_inbox_item(
        str(focused_item_id),
        NotificationStatus.HANDLED,
        notes=notes,
        completion=completion,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def park_focused_work(notes: str) -> str:
    """Park the currently focused inbox item as errored.

    Use this when the focused work cannot be completed and should be
    handed back to the operator or original sender.  The explanation
    in *notes* is required, becomes the item's error reason, and is
    carried through any RSVP dispatch.  The session working set is
    cleared back to ``intake`` after the terminal transition lands.

    Args:
        notes: Brief operator-readable explanation of why the focused
            item cannot be handled.
    """
    focused_item_id = _focused_item_id_or_error()
    if isinstance(focused_item_id, str) and focused_item_id.startswith("Error:"):
        return focused_item_id
    return await _transition_terminal_inbox_item(
        str(focused_item_id),
        NotificationStatus.ERRORED,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Tool: update_inbox_item
# ---------------------------------------------------------------------------

@tool(venue=ToolVenue.IN_PROCESS)
async def update_inbox_item(
    item_id: str,
    status: _AgentTerminalStatus,
    notes: str = "",
    completion: InboxCompletionRationale | None = None,
) -> str:
    """Compatibility helper for terminal inbox transitions.

    This low-level helper is not part of the default agent-facing
    toolset.  Ordinary workflow should use :func:`update_focus` to
    claim or resume the item, then :func:`complete_focused_work` or
    :func:`park_focused_work` for close-out.  This helper remains
    importable for recovery code and tests that need to name an item
    explicitly.  Do not use this helper to claim ``in_progress`` work;
    use :func:`update_focus` instead.

    Status values:

    - ``handled``: you have completed every action requested by this
      notification.  Before using this status, self-review against
      the inbox item and any linked issue, comment, or change
      request.  Partial, first-pass, exploratory, or incremental
      work must remain focused until it is actually complete.
      ``handled`` means the
      notification/request is done; it does not necessarily mean an
      external issue or change request is closed.  You must provide a
      structured *completion* rationale with completed actions,
      request coverage, remaining work, self-review, and external
      follow-up.  The item must already be the focused work and the
      working set must be in ``closeout`` with validation evidence or
      ``no_validation_rationale``.  The item will be removed from
      your inbox (or forwarded to its original sender if an RSVP was
      requested).  Optionally attach *notes* for extra context; the
      notes and completion rationale travel to any RSVP recipient.
    - ``errored``: you cannot handle this item.  *notes* is
      **required** and should explain why.  The item will be moved
      to an ``errored/`` parking area for operator inspection, or
      forwarded to its original sender (with your explanation) if
      an RSVP was requested.

    Args:
        item_id: The ULID of the item.  See
            :func:`list_inbox_items` for available IDs.
        status: Terminal status.  Must be ``handled`` or ``errored``.
            Use :func:`update_focus` instead of this helper to claim
            work as ``in_progress``.
        notes: Free-form explanation.  Required for ``errored``
            (and rejected if empty); optional for the other statuses.
            When present, carried through to any RSVP recipient of
            this item.
        completion: Required when *status* is ``handled``.  Provide
            completed_actions, request_coverage, remaining_work,
            self_review, and external_follow_up.  Leave
            remaining_work empty when no follow-up remains.
    """
    inbox = _current_session_inbox()
    if isinstance(inbox, str):
        return inbox

    try:
        target_status = NotificationStatus(status)
    except ValueError:
        return (
            f"Error: invalid status {status!r}. "
            "Must be one of 'handled', 'errored'. Use update_focus(...) "
            "to claim or resume active work."
        )

    if target_status is NotificationStatus.IN_PROGRESS:
        return (
            "Error: update_inbox_item no longer claims work as "
            "in_progress. Use update_focus(phase='inspect', item_id=..., "
            "objective=..., notes=...) to claim or resume work."
        )

    return await _transition_terminal_inbox_item(
        item_id,
        target_status,
        notes=notes,
        completion=completion,
    )


INBOX_TOOLS: list = [
    list_inbox_items,
    read_inbox_item,
    update_focus,
    complete_focused_work,
    park_focused_work,
]
"""Default inbox tools added to every agent via ``Agent._collect_tools``."""


__all__ = [
    "INBOX_TOOLS",
    "list_inbox_items",
    "read_inbox_item",
    "update_focus",
    "complete_focused_work",
    "park_focused_work",
    "update_inbox_item",
]
