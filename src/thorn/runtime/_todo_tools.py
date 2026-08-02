"""Agent-facing tools for session TODO lists."""

from __future__ import annotations

from typing import Literal

from thorn.core._executor import ToolVenue
from thorn.core._func import tool
from thorn.runtime._current_session import current_session_runtime
from thorn.runtime._notification import NotificationID
from thorn.runtime._todo import SessionTodoList, TodoID, TodoItem, TodoStatus

_TodoStatusFilter = Literal["open", "resolved", "all"]


def _current_session_todos() -> SessionTodoList | str:
    """Resolve the current scope's session TODO list."""
    resolved = current_session_runtime("TODO tools")
    if isinstance(resolved, str):
        return resolved
    todo_file = resolved.runtime.paths.session_todo_file(
        resolved.agent_id,
        resolved.session_key,
    )
    return SessionTodoList(todo_file)


@tool(venue=ToolVenue.IN_PROCESS)
async def create_session_todo(
    title: str,
    linked_inbox_item_id: str | None = None,
    notes: str | None = None,
) -> str:
    """Create a TODO item for the current session.

    Args:
        title: Short description of the work to track.
        linked_inbox_item_id: Optional inbox item ID this TODO helps
            complete. If supplied, the inbox item cannot be marked
            handled while this TODO remains open.
        notes: Optional context for the TODO.
    """
    todos = _current_session_todos()
    if isinstance(todos, str):
        return todos

    linked_id: NotificationID | None = None
    if linked_inbox_item_id is not None:
        cleaned_linked_id = linked_inbox_item_id.strip()
        if cleaned_linked_id:
            linked_id = NotificationID(cleaned_linked_id)
    try:
        item = todos.create(
            title=title,
            linked_inbox_item_id=linked_id,
            notes=notes,
        )
    except ValueError as exc:
        return f"Error: {exc}"
    return f"Created TODO {_render_todo_id(item.id)}."


@tool(venue=ToolVenue.IN_PROCESS)
async def list_session_todos(
    status_filter: _TodoStatusFilter = "open",
) -> str:
    """List TODO items for the current session.

    Args:
        status_filter: ``open`` shows unresolved TODOs, ``resolved``
            shows completed and abandoned TODOs, and ``all`` shows
            every TODO.
    """
    todos = _current_session_todos()
    if isinstance(todos, str):
        return todos

    try:
        items = _filtered_items(todos.list_items(), status_filter)
    except ValueError as exc:
        return f"Error: {exc}"

    if not items:
        if status_filter == "open":
            return "No open session TODOs."
        if status_filter == "resolved":
            return "No resolved session TODOs."
        return "No session TODOs."

    lines = [f"Session TODOs ({status_filter}):"]
    for item in items:
        lines.extend(_render_item_lines(item))
    return "\n".join(lines)


@tool(venue=ToolVenue.IN_PROCESS)
async def update_session_todo(
    todo_id: str,
    title: str | None = None,
    notes: str | None = None,
    linked_inbox_item_id: str | None = None,
) -> str:
    """Update editable fields on a session TODO item.

    Args:
        todo_id: ID shown by :func:`list_session_todos` or
            :func:`create_session_todo`.
        title: Optional replacement title.
        notes: Optional replacement notes. Pass an empty string to
            clear existing notes.
        linked_inbox_item_id: Optional replacement inbox item link.
            Pass an empty string to clear existing inbox links.
    """
    todos = _current_session_todos()
    if isinstance(todos, str):
        return todos
    if title is None and notes is None and linked_inbox_item_id is None:
        return "Error: provide at least one field to update."

    linked_ids: tuple[NotificationID, ...] | None = None
    if linked_inbox_item_id is not None:
        cleaned_linked_id = linked_inbox_item_id.strip()
        linked_ids = (
            (NotificationID(cleaned_linked_id),)
            if cleaned_linked_id else ()
        )
    try:
        item = todos.update(
            todo_id,
            title=title,
            notes=notes,
            linked_inbox_item_ids=linked_ids,
        )
    except KeyError:
        return f"Error: no session TODO with id {_render_todo_id(todo_id)}."
    except ValueError as exc:
        return f"Error: {exc}"
    return f"TODO {_render_todo_id(item.id)} updated."


@tool(venue=ToolVenue.IN_PROCESS)
async def complete_session_todo(todo_id: str, rationale: str) -> str:
    """Mark a session TODO completed with an explicit rationale.

    Completion means the tracked work is finished and no longer
    blocks a linked inbox item's ``handled`` transition.
    """
    todos = _current_session_todos()
    if isinstance(todos, str):
        return todos
    try:
        item = todos.complete(todo_id, rationale=rationale)
    except KeyError:
        return f"Error: no session TODO with id {_render_todo_id(todo_id)}."
    except ValueError as exc:
        return f"Error: {exc}"
    return f"TODO {_render_todo_id(item.id)} completed."


@tool(venue=ToolVenue.IN_PROCESS)
async def abandon_session_todo(todo_id: str, rationale: str) -> str:
    """Mark a session TODO abandoned with an explicit rationale.

    Abandoning is for work that is deliberately dropped, superseded,
    or no longer required. It no longer blocks a linked inbox item's
    ``handled`` transition, so the rationale should explain why.
    """
    todos = _current_session_todos()
    if isinstance(todos, str):
        return todos
    try:
        item = todos.abandon(todo_id, rationale=rationale)
    except KeyError:
        return f"Error: no session TODO with id {_render_todo_id(todo_id)}."
    except ValueError as exc:
        return f"Error: {exc}"
    return f"TODO {_render_todo_id(item.id)} abandoned."


def _filtered_items(
    items: list[TodoItem],
    status_filter: str,
) -> list[TodoItem]:
    if status_filter == "open":
        return [item for item in items if item.status is TodoStatus.OPEN]
    if status_filter == "resolved":
        return [item for item in items if item.status.is_resolved]
    if status_filter == "all":
        return items
    raise ValueError(
        "status_filter must be one of 'open', 'resolved', or 'all'"
    )


def _render_item_lines(item: TodoItem) -> list[str]:
    lines = [
        f"- {_render_todo_id(item.id)} status={item.status.value}: {item.title}",
    ]
    if item.linked_inbox_item_ids:
        linked = ", ".join(
            str(item_id) for item_id in item.linked_inbox_item_ids
        )
        lines.append(f"  linked inbox: {linked}")
    if item.notes:
        lines.append(f"  notes: {item.notes}")
    if item.resolution_rationale:
        lines.append(f"  rationale: {item.resolution_rationale}")
    return lines


def _render_todo_id(todo_id: TodoID | str) -> str:
    text = str(todo_id)
    delimiter = "`"
    while delimiter in text:
        delimiter += "`"
    if text.startswith("`") or text.endswith("`"):
        return f"{delimiter} {text} {delimiter}"
    return f"{delimiter}{text}{delimiter}"


TODO_TOOLS: list = [
    create_session_todo,
    list_session_todos,
    update_session_todo,
    complete_session_todo,
    abandon_session_todo,
]
"""Default TODO tools added to every agent via ``Agent._collect_tools``."""


__all__ = [
    "TODO_TOOLS",
    "abandon_session_todo",
    "complete_session_todo",
    "create_session_todo",
    "list_session_todos",
    "update_session_todo",
]
