"""Tests for session working-set state and rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from thorn.runtime._notification import NotificationID
from thorn.runtime._todo import TodoID, TodoItem, TodoStatus
from thorn.runtime._working_set import (
    ActiveContextDetailLevel,
    ActiveContextEntry,
    ActiveContextKind,
    HandlingPhase,
    LastValidation,
    ValidationOutcome,
    WorkingSet,
    WorkingSetBlocker,
    open_todo_items,
    render_working_set_block,
)


def _todo(
    todo_id: str,
    title: str,
    *,
    status: TodoStatus = TodoStatus.OPEN,
) -> TodoItem:
    now = datetime(2026, 6, 22, tzinfo=timezone.utc)
    return TodoItem(
        id=TodoID(todo_id),
        title=title,
        status=status,
        linked_inbox_item_ids=(NotificationID("01INBOX"),),
        notes=None,
        resolution_rationale=None,
        created_at=now,
        updated_at=now,
    )


def test_default_working_set_renders_compact_intake_state() -> None:
    rendered = render_working_set_block(WorkingSet())

    assert "Phase: intake" in rendered.text
    assert "Focused inbox item: none" in rendered.text
    assert "Open TODOs:\n- none" in rendered.text
    assert "Active context:\n- none" in rendered.text
    assert "Validation: not recorded" in rendered.text
    assert rendered.diagnostics == ()


def test_render_includes_open_todos_context_and_validation() -> None:
    working_set = WorkingSet(
        phase=HandlingPhase.VALIDATE,
        focused_inbox_item_id=NotificationID("01FOCUS"),
        objective="Fix the flaky TODO lifecycle test.",
        last_action_summary="patched the TODO lifecycle implementation",
        active_context=(
            ActiveContextEntry(
                kind=ActiveContextKind.FILE,
                label="tests/test_todos.py",
                detail_level=ActiveContextDetailLevel.SPAN,
                summary="failing lifecycle test",
            ),
        ),
        last_validation=LastValidation(
            outcome=ValidationOutcome.PASSED,
            command="uv run pytest tests/test_todos.py -q",
            summary="targeted TODO tests passed",
        ),
        override_rationale="validation-only follow-up requested by operator",
    )

    rendered = render_working_set_block(
        working_set,
        open_todos=open_todo_items([
            _todo("todo-a", "write regression coverage"),
            _todo("todo-b", "old completed work", status=TodoStatus.COMPLETED),
        ]),
    )

    assert "Phase: validate" in rendered.text
    assert "Focused inbox item: 01FOCUS" in rendered.text
    assert "Objective: Fix the flaky TODO lifecycle test." in rendered.text
    assert "Last action: patched the TODO lifecycle implementation" in rendered.text
    assert "- todo-a: write regression coverage" in rendered.text
    assert "todo-b" not in rendered.text
    assert "file tests/test_todos.py (span): failing lifecycle test" in rendered.text
    assert "Last validation: passed: targeted TODO tests passed" in rendered.text
    assert "Validation command: uv run pytest tests/test_todos.py -q" in rendered.text
    assert (
        "Override rationale: validation-only follow-up requested by operator"
        in rendered.text
    )


def test_render_surfaces_inconsistent_state_diagnostics() -> None:
    working_set = WorkingSet(
        phase=HandlingPhase.INTAKE,
        focused_inbox_item_id=NotificationID("01FOCUS"),
        blocker=WorkingSetBlocker(
            summary="needs credentials",
            unblock_condition="operator adds token",
        ),
    )

    rendered = render_working_set_block(
        working_set,
        todo_diagnostic="session TODO list is unreadable",
    )

    assert "Diagnostics:" in rendered.text
    assert "phase is intake but a focused inbox item is set" in rendered.text
    assert "blocker is recorded while phase is not blocked" in rendered.text
    assert "session TODO list is unreadable" in rendered.text
    assert len(rendered.diagnostics) == 3


def test_render_truncates_individual_text_fields() -> None:
    long_prefix = "x" * 400
    hidden_tail = "SHOULD_NOT_RENDER"
    working_set = WorkingSet(objective=f"{long_prefix} {hidden_tail}")

    rendered = render_working_set_block(
        working_set,
        open_todos=(_todo("todo-a", f"{long_prefix} {hidden_tail}"),),
    )

    assert hidden_tail not in rendered.text
    assert "- todo-a: " in rendered.text
    assert "..." in rendered.text


def test_working_set_roundtrips_json_data() -> None:
    working_set = WorkingSet(
        phase=HandlingPhase.BLOCKED,
        focused_inbox_item_id=NotificationID("01FOCUS"),
        objective="Wait for operator input.",
        last_action_summary="asked the operator for a policy decision",
        active_context=(
            ActiveContextEntry(
                kind=ActiveContextKind.DIRECTORY,
                label="src/thorn/runtime",
                detail_level=ActiveContextDetailLevel.DIRECTORY,
                summary="runtime state implementation area",
            ),
        ),
        no_validation_rationale="blocked before code changed",
        blocker=WorkingSetBlocker(
            summary="missing policy decision",
            unblock_condition="human decides transition gate behavior",
        ),
        override_rationale="blocked before validation could produce evidence",
    )

    restored = WorkingSet.from_data(working_set.to_data())

    assert restored == working_set
