"""Tests for session-scoped TODO persistence and tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._func import wrap_function
from thorn.core._provider import MockProvider
from thorn.runtime import Runtime
from thorn.runtime._notification import NotificationID
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._todo import SessionTodoList, TodoStatus
from thorn.runtime._todo_tools import (
    abandon_session_todo,
    complete_session_todo,
    create_session_todo,
    list_session_todos,
    update_session_todo,
)

AGENT_ID = AgentID("todo-agent")
SESSION_KEY = SessionKey("project/issue-7")


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    return Runtime(
        provider=MockProvider(),
        workspace_root=tmp_path / "ws",
    )


@pytest.fixture
def agent() -> Agent:
    return Agent(id=AGENT_ID, name="todoer")


@pytest.fixture
def scoped_ctx(
    runtime: Runtime,
    agent: Agent,
) -> Iterator[ExecutionContext]:
    base = runtime.create_context()
    scoped = base.push_scope(
        "test-agent-scope",
        agent=agent,
        session_key=str(SESSION_KEY),
    )
    token = set_context(scoped)
    try:
        yield scoped
    finally:
        reset_context(token)


def _todo_id_from(result: str) -> str:
    match = re.search(r"`(todo-[A-Za-z0-9_-]+)`", result)
    assert match is not None, result
    return match.group(1)


class TestSessionTodoList:
    def test_persists_to_readable_markdown(self, tmp_path: Path) -> None:
        todo_file = tmp_path / "TODO.md"
        todos = SessionTodoList(todo_file)

        created = todos.create(
            title="Add regression coverage",
            linked_inbox_item_id=NotificationID("01HZYTODOITEM"),
            notes="Cover the close-out guard.",
        )

        assert todo_file.is_file()
        text = todo_file.read_text(encoding="utf-8")
        assert "# Session TODOs" in text
        assert "Add regression coverage" in text
        assert "01HZYTODOITEM" in text

        reloaded = SessionTodoList(todo_file).list_items()
        assert [item.id for item in reloaded] == [created.id]
        assert reloaded[0].title == "Add regression coverage"
        assert reloaded[0].status is TodoStatus.OPEN
        assert reloaded[0].linked_inbox_item_ids == (
            NotificationID("01HZYTODOITEM"),
        )

    def test_completion_and_abandonment_require_rationale(
        self,
        tmp_path: Path,
    ) -> None:
        todos = SessionTodoList(tmp_path / "TODO.md")
        completed = todos.create(title="Finish implementation")
        abandoned = todos.create(title="Try discarded approach")

        todos.complete(completed.id, rationale="Tests and review are clean.")
        todos.abandon(abandoned.id, rationale="No longer matches the design.")

        statuses = {
            item.id: item.status
            for item in SessionTodoList(tmp_path / "TODO.md").list_items()
        }
        assert statuses[completed.id] is TodoStatus.COMPLETED
        assert statuses[abandoned.id] is TodoStatus.ABANDONED
        assert todos.unresolved_linked_to(NotificationID("missing")) == []

    def test_open_linked_items_are_reported(self, tmp_path: Path) -> None:
        todos = SessionTodoList(tmp_path / "TODO.md")
        open_item = todos.create(
            title="Open linked work",
            linked_inbox_item_id=NotificationID("01LINKED"),
        )
        closed_item = todos.create(
            title="Closed linked work",
            linked_inbox_item_id=NotificationID("01LINKED"),
        )
        todos.complete(closed_item.id, rationale="Done.")

        unresolved = todos.unresolved_linked_to(NotificationID("01LINKED"))

        assert [item.id for item in unresolved] == [open_item.id]


class TestTodoToolSchema:
    def test_todo_tools_are_in_process(self) -> None:
        for tool_fn in (
            create_session_todo,
            list_session_todos,
            update_session_todo,
            complete_session_todo,
            abandon_session_todo,
        ):
            assert wrap_function(tool_fn).venue.name == "IN_PROCESS"


class TestSessionTodoTools:
    async def test_create_list_update_complete_and_abandon(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        todo_id_suffixes = iter(("InAyU1b-", "oldShape-"))
        monkeypatch.setattr(
            "thorn.runtime._todo.secrets.token_urlsafe",
            lambda _bytes: next(todo_id_suffixes),
        )

        created = await create_session_todo(
            "Write black-box tests",
            linked_inbox_item_id="01INBOX",
            notes="Start with user-visible behavior.",
        )
        todo_id = _todo_id_from(created)
        assert todo_id == "todo-InAyU1b-"

        listing = await list_session_todos()
        assert f"`{todo_id}`" in listing
        assert "open" in listing
        assert "Write black-box tests" in listing
        assert "01INBOX" in listing

        updated = await update_session_todo(
            todo_id,
            title="Write regression tests",
            notes="Covers persistence and guard behavior.",
        )
        assert "updated" in updated
        assert f"`{todo_id}`" in updated
        listing = await list_session_todos()
        assert "Write regression tests" in listing
        assert "Covers persistence" in listing

        cleared_link = await update_session_todo(
            todo_id,
            linked_inbox_item_id="",
        )
        assert "updated" in cleared_link
        assert f"`{todo_id}`" in cleared_link
        listing = await list_session_todos()
        assert "01INBOX" not in listing

        completed = await complete_session_todo(
            todo_id,
            rationale="Regression tests are in place.",
        )
        assert "completed" in completed
        assert f"`{todo_id}`" in completed
        assert "No open session TODOs" in await list_session_todos()

        abandoned_created = await create_session_todo("Investigate old shape")
        abandoned_id = _todo_id_from(abandoned_created)
        abandoned = await abandon_session_todo(
            abandoned_id,
            rationale="The selected design superseded it.",
        )
        assert "abandoned" in abandoned
        assert f"`{abandoned_id}`" in abandoned

        all_items = await list_session_todos(status_filter="all")
        assert f"`{todo_id}`" in all_items
        assert f"`{abandoned_id}`" in all_items
        assert "completed" in all_items
        assert "abandoned" in all_items

        todo_file = runtime.paths.session_todo_file(AGENT_ID, SESSION_KEY)
        assert todo_file.is_file()

    async def test_errors_without_session_scope(
        self,
        runtime: Runtime,
        agent: Agent,
    ) -> None:
        base = runtime.create_context()
        scoped = base.push_scope("no-session", agent=agent)
        token = set_context(scoped)
        try:
            result = await list_session_todos()
        finally:
            reset_context(token)

        assert "Error" in result
        assert "session" in result.lower()

    async def test_completion_and_abandonment_require_rationale(
        self,
        scoped_ctx: ExecutionContext,
    ) -> None:
        created = await create_session_todo("Resolve explicitly")
        todo_id = _todo_id_from(created)

        completion_result = await complete_session_todo(todo_id, rationale="")
        abandon_result = await abandon_session_todo(todo_id, rationale="")

        assert "Error" in completion_result
        assert "rationale" in completion_result
        assert "Error" in abandon_result
        assert "rationale" in abandon_result
