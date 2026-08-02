"""Unit tests for the agent-facing inbox tools.

Tools are tested by directly awaiting them inside a manufactured
``ExecutionContext`` that carries the ambient state the tools expect:

- ``ctx.runtime`` -- a :class:`Runtime` (with its address book
  populated by the fixture).
- ``ctx.agent`` -- an :class:`Agent` with an ID.
- ``ctx.scope`` -- a scope carrying ``session_key`` metadata, as the
  agent loop sets up in ``_run_session_prompt``.

The fixtures are deliberately minimal: they build up the context by
hand rather than running an actual prompt loop, which keeps the tests
focused on the tool logic itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._func import wrap_function
from thorn.core._provider import MockProvider
from thorn.runtime import (
    Runtime,
    SessionAddress,
    SessionInbox,
)
from thorn.runtime._address import ServiceAddress
from thorn.runtime._inbox_tools import (
    INBOX_TOOLS,
    complete_focused_work,
    list_inbox_items,
    park_focused_work,
    read_inbox_item,
    update_focus,
    update_inbox_item,
)
from thorn.runtime._notification import (
    InboxCompletionRationale,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._todo_tools import (
    complete_session_todo,
    create_session_todo,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AGENT_ID = AgentID("tool-test-agent")
SESSION_KEY = SessionKey("demo/session")


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    return Runtime(
        provider=MockProvider(),
        workspace_root=tmp_path / "ws",
    )


@pytest.fixture
def agent() -> Agent:
    return Agent(id=AGENT_ID, name="tester")


@pytest.fixture
def inbox(runtime: Runtime) -> SessionInbox:
    """Build and register a SessionInbox for the test agent/session."""
    address = SessionAddress(AGENT_ID, SESSION_KEY)
    sib = SessionInbox(
        runtime.paths.session_inbox_dir(AGENT_ID, SESSION_KEY),
        address,
        in_flight_index=runtime.in_flight_index,
    )
    runtime.address_book.register(address, sib)
    return sib


@pytest.fixture
def scoped_ctx(
    runtime: Runtime, agent: Agent,
) -> Iterator[ExecutionContext]:
    """Install an ExecutionContext with runtime + agent + session_key."""
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


def _spec(content: str = "hello", **fields) -> NotificationSpec:
    from thorn.runtime._address import Address
    target: Address = fields.pop(
        "target", SessionAddress(AGENT_ID, SESSION_KEY),
    )
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        metadata=fields.pop("metadata", {}),
        rsvp_to=fields.pop("rsvp_to", None),
        external_key=fields.pop("external_key", None),
    )


def _completion(
    *,
    completed_actions: tuple[str, ...] = ("opened change request !1",),
    request_coverage: tuple[str, ...] = ("checked every requested action",),
    remaining_work: tuple[str, ...] = (),
    self_review: str = "reviewed the inbox item and final diff",
    external_follow_up: tuple[str, ...] = ("posted a source comment",),
) -> InboxCompletionRationale:
    return InboxCompletionRationale(
        completed_actions=completed_actions,
        request_coverage=request_coverage,
        remaining_work=remaining_work,
        self_review=self_review,
        external_follow_up=external_follow_up,
    )


def _todo_id_from(result: str) -> str:
    match = re.search(r"`(todo-[A-Za-z0-9_-]+)`", result)
    assert match is not None, result
    return match.group(1)


async def _focus_item_for_closeout(item_id: str) -> None:
    focused = await update_focus(
        phase="inspect",
        item_id=item_id,
        objective="Handle the test inbox item.",
        notes="starting",
    )
    assert "Focus updated" in focused
    closeout = await update_focus(
        phase="closeout",
        no_validation_rationale="unit test sets closeout state directly",
    )
    assert "phase=closeout" in closeout


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

class TestToolSchema:
    def test_default_inbox_tools_are_focused_workflow(self) -> None:
        tool_names = [tool.__name__ for tool in INBOX_TOOLS]
        assert tool_names == [
            "list_inbox_items",
            "read_inbox_item",
            "update_focus",
            "complete_focused_work",
            "park_focused_work",
        ]

    def test_complete_focused_work_schema_describes_closeout_policy(self) -> None:
        wrapped = wrap_function(complete_focused_work)
        description = " ".join(
            wrapped.schema["function"]["description"].lower().split()
        )
        assert "focused inbox item" in description
        assert "closeout" in description
        assert "validation evidence" in description
        assert "linked session todos" in description
        assert "self-review" in description
        assert "structured" in description
        params = wrapped.schema["function"]["parameters"]
        assert "completion" in params["properties"]
        schema_text = str(params["properties"]["completion"]).lower()
        assert "completed_actions" in schema_text
        assert "request_coverage" in schema_text
        assert "remaining_work" in schema_text
        assert "self_review" in schema_text
        assert "external_follow_up" in schema_text

    def test_update_inbox_item_schema_is_terminal_compatibility_only(self) -> None:
        wrapped = wrap_function(update_inbox_item)
        description = " ".join(
            wrapped.schema["function"]["description"].lower().split()
        )
        assert "compatibility helper" in description
        assert "not part of the default agent-facing toolset" in description
        assert "update_focus" in description
        assert "complete_focused_work" in description
        assert "park_focused_work" in description
        assert "in_progress" in description
        params = wrapped.schema["function"]["parameters"]
        status_schema = params["properties"]["status"]
        status_schema_text = str(status_schema).lower()
        assert set(status_schema["enum"]) == {"handled", "errored"}
        assert "in_progress" not in status_schema["enum"]
        assert "handled" in status_schema_text
        assert "errored" in status_schema_text


# ---------------------------------------------------------------------------
# list_inbox_items
# ---------------------------------------------------------------------------

class TestListInboxItems:
    async def test_empty_inbox(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        result = await list_inbox_items()
        assert result == "Your inbox is empty."

    async def test_lists_pending_items(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        inbox.post(_spec("first item"))
        inbox.post(_spec("second item"))
        result = await list_inbox_items()
        assert "2 inbox item(s)" in result
        assert "first item" in result
        assert "second item" in result
        assert "status=pending" in result

    async def test_excludes_handled_and_errored(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        visible = inbox.post(_spec("visible"))
        hidden = inbox.post(_spec("hidden"))
        inbox.update_status(
            hidden.id, NotificationStatus.HANDLED, notes="done",
        )
        result = await list_inbox_items()
        assert visible.id in result
        assert hidden.id not in result

    async def test_includes_in_progress(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("active"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        result = await list_inbox_items()
        assert "status=in_progress" in result
        assert item.id in result

    async def test_shows_linked_todo_summary(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("active"))
        await create_session_todo(
            "finish the acceptance tests",
            linked_inbox_item_id=item.id,
        )

        result = await list_inbox_items()

        assert "TODOs: 1 open" in result
        assert "finish the acceptance tests" in result


# ---------------------------------------------------------------------------
# read_inbox_item
# ---------------------------------------------------------------------------

class TestReadInboxItem:
    async def test_reads_existing_item(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("some content"))
        result = await read_inbox_item(item.id)
        assert f"Item: {item.id}" in result
        assert "Status: pending" in result
        assert "Source: test" in result
        assert "some content" in result

    async def test_missing_id_returns_error(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        result = await read_inbox_item("NO-SUCH-ID")
        assert "Error" in result and "NO-SUCH-ID" in result

    async def test_traversal_id_returns_error(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        result = await read_inbox_item("../forged")
        assert "Error" in result and "../forged" in result

    async def test_shows_error_reason(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("failing"))
        inbox.update_status(
            item.id, NotificationStatus.ERRORED, error_reason="disk full",
        )
        result = await read_inbox_item(item.id)
        assert "Error reason: disk full" in result

    async def test_shows_completion_rationale(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("done"))
        completion = _completion()
        inbox.update_status(
            item.id,
            NotificationStatus.HANDLED,
            completion_rationale=completion,
        )

        result = await read_inbox_item(item.id)

        assert "Completion rationale" in result
        assert "opened change request !1" in result
        assert "reviewed the inbox item" in result


# ---------------------------------------------------------------------------
# update_inbox_item
# ---------------------------------------------------------------------------

class TestUpdateInboxItem:
    async def test_update_inbox_item_rejects_in_progress_claim(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        result = await update_inbox_item(
            item.id, "in_progress", notes="starting",  # type: ignore[arg-type]
        )
        assert "Error" in result
        assert "update_focus" in result
        updated = inbox.get(item.id)
        assert updated.status is NotificationStatus.PENDING
        assert updated.notes is None

    async def test_handled_requires_completion(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        result = await update_inbox_item(
            item.id, "handled", notes="done",
        )
        assert "Error" in result and "completion is required" in result
        assert inbox.get(item.id).status is NotificationStatus.PENDING

    async def test_handled_requires_focused_closeout(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        result = await update_inbox_item(
            item.id, "handled", completion=_completion(),
        )
        assert "Error" in result
        assert "update_focus" in result
        assert inbox.get(item.id).status is NotificationStatus.PENDING

    async def test_handled_deletes_item_without_rsvp(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        await _focus_item_for_closeout(item.id)
        result = await complete_focused_work(_completion(), notes="done")
        assert "handled" in result
        assert inbox.list() == []

    async def test_handled_rejects_open_linked_todos(
        self,
        scoped_ctx: ExecutionContext,
        inbox: SessionInbox,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "thorn.runtime._todo.secrets.token_urlsafe",
            lambda _bytes: "Linked1-",
        )
        item = inbox.post(_spec("work"))
        await _focus_item_for_closeout(item.id)
        created = await create_session_todo(
            "finish linked work",
            linked_inbox_item_id=item.id,
        )
        todo_id = _todo_id_from(created)
        assert todo_id == "todo-Linked1-"

        blocked = await complete_focused_work(_completion())

        assert "Error" in blocked
        assert "open TODO" in blocked
        assert f"`{todo_id}`" in blocked
        assert inbox.get(item.id).status is NotificationStatus.IN_PROGRESS

        completed = await complete_session_todo(
            todo_id, rationale="Linked work is done.",
        )
        assert completed == f"TODO `{todo_id}` completed."

        handled = await complete_focused_work(_completion())

        assert handled == f"Item {item.id} is now handled."
        assert inbox.list() == []

    async def test_handled_with_rsvp_moves_to_target(
        self,
        scoped_ctx: ExecutionContext,
        inbox: SessionInbox,
        runtime: Runtime,
        tmp_path: Path,
    ) -> None:
        forge_addr = ServiceAddress("forge")
        forge = NotificationQueue(
            runtime.paths.service_queue_dir("forge"), forge_addr,
            in_flight_index=runtime.in_flight_index,
        )
        runtime.address_book.register(forge_addr, forge)

        item = inbox.post(_spec("work", rsvp_to=forge_addr))
        completion = _completion()
        await _focus_item_for_closeout(item.id)
        await complete_focused_work(completion, notes="rsvp-done")

        assert inbox.list() == []
        forge_items = forge.list()
        assert [n.id for n in forge_items] == [item.id]
        assert forge_items[0].notes == "rsvp-done"
        assert forge_items[0].completion_rationale == completion

    async def test_handled_rejects_incomplete_completion(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        await _focus_item_for_closeout(item.id)
        result = await complete_focused_work(
            _completion(completed_actions=()),
        )

        assert "Error" in result
        assert "completed_actions" in result
        assert inbox.get(item.id).status is NotificationStatus.IN_PROGRESS

    async def test_errored_requires_notes(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("fail"))
        result = await update_inbox_item(item.id, "errored")
        assert "Error" in result and "notes is required" in result
        # Item unchanged.
        assert inbox.get(item.id).status is NotificationStatus.PENDING

    async def test_errored_with_notes_moves_to_errored_dir(
        self,
        scoped_ctx: ExecutionContext,
        inbox: SessionInbox,
        runtime: Runtime,
    ) -> None:
        item = inbox.post(_spec("fail"))
        await update_focus(
            phase="inspect",
            item_id=item.id,
            objective="Park this failing item.",
        )
        result = await park_focused_work(notes="out of scope")
        assert "errored" in result
        assert inbox.list() == []
        errored_dir = runtime.paths.session_inbox_errored_dir(
            AGENT_ID, SESSION_KEY,
        )
        assert errored_dir.is_dir()
        from thorn.runtime._queue import DurableQueue
        park = DurableQueue(errored_dir)
        parked = park.list()
        assert [n.id for n in parked] == [item.id]
        assert parked[0].error_reason == "out of scope"

    async def test_invalid_status_returns_error(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("x"))
        # Caller bypasses the Literal by passing a raw string.  The
        # tool should reject it gracefully.
        result = await update_inbox_item(item.id, "weird")  # type: ignore[arg-type]
        assert "Error" in result and "invalid status" in result

    async def test_missing_id_returns_error(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        result = await update_inbox_item(
            "NO-SUCH-ID", "handled", completion=_completion(),
        )
        assert "Error" in result

    async def test_traversal_id_returns_error(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        result = await update_inbox_item(
            "../forged", "handled", completion=_completion(),
        )
        assert "Error" in result and "../forged" in result

    async def test_unregistered_rsvp_returns_warning(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        # Post an item that demands RSVP to a service that isn't
        # registered.  Step 1 lands; step 2 fails with DispatchError;
        # the tool surfaces that as a warning string.
        ghost = ServiceAddress("ghost")
        item = inbox.post(_spec("ghost-rsvp", rsvp_to=ghost))
        await _focus_item_for_closeout(item.id)
        result = await complete_focused_work(_completion(), notes="n")
        assert "Warning" in result
        # Item is now handled but still on disk for the sweep.
        stuck = inbox.get(item.id)
        assert stuck.status is NotificationStatus.HANDLED


# ---------------------------------------------------------------------------
# Missing ambient state
# ---------------------------------------------------------------------------

class TestMissingAmbient:
    async def test_no_context_reports_error(self) -> None:
        # No context pushed -> get_context raises RuntimeError inside
        # the tool's resolver, which returns a clear string.
        result = await list_inbox_items()
        assert "Error" in result and "execution context" in result

    async def test_no_runtime_reports_error(
        self, mock_provider: MockProvider,
    ) -> None:
        ctx = ExecutionContext(provider=mock_provider)
        token = set_context(ctx)
        try:
            result = await list_inbox_items()
        finally:
            reset_context(token)
        assert "Error" in result and "runtime" in result.lower()

    async def test_no_session_scope_reports_error(
        self, runtime: Runtime, agent: Agent,
    ) -> None:
        base = runtime.create_context()
        scoped = base.push_scope(
            "no-session-scope", agent=agent,
        )  # deliberately omit session_key
        token = set_context(scoped)
        try:
            result = await list_inbox_items()
        finally:
            reset_context(token)
        assert "Error" in result and "session" in result.lower()

    async def test_unregistered_inbox_reports_error(
        self, runtime: Runtime, agent: Agent,
    ) -> None:
        # Agent + session scope present, but no inbox registered.
        base = runtime.create_context()
        scoped = base.push_scope(
            "no-inbox",
            agent=agent,
            session_key=str(SESSION_KEY),
        )
        token = set_context(scoped)
        try:
            result = await list_inbox_items()
        finally:
            reset_context(token)
        assert "Error" in result and "not wired" in result
