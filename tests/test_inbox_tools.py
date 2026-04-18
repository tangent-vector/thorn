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

from pathlib import Path
from typing import Iterator

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._provider import MockProvider
from thorn.runtime import (
    Runtime,
    SessionAddress,
    SessionInbox,
)
from thorn.runtime._inbox_tools import (
    list_inbox_items,
    read_inbox_item,
    update_inbox_item,
)
from thorn.runtime._notification import (
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._address import ServiceAddress
from thorn.runtime._session import AgentID, SessionKey


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

    async def test_shows_error_reason(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("failing"))
        inbox.update_status(
            item.id, NotificationStatus.ERRORED, error_reason="disk full",
        )
        result = await read_inbox_item(item.id)
        assert "Error reason: disk full" in result


# ---------------------------------------------------------------------------
# update_inbox_item
# ---------------------------------------------------------------------------

class TestUpdateInboxItem:
    async def test_in_progress_keeps_item_and_updates_status(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        result = await update_inbox_item(
            item.id, "in_progress", notes="starting",
        )
        assert "in_progress" in result
        updated = inbox.get(item.id)
        assert updated.status is NotificationStatus.IN_PROGRESS
        assert updated.notes == "starting"

    async def test_handled_deletes_item_without_rsvp(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("work"))
        result = await update_inbox_item(
            item.id, "handled", notes="done",
        )
        assert "handled" in result
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
        await update_inbox_item(item.id, "handled", notes="rsvp-done")

        assert inbox.list() == []
        forge_items = forge.list()
        assert [n.id for n in forge_items] == [item.id]
        assert forge_items[0].notes == "rsvp-done"

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
        result = await update_inbox_item(
            item.id, "errored", notes="out of scope",
        )
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
        result = await update_inbox_item("NO-SUCH-ID", "handled")
        assert "Error" in result

    async def test_unregistered_rsvp_returns_warning(
        self, scoped_ctx: ExecutionContext, inbox: SessionInbox,
    ) -> None:
        # Post an item that demands RSVP to a service that isn't
        # registered.  Step 1 lands; step 2 fails with DispatchError;
        # the tool surfaces that as a warning string.
        ghost = ServiceAddress("ghost")
        item = inbox.post(_spec("ghost-rsvp", rsvp_to=ghost))
        result = await update_inbox_item(item.id, "handled", notes="n")
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
