"""Unit tests for :mod:`thorn.runtime._prompt_format`.

Split in two halves:

- Pure-formatting tests exercise :func:`build_inbox_prompt` and
  :func:`summarize_notification_content`.  These build
  :class:`Notification` objects directly (bypassing a queue) so the
  format can be audited in isolation.
- Dispatcher tests exercise :func:`inbox_prompt_dispatcher` against a
  real :class:`SessionInbox` on ``tmp_path`` and a hand-rolled
  session stub with a captured ``prompt`` coroutine.  We use a stub
  rather than a real :class:`Session` because spinning up the full
  agent prompt loop would drag in :class:`ExecutionContext` and a
  provider, which is far outside the dispatcher's own contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from thorn.runtime._address import SessionAddress
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    Notification,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._prompt_format import (
    build_inbox_prompt,
    inbox_prompt_dispatcher,
    summarize_notification_content,
)
from thorn.runtime._session import AgentID, SessionKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _addr(key: str = "proj/s") -> SessionAddress:
    return SessionAddress(AgentID("coord"), SessionKey(key))


def _make_notification(
    *,
    id: str = "01HZY00000000000000000000A",
    source: str = "gitlab-poller",
    content: str = "hello",
    status: NotificationStatus = NotificationStatus.PENDING,
    notes: str | None = None,
    error_reason: str | None = None,
    target: SessionAddress | None = None,
) -> Notification:
    return Notification(
        id=id,
        source=source,
        content=content,
        target=target or _addr(),
        metadata={},
        rsvp_to=None,
        external_key=None,
        posted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=status,
        attempt_count=0,
        notes=notes,
        error_reason=error_reason,
    )


# ---------------------------------------------------------------------------
# summarize_notification_content
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_short_single_line_returned_unchanged(self) -> None:
        n = _make_notification(content="fix the widget")
        assert summarize_notification_content(n) == "fix the widget"

    def test_strips_surrounding_whitespace(self) -> None:
        n = _make_notification(content="   padded   ")
        assert summarize_notification_content(n) == "padded"

    def test_multi_line_takes_first_line_only(self) -> None:
        n = _make_notification(content="line one\nline two\nline three")
        assert summarize_notification_content(n) == "line one"

    def test_long_line_truncated_with_ellipsis(self) -> None:
        n = _make_notification(content="x" * 200)
        summary = summarize_notification_content(n)
        assert len(summary) <= 80
        assert summary.endswith("\u2026")

    def test_empty_content_renders_placeholder(self) -> None:
        n = _make_notification(content="")
        assert summarize_notification_content(n) == "(empty content)"

    def test_whitespace_only_content_renders_placeholder(self) -> None:
        n = _make_notification(content="   \n\t  ")
        assert summarize_notification_content(n) == "(empty content)"


# ---------------------------------------------------------------------------
# build_inbox_prompt
# ---------------------------------------------------------------------------

class TestBuildInboxPromptSingleItem:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            build_inbox_prompt([])

    def test_single_item_includes_id_source_status(self) -> None:
        n = _make_notification(
            id="01ITEM1",
            source="gitlab-poller",
            content="Please review MR !42.",
        )
        text = build_inbox_prompt([n])
        assert "01ITEM1" in text
        assert "gitlab-poller" in text
        assert "pending" in text
        assert "Please review MR !42." in text

    def test_single_item_close_out_references_id(self) -> None:
        # The close-out instruction must include the item's exact ID
        # so the agent can call update_inbox_item without guessing.
        n = _make_notification(id="01XYZ")
        text = build_inbox_prompt([n])
        assert 'update_inbox_item("01XYZ", "handled")' in text

    def test_single_item_does_not_emit_summary_list_header(self) -> None:
        # The multi-item counts header would be misleading here.
        n = _make_notification()
        text = build_inbox_prompt([n])
        assert "You have 1 inbox item" not in text

    def test_single_in_progress_with_notes_surfaces_prior_notes(self) -> None:
        n = _make_notification(
            status=NotificationStatus.IN_PROGRESS,
            notes="started drafting a response",
        )
        text = build_inbox_prompt([n])
        assert "started drafting a response" in text
        assert "prior notes" in text.lower()

    def test_single_in_progress_without_notes_has_no_notes_line(self) -> None:
        n = _make_notification(
            status=NotificationStatus.IN_PROGRESS,
            notes=None,
        )
        text = build_inbox_prompt([n])
        assert "prior notes" not in text.lower()

    def test_single_pending_with_notes_does_not_surface_notes(self) -> None:
        # Notes on a pending item would only be present if the
        # lifecycle somehow regressed; defensively, we do not surface
        # them on pending because the "prior notes" framing is
        # meaningless (the agent has not worked on it yet).
        n = _make_notification(
            status=NotificationStatus.PENDING,
            notes="should not show",
        )
        text = build_inbox_prompt([n])
        assert "should not show" not in text

    def test_single_item_full_content_verbatim(self) -> None:
        # Multi-line content must survive intact in the single-item
        # form (no truncation, unlike the multi-item summary form).
        content = "line one\nline two\nline three"
        n = _make_notification(content=content)
        text = build_inbox_prompt([n])
        assert content in text


class TestBuildInboxPromptMultiItem:
    def test_header_reports_counts(self) -> None:
        items = [
            _make_notification(id="01A", status=NotificationStatus.PENDING),
            _make_notification(id="01B", status=NotificationStatus.PENDING),
            _make_notification(id="01C", status=NotificationStatus.IN_PROGRESS),
        ]
        text = build_inbox_prompt(items)
        assert "3 inbox item" in text
        assert "2 pending" in text
        assert "1 in progress" in text

    def test_each_item_appears_as_summary_line(self) -> None:
        items = [
            _make_notification(id="01A", source="s1", content="first"),
            _make_notification(id="01B", source="s2", content="second"),
        ]
        text = build_inbox_prompt(items)
        assert "[01A]" in text and "first" in text and "source=s1" in text
        assert "[01B]" in text and "second" in text and "source=s2" in text

    def test_items_appear_in_caller_supplied_order(self) -> None:
        # build_inbox_prompt does no sorting of its own; the caller
        # (SessionInbox.prompt_pending) is responsible for post order.
        items = [
            _make_notification(id="01Z", content="last"),
            _make_notification(id="01A", content="first"),
        ]
        text = build_inbox_prompt(items)
        pos_z = text.index("01Z")
        pos_a = text.index("01A")
        assert pos_z < pos_a

    def test_in_progress_notes_annotation_present(self) -> None:
        items = [
            _make_notification(id="01A"),
            _make_notification(
                id="01B",
                status=NotificationStatus.IN_PROGRESS,
                notes="halfway done",
            ),
        ]
        text = build_inbox_prompt(items)
        assert "[notes: halfway done]" in text

    def test_pending_notes_not_annotated(self) -> None:
        # Only in_progress items should get the [notes: ...] hint.
        # A pending item with notes (rare but possible) should not
        # confuse the agent with stale context.
        items = [
            _make_notification(id="01A", notes="stale"),
            _make_notification(id="01B"),
        ]
        text = build_inbox_prompt(items)
        assert "[notes:" not in text

    def test_multi_item_close_out_mentions_inbox_tools(self) -> None:
        items = [_make_notification(id="01A"), _make_notification(id="01B")]
        text = build_inbox_prompt(items)
        assert "read_inbox_item" in text
        assert "update_inbox_item" in text

    def test_multi_item_long_content_is_truncated_in_summary(self) -> None:
        items = [
            _make_notification(id="01A", content="x" * 300),
            _make_notification(id="01B", content="short"),
        ]
        text = build_inbox_prompt(items)
        # No individual summary line should exceed the summary cap
        # plus a small formatting envelope.
        for line in text.splitlines():
            if line.startswith("- [01A]"):
                # Line is "- [id] status=... source=...: <summary>"
                # The summary portion is bounded; assert a generous
                # upper bound on the whole line to detect gross
                # overflow.
                assert len(line) < 200

    def test_multi_item_in_progress_notes_annotation_truncated(self) -> None:
        # Extremely long notes on an in_progress item get truncated
        # so one noisy item cannot balloon the prompt.
        items = [
            _make_notification(id="01A"),
            _make_notification(
                id="01B",
                status=NotificationStatus.IN_PROGRESS,
                notes="y" * 300,
            ),
        ]
        text = build_inbox_prompt(items)
        for line in text.splitlines():
            if line.startswith("- [01B]"):
                assert len(line) < 250
                assert "[notes:" in line


# ---------------------------------------------------------------------------
# inbox_prompt_dispatcher
# ---------------------------------------------------------------------------

class _PromptAccessor:
    """Callable stub matching ``Session.prompt``'s shape for tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, text: str, **kwargs: object) -> str:
        self.calls.append(text)
        return ""


class _FakeSession:
    """Minimal duck-typed Session for dispatcher tests.

    The dispatcher's only interactions with a session are
    ``session.prompt(text)`` and ``session.agent.id`` / ``session.key``
    for logging.  A bare class with those attributes is sufficient
    and avoids spinning up the full prompt machinery.
    """

    def __init__(self, agent_id: str, session_key: str) -> None:
        class _Agent:
            id = AgentID(agent_id)

        self.agent = _Agent()
        self.key = SessionKey(session_key)
        self.prompt = _PromptAccessor()


def _spec(target: SessionAddress, content: str = "hi") -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        metadata={},
    )


@pytest.mark.asyncio
class TestInboxPromptDispatcher:
    async def test_no_items_pending_does_not_call_prompt(
        self, tmp_path: Path,
    ) -> None:
        addr = _addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)
        session = _FakeSession("coord", "proj/s")

        await inbox_prompt_dispatcher(session, inbox)  # type: ignore[arg-type]

        assert session.prompt.calls == []

    async def test_single_pending_item_is_prompted(
        self, tmp_path: Path,
    ) -> None:
        addr = _addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)
        inbox.post(_spec(addr, content="please do X"))
        session = _FakeSession("coord", "proj/s")

        await inbox_prompt_dispatcher(session, inbox)  # type: ignore[arg-type]

        assert len(session.prompt.calls) == 1
        text = session.prompt.calls[0]
        assert "please do X" in text
        # The single-item prompt must reference the item's ID so the
        # close-out instruction is actionable.
        pending = inbox.prompt_pending()
        assert pending[0].id in text

    async def test_multiple_pending_items_are_batched_into_one_prompt(
        self, tmp_path: Path,
    ) -> None:
        addr = _addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)
        inbox.post(_spec(addr, content="first task"))
        inbox.post(_spec(addr, content="second task"))
        inbox.post(_spec(addr, content="third task"))
        session = _FakeSession("coord", "proj/s")

        await inbox_prompt_dispatcher(session, inbox)  # type: ignore[arg-type]

        assert len(session.prompt.calls) == 1
        text = session.prompt.calls[0]
        assert "first task" in text
        assert "second task" in text
        assert "third task" in text
        assert "3 inbox item" in text

    async def test_inbox_unchanged_after_dispatch(self, tmp_path: Path) -> None:
        # Dispatcher is read-only with respect to notification state.
        # Mutation is the agent's responsibility via update_inbox_item.
        addr = _addr()
        inbox = SessionInbox(tmp_path / "inbox", addr)
        inbox.post(_spec(addr))
        before = inbox.prompt_pending()
        session = _FakeSession("coord", "proj/s")

        await inbox_prompt_dispatcher(session, inbox)  # type: ignore[arg-type]

        after = inbox.prompt_pending()
        assert [item.id for item in after] == [item.id for item in before]
        assert [item.status for item in after] == [
            NotificationStatus.PENDING,
        ]
