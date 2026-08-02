"""Tests for thorn.core._journal — journal helper functions, tools, and injection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from thorn.core._agent import Agent
from thorn.core._context import (
    ExecutionContext,
    Scope,
    reset_context,
    set_context,
)
from thorn.core._journal import (
    JOURNAL_TOOLS,
    _extract_section_heading,
    _filter_session_entries,
    _partial_journal_content,
    _resolve_journal_directory,
    _resolve_session_key,
    _split_into_sections,
    append_journal_entry,
    list_journal_dates,
    read_journal,
    read_journal_day,
    read_recent_journal,
    write_journal,
)
from thorn.core._provider import MockProvider
from thorn.runtime._session import AgentID

# ---------------------------------------------------------------------------
# append_journal_entry
# ---------------------------------------------------------------------------


class TestAppendJournalEntry:
    def test_creates_directory_and_file(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        assert not journal_dir.exists()

        result = append_journal_entry(journal_dir, "test content")

        assert journal_dir.is_dir()
        assert result.exists()
        assert result.suffix == ".md"

        content = result.read_text(encoding="utf-8")
        assert "test content" in content

    def test_entry_has_timestamp_heading(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        append_journal_entry(journal_dir, "some note")

        files = list(journal_dir.glob("*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "## " in content
        assert "UTC" in content

    def test_entry_includes_session_key(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        append_journal_entry(
            journal_dir, "with session", session_key="my-session-42",
        )

        files = list(journal_dir.glob("*.md"))
        content = files[0].read_text(encoding="utf-8")
        assert "-- my-session-42" in content

    def test_entry_omits_attribution_when_no_session_key(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        append_journal_entry(journal_dir, "no session")

        files = list(journal_dir.glob("*.md"))
        content = files[0].read_text(encoding="utf-8")
        assert "--" not in content

    def test_multiple_entries_append(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        append_journal_entry(journal_dir, "first entry")
        append_journal_entry(journal_dir, "second entry")

        files = list(journal_dir.glob("*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "first entry" in content
        assert "second entry" in content

    def test_returns_correct_file_path(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        result = append_journal_entry(journal_dir, "test")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert result.name == f"{today}.md"
        assert result.parent == journal_dir


# ---------------------------------------------------------------------------
# read_journal_day
# ---------------------------------------------------------------------------


class TestReadJournalDay:
    def test_reads_existing_file(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        (journal_dir / "2026-04-07.md").write_text("day content", encoding="utf-8")

        result = read_journal_day(journal_dir, "2026-04-07")
        assert result == "day content"

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()

        result = read_journal_day(journal_dir, "2026-01-01")
        assert result is None

    def test_returns_none_for_invalid_date_format(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        (journal_dir / "not-a-date.md").write_text("content", encoding="utf-8")

        assert read_journal_day(journal_dir, "not-a-date") is None

    def test_returns_none_for_missing_directory(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        assert read_journal_day(journal_dir, "2026-04-07") is None


# ---------------------------------------------------------------------------
# list_journal_dates
# ---------------------------------------------------------------------------


class TestListJournalDates:
    def test_lists_dates_sorted(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        (journal_dir / "2026-04-08.md").write_text("", encoding="utf-8")
        (journal_dir / "2026-04-06.md").write_text("", encoding="utf-8")
        (journal_dir / "2026-04-07.md").write_text("", encoding="utf-8")

        result = list_journal_dates(journal_dir)
        assert result == ["2026-04-06", "2026-04-07", "2026-04-08"]

    def test_ignores_non_date_files(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        (journal_dir / "2026-04-07.md").write_text("", encoding="utf-8")
        (journal_dir / "notes.md").write_text("", encoding="utf-8")
        (journal_dir / "2026-04-07.txt").write_text("", encoding="utf-8")

        result = list_journal_dates(journal_dir)
        assert result == ["2026-04-07"]

    def test_returns_empty_for_missing_directory(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        assert list_journal_dates(journal_dir) == []

    def test_returns_empty_for_empty_directory(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        assert list_journal_dates(journal_dir) == []


# ---------------------------------------------------------------------------
# _split_into_sections
# ---------------------------------------------------------------------------


class TestSplitIntoSections:
    def test_single_section(self):
        content = "## 10:00 UTC\n\nsome content\n"
        sections = _split_into_sections(content)
        assert len(sections) == 1
        assert "10:00 UTC" in sections[0]
        assert "some content" in sections[0]

    def test_multiple_sections(self):
        content = (
            "## 10:00 UTC\n\nfirst\n\n"
            "## 12:00 UTC\n\nsecond\n"
        )
        sections = _split_into_sections(content)
        assert len(sections) == 2
        assert "first" in sections[0]
        assert "second" in sections[1]

    def test_content_before_first_heading(self):
        content = "preamble\n\n## 10:00 UTC\n\nsection content\n"
        sections = _split_into_sections(content)
        assert len(sections) == 2
        assert "preamble" in sections[0]
        assert "section content" in sections[1]

    def test_empty_content(self):
        sections = _split_into_sections("")
        assert len(sections) == 1
        assert sections[0] == ""


# ---------------------------------------------------------------------------
# _extract_section_heading
# ---------------------------------------------------------------------------


class TestExtractSectionHeading:
    def test_extracts_heading(self):
        section = "## 10:00 UTC -- session-1\n\nBody text.\n"
        assert _extract_section_heading(section) == "## 10:00 UTC -- session-1"

    def test_returns_none_for_no_heading(self):
        assert _extract_section_heading("just plain text\n") is None


# ---------------------------------------------------------------------------
# _filter_session_entries
# ---------------------------------------------------------------------------


class TestFilterSessionEntries:
    def test_removes_matching_session(self):
        content = (
            "## 10:00 UTC -- session-A\n\nfirst\n\n"
            "## 12:00 UTC -- session-B\n\nsecond\n"
        )
        result = _filter_session_entries(content, "session-A")
        assert "first" not in result
        assert "second" in result
        assert "session-B" in result

    def test_keeps_all_when_no_match(self):
        content = "## 10:00 UTC -- session-A\n\ncontent\n"
        result = _filter_session_entries(content, "session-Z")
        assert "content" in result

    def test_does_not_falsely_match_substring(self):
        content = (
            "## 10:00 UTC -- session-AB\n\nshould keep\n\n"
            "## 12:00 UTC -- session-A\n\nshould remove\n"
        )
        result = _filter_session_entries(content, "session-A")
        assert "should keep" in result
        assert "should remove" not in result


# ---------------------------------------------------------------------------
# _partial_journal_content
# ---------------------------------------------------------------------------


class TestPartialJournalContent:
    def test_returns_empty_for_empty_content(self):
        assert _partial_journal_content("", 1000) == ""

    def test_includes_everything_when_budget_is_large(self):
        content = "## 10:00 UTC\n\nshort entry\n"
        result = _partial_journal_content(content, 100_000)
        assert "short entry" in result

    def test_tail_bias_keeps_last_section(self):
        sections = []
        for i in range(10):
            sections.append(f"## {i:02d}:00 UTC\n\n{'x' * 200}\n")
        content = "\n".join(sections)

        result = _partial_journal_content(content, 100)

        assert "09:00 UTC" in result
        last_section_content = "x" * 200
        assert last_section_content in result

    def test_earlier_sections_show_heading_only(self):
        content = (
            "## 10:00 UTC\n\n" + "a" * 400 + "\n\n"
            "## 20:00 UTC\n\n" + "b" * 100 + "\n"
        )
        result = _partial_journal_content(content, 80)

        assert "20:00 UTC" in result
        assert "b" * 100 in result
        assert "10:00 UTC" in result
        assert "[...content omitted...]" in result
        assert "a" * 400 not in result


# ---------------------------------------------------------------------------
# read_recent_journal
# ---------------------------------------------------------------------------


class TestReadRecentJournal:
    def _make_journal(self, journal_dir: Path, entries: dict[str, str]) -> None:
        """Write journal files.  Keys are YYYY-MM-DD, values are content."""
        journal_dir.mkdir(parents=True, exist_ok=True)
        for date, content in entries.items():
            (journal_dir / f"{date}.md").write_text(content, encoding="utf-8")

    def test_returns_empty_for_missing_directory(self, tmp_path: Path):
        assert read_recent_journal(tmp_path / "journal") == ""

    def test_single_day(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        self._make_journal(journal_dir, {
            "2026-04-07": "## 10:00 UTC\n\nhello\n",
        })

        result = read_recent_journal(journal_dir, days=1, token_budget=10_000)
        assert "hello" in result
        assert "2026-04-07" in result

    def test_day_budget_limits_days(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        self._make_journal(journal_dir, {
            "2026-04-05": "## 10:00 UTC\n\nday5\n",
            "2026-04-06": "## 10:00 UTC\n\nday6\n",
            "2026-04-07": "## 10:00 UTC\n\nday7\n",
        })

        result = read_recent_journal(journal_dir, days=2, token_budget=10_000)
        assert "day7" in result
        assert "day6" in result
        assert "day5" not in result

    def test_token_budget_limits_content(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        self._make_journal(journal_dir, {
            "2026-04-06": "## 10:00 UTC\n\n" + "x" * 8000 + "\n",
            "2026-04-07": "## 10:00 UTC\n\nshort\n",
        })

        result = read_recent_journal(journal_dir, days=5, token_budget=50)
        assert "short" in result
        assert "x" * 8000 not in result

    def test_excludes_session_entries(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        self._make_journal(journal_dir, {
            "2026-04-07": (
                "## 10:00 UTC -- session-A\n\nfrom A\n\n"
                "## 12:00 UTC -- session-B\n\nfrom B\n"
            ),
        })

        result = read_recent_journal(
            journal_dir,
            days=1,
            token_budget=10_000,
            exclude_session_key="session-A",
        )
        assert "from A" not in result
        assert "from B" in result

    def test_partial_inclusion_for_large_file(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        entries = []
        for i in range(20):
            entries.append(f"## {i:02d}:00 UTC\n\n{'y' * 200}\n")
        self._make_journal(journal_dir, {
            "2026-04-07": "\n".join(entries),
        })

        result = read_recent_journal(journal_dir, days=1, token_budget=200)
        assert "19:00 UTC" in result
        assert "[...content omitted...]" in result

    def test_most_recent_day_prioritized(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        self._make_journal(journal_dir, {
            "2026-04-06": "## 10:00 UTC\n\nold stuff\n",
            "2026-04-07": "## 10:00 UTC\n\nnew stuff\n",
        })

        result = read_recent_journal(journal_dir, days=5, token_budget=10_000)
        idx_old = result.index("old stuff")
        idx_new = result.index("new stuff")
        assert idx_old < idx_new

    def test_skips_empty_files(self, tmp_path: Path):
        journal_dir = tmp_path / "journal"
        self._make_journal(journal_dir, {
            "2026-04-06": "## 10:00 UTC\n\nhas content\n",
            "2026-04-07": "",
        })

        result = read_recent_journal(journal_dir, days=2, token_budget=10_000)
        assert "has content" in result
        assert "2026-04-07" not in result


# ---------------------------------------------------------------------------
# _resolve_session_key
# ---------------------------------------------------------------------------


class TestResolveSessionKey:
    def test_returns_none_without_context(self):
        assert _resolve_session_key() is None

    def test_finds_session_key_in_scope(self, tmp_path: Path):
        ctx = ExecutionContext(
            provider=MockProvider(),
            scope=Scope(description="test", metadata={"session_key": "sk-1"}),
        )
        token = set_context(ctx)
        try:
            assert _resolve_session_key() == "sk-1"
        finally:
            reset_context(token)

    def test_walks_parent_scopes(self, tmp_path: Path):
        outer = Scope(description="outer", metadata={"session_key": "sk-outer"})
        inner = Scope(description="inner", outer=outer, metadata={})
        ctx = ExecutionContext(provider=MockProvider(), scope=inner)
        token = set_context(ctx)
        try:
            assert _resolve_session_key() == "sk-outer"
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# _resolve_journal_directory
# ---------------------------------------------------------------------------


class TestResolveJournalDirectory:
    def test_returns_none_without_context(self):
        assert _resolve_journal_directory() is None

    def test_returns_journal_under_home(self, tmp_path: Path):
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agency_root_directory=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = Agent(id=AgentID("test-agent"))
            ctx_inner = ExecutionContext(
                provider=MockProvider(),
                agent=agent,
                workspace_root=tmp_path,
                agency_root_directory=tmp_path,
            )
            token_inner = set_context(ctx_inner)
            try:
                result = _resolve_journal_directory()
                assert result == agent.home / "journal"
            finally:
                reset_context(token_inner)
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# write_journal tool
# ---------------------------------------------------------------------------


class TestWriteJournalTool:
    @pytest.mark.asyncio
    async def test_writes_entry(self, tmp_path: Path):
        agent = Agent(id=AgentID("writer"), home=tmp_path / "home")
        ctx = ExecutionContext(
            provider=MockProvider(),
            agent=agent,
            workspace_root=tmp_path,
            scope=Scope(
                description="test",
                metadata={"session_key": "sess-42"},
            ),
        )
        token = set_context(ctx)
        try:
            result = await write_journal("my important note")
            assert "Journal entry appended" in result

            journal_dir = agent.home / "journal"
            assert journal_dir.is_dir()
            files = list(journal_dir.glob("*.md"))
            assert len(files) == 1
            content = files[0].read_text(encoding="utf-8")
            assert "my important note" in content
            assert "sess-42" in content
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_error_when_no_home(self):
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            result = await write_journal("test")
            assert "Error" in result
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# read_journal tool
# ---------------------------------------------------------------------------


class TestReadJournalTool:
    @pytest.mark.asyncio
    async def test_reads_specific_date(self, tmp_path: Path):
        agent = Agent(id=AgentID("reader"), home=tmp_path / "home")
        journal_dir = agent.home / "journal"
        journal_dir.mkdir(parents=True)
        (journal_dir / "2026-04-07.md").write_text(
            "## 10:00 UTC\n\nday content\n", encoding="utf-8",
        )

        ctx = ExecutionContext(
            provider=MockProvider(),
            agent=agent,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = await read_journal(date="2026-04-07")
            assert "day content" in result
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_reads_recent_days(self, tmp_path: Path):
        agent = Agent(id=AgentID("reader"), home=tmp_path / "home")
        journal_dir = agent.home / "journal"
        journal_dir.mkdir(parents=True)
        (journal_dir / "2026-04-06.md").write_text(
            "## 10:00 UTC\n\nday6\n", encoding="utf-8",
        )
        (journal_dir / "2026-04-07.md").write_text(
            "## 10:00 UTC\n\nday7\n", encoding="utf-8",
        )

        ctx = ExecutionContext(
            provider=MockProvider(),
            agent=agent,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = await read_journal(days=2)
            assert "day6" in result
            assert "day7" in result
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_no_entries(self, tmp_path: Path):
        agent = Agent(id=AgentID("reader"), home=tmp_path / "home")

        ctx = ExecutionContext(
            provider=MockProvider(),
            agent=agent,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = await read_journal()
            assert "No journal entries" in result
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_error_when_no_home(self):
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            result = await read_journal()
            assert "Error" in result
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# JOURNAL_TOOLS and Agent._collect_tools integration
# ---------------------------------------------------------------------------


class TestJournalToolsCollected:
    def test_journal_tools_constant(self):
        assert len(JOURNAL_TOOLS) == 2
        names = {getattr(t, "__name__", None) for t in JOURNAL_TOOLS}
        assert names == {"write_journal", "read_journal"}

    def test_base_agent_includes_journal_tools(self):
        tool_names = {
            getattr(t, "__name__", None)
            for t in Agent._collect_tools()
        }
        assert "write_journal" in tool_names
        assert "read_journal" in tool_names

    def test_subclass_with_own_tools_still_gets_journal(self):
        def custom_tool() -> str:
            """Custom."""

        class CustomAgent(Agent):
            tools = [custom_tool]

        tool_names = {
            getattr(t, "__name__", None)
            for t in CustomAgent._collect_tools()
        }
        assert "custom_tool" in tool_names
        assert "write_journal" in tool_names
        assert "read_journal" in tool_names

    def test_subclass_can_shadow_journal_tool(self):
        async def write_journal() -> str:
            """Override."""

        class ShadowAgent(Agent):
            tools = [write_journal]

        tools = ShadowAgent._collect_tools()
        tool_names = [getattr(t, "__name__", None) for t in tools]
        assert tool_names.count("write_journal") == 1
        wj = next(t for t in tools if getattr(t, "__name__", None) == "write_journal")
        assert wj is write_journal


# ---------------------------------------------------------------------------
# session_key in push_scope (integration with _run_session_prompt)
# ---------------------------------------------------------------------------


class TestSessionKeyInScope:
    def test_push_scope_stores_session_key(self, tmp_path: Path):
        from thorn.runtime._session import SessionKey

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agency_root_directory=tmp_path,
        )
        session_key = SessionKey("my-session-key")
        child = ctx.push_scope(
            "test",
            session_key=str(session_key),
        )
        assert child.scope is not None
        assert child.scope.metadata.get("session_key") == "my-session-key"
