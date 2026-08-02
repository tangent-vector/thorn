"""Tests for active-context salience tracking."""

from __future__ import annotations

from pathlib import Path

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext
from thorn.core._executor import ToolVenue
from thorn.core._func import wrap_function
from thorn.core._loop import run_agent_loop
from thorn.core._provider import FinishChunk, MockProvider, TextChunk, ToolCallChunk
from thorn.core._session import Session
from thorn.runtime._active_context import (
    MAX_ACTIVE_CONTEXT_ENTRIES,
    MAX_DETAILED_ACTIVE_CONTEXT_ENTRIES,
    ActiveContextToolEvent,
    update_active_context_from_tool_event,
)
from thorn.runtime._working_set import (
    ActiveContextDetailLevel,
    ActiveContextEvidence,
    ActiveContextKind,
    WorkingSet,
    render_working_set_block,
)


def _read_event(
    workspace_root: Path,
    path: str,
    result: str = "1| first\n2| second",
) -> ActiveContextToolEvent:
    return ActiveContextToolEvent(
        tool_name="read_file",
        arguments={"path": path},
        result=result,
        workspace_root=workspace_root,
    )


def _tool_call_response(call_id: str, name: str, arguments: str):
    return [
        ToolCallChunk(call_id=call_id, name=name, arguments=arguments),
        FinishChunk(reason="tool_calls"),
    ]


def _text_response(text: str):
    return [TextChunk(text=text), FinishChunk(reason="stop")]


def test_read_file_event_adds_span_context_with_hash(tmp_path: Path) -> None:
    path = tmp_path / "src" / "app.py"
    path.parent.mkdir()
    path.write_text("first\nsecond\n", encoding="utf-8")

    working_set = update_active_context_from_tool_event(
        WorkingSet(),
        _read_event(tmp_path, "src/app.py"),
    )

    assert len(working_set.active_context) == 1
    entry = working_set.active_context[0]
    assert entry.kind is ActiveContextKind.FILE
    assert entry.label == "src/app.py"
    assert entry.detail_level is ActiveContextDetailLevel.SPAN
    assert entry.summary == "read lines 1-2"
    assert entry.evidence == (ActiveContextEvidence.READ,)
    assert entry.salience == 2
    assert entry.content_hash is not None


def test_search_file_event_adds_directory_context(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()

    working_set = update_active_context_from_tool_event(
        WorkingSet(),
        ActiveContextToolEvent(
            tool_name="search_files",
            arguments={"path": "src", "pattern": "needle"},
            result="src/app.py:\n1| needle",
            workspace_root=tmp_path,
        ),
    )

    directory_entry = next(
        entry for entry in working_set.active_context
        if entry.kind is ActiveContextKind.DIRECTORY
    )
    assert directory_entry.label == "src"
    assert directory_entry.detail_level is ActiveContextDetailLevel.DIRECTORY
    assert directory_entry.summary == "searched for 'needle'"
    assert directory_entry.evidence == (ActiveContextEvidence.SEARCH,)

    hit_entry = next(
        entry for entry in working_set.active_context
        if entry.kind is ActiveContextKind.FILE
    )
    assert hit_entry.label == "src/app.py"
    assert hit_entry.summary == "search hit for 'needle'"


def test_run_shell_sed_event_adds_span_context(tmp_path: Path) -> None:
    path = tmp_path / "src" / "app.py"
    path.parent.mkdir()
    path.write_text(
        "\n".join(f"line {line_number}" for line_number in range(1, 21)),
        encoding="utf-8",
    )

    working_set = update_active_context_from_tool_event(
        WorkingSet(),
        ActiveContextToolEvent(
            tool_name="run_shell",
            arguments={"command": "sed -n '10,14p' src/app.py"},
            result="line 10\nline 11\nline 12\nline 13\nline 14",
            workspace_root=tmp_path,
        ),
    )

    assert len(working_set.active_context) == 1
    entry = working_set.active_context[0]
    assert entry.kind is ActiveContextKind.FILE
    assert entry.label == "src/app.py"
    assert entry.detail_level is ActiveContextDetailLevel.SPAN
    assert entry.summary == "shell read lines 10-14"
    assert entry.evidence == (ActiveContextEvidence.READ,)
    assert entry.salience == 2
    assert entry.content_hash is not None


def test_run_shell_rg_event_adds_directory_context(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()

    working_set = update_active_context_from_tool_event(
        WorkingSet(),
        ActiveContextToolEvent(
            tool_name="run_shell",
            arguments={"command": "rg -n needle src"},
            result="src/app.py:10:needle",
            workspace_root=tmp_path,
        ),
    )

    assert len(working_set.active_context) == 1
    entry = working_set.active_context[0]
    assert entry.kind is ActiveContextKind.DIRECTORY
    assert entry.label == "src"
    assert entry.detail_level is ActiveContextDetailLevel.DIRECTORY
    assert entry.summary == "shell searched for 'needle' via rg"
    assert entry.evidence == (ActiveContextEvidence.SEARCH,)


def test_edit_file_event_stales_prior_span_context(tmp_path: Path) -> None:
    path = tmp_path / "src" / "app.py"
    path.parent.mkdir()
    path.write_text("old\n", encoding="utf-8")
    working_set = update_active_context_from_tool_event(
        WorkingSet(),
        _read_event(tmp_path, "src/app.py", "1| old"),
    )

    path.write_text("new\n", encoding="utf-8")
    working_set = update_active_context_from_tool_event(
        working_set,
        ActiveContextToolEvent(
            tool_name="edit_file",
            arguments={"path": "src/app.py"},
            result="Applied 1 edit(s) to src/app.py.\n1| new",
            workspace_root=tmp_path,
        ),
    )

    stale_entry = next(
        entry for entry in working_set.active_context
        if entry.detail_level is ActiveContextDetailLevel.STALE_SUMMARY
    )
    assert stale_entry.label == "src/app.py"
    assert stale_entry.summary == "previously inspected before file changed"
    assert stale_entry.content_hash is None
    assert ActiveContextEvidence.EDIT in stale_entry.evidence
    assert all(
        entry.detail_level is not ActiveContextDetailLevel.SPAN
        for entry in working_set.active_context
    )


def test_repeated_access_renders_evidence_and_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "src" / "app.py"
    path.parent.mkdir()
    path.write_text("first\nsecond\n", encoding="utf-8")

    working_set = WorkingSet()
    for _ in range(3):
        working_set = update_active_context_from_tool_event(
            working_set,
            _read_event(tmp_path, "src/app.py"),
        )

    rendered = render_working_set_block(working_set)

    assert "file src/app.py (span): read lines 1-2 [read x3]" in rendered.text
    assert "active context repeatedly accessed: src/app.py" in rendered.text
    assert rendered.diagnostics == (
        "active context repeatedly accessed: "
        "src/app.py (read x3; salience 6)",
    )


def test_active_context_is_bounded_and_downgrades_tail_detail(
    tmp_path: Path,
) -> None:
    working_set = WorkingSet()
    for index in range(MAX_ACTIVE_CONTEXT_ENTRIES + 2):
        path = tmp_path / f"file{index}.py"
        path.write_text("line\n", encoding="utf-8")
        working_set = update_active_context_from_tool_event(
            working_set,
            _read_event(tmp_path, path.name, "1| line"),
        )

    assert len(working_set.active_context) == MAX_ACTIVE_CONTEXT_ENTRIES
    assert "file0.py" not in {entry.label for entry in working_set.active_context}
    assert "file1.py" not in {entry.label for entry in working_set.active_context}
    assert all(
        entry.detail_level is ActiveContextDetailLevel.SPAN
        for entry in working_set.active_context[:MAX_DETAILED_ACTIVE_CONTEXT_ENTRIES]
    )
    assert any(
        entry.detail_level is ActiveContextDetailLevel.FILE
        and "line detail elided by active-context budget" in (entry.summary or "")
        for entry in working_set.active_context[
            MAX_DETAILED_ACTIVE_CONTEXT_ENTRIES:
        ]
    )


async def test_agent_loop_records_successful_tool_events(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    async def read_file(path: str) -> str:
        """Read a file."""
        return "1| alpha\n2| beta"

    provider = MockProvider(
        canned_responses=[
            _tool_call_response("c1", "read_file", '{"path": "a.txt"}'),
            _text_response("done"),
        ],
    )
    session = Session(agent=Agent(name="active context test"))
    context = ExecutionContext(provider=provider, workspace_root=tmp_path)

    result = await run_agent_loop(
        context=context,
        user_prompt="read a.txt",
        tools=[wrap_function(read_file, venue=ToolVenue.IN_PROCESS)],
        session=session,
    )

    assert result == "done"
    assert session.working_set.active_context[0].label == "a.txt"
    assert session.working_set.active_context[0].summary == "read lines 1-2"
