"""Agent journal: chronologically-organized, append-only durable notes.

Each agent has a ``journal/`` directory under its home.  Files are named
by date (``YYYY-MM-DD.md``) and contain timestamped Markdown entries
attributed to specific sessions.

This module provides:

- Helper functions for reading/writing journal files (used by both the
  harness for injection and the agent tools for direct access).
- ``write_journal`` and ``read_journal`` tool functions (available on
  all agents by default via ``Agent._collect_tools``).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from thorn.core._func import tool
from thorn.core._history import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATE_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""Matches ``YYYY-MM-DD`` (the stem of a journal day file)."""

_SECTION_HEADING_RE = re.compile(r"^##\s")
"""Matches the start of a level-2 Markdown heading (journal entry boundary)."""

DEFAULT_INJECTION_DAYS: int = 2
"""Default number of days of journal entries to inject into context."""

DEFAULT_INJECTION_TOKEN_BUDGET: int = 2000
"""Default token budget for journal injection."""


# ---------------------------------------------------------------------------
# Helper functions (harness-level, all take journal_directory: Path)
# ---------------------------------------------------------------------------


def append_journal_entry(
    journal_directory: Path,
    content: str,
    *,
    session_key: str | None = None,
) -> Path:
    """Append a timestamped entry to today's journal file.

    Creates the journal directory and today's file if they don't exist.
    Returns the path to the journal file that was written to.
    """
    journal_directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    file_path = journal_directory / f"{date_str}.md"

    attribution = f" -- {session_key}" if session_key else ""
    header = f"## {time_str} UTC{attribution}"

    needs_separator = file_path.exists() and file_path.stat().st_size > 0
    prefix = "\n" if needs_separator else ""
    entry = f"{prefix}{header}\n\n{content}\n"

    with file_path.open("a", encoding="utf-8") as f:
        f.write(entry)

    return file_path


def read_journal_day(
    journal_directory: Path,
    date: str,
) -> str | None:
    """Read a specific day's journal file.

    *date* must be in ``YYYY-MM-DD`` format.  Returns the file contents
    as a string, or ``None`` if the file doesn't exist or the date
    format is invalid.
    """
    if not _DATE_FILENAME_RE.match(date):
        return None
    file_path = journal_directory / f"{date}.md"
    if not file_path.is_file():
        return None
    try:
        return file_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("failed to read journal file %s", file_path, exc_info=True)
        return None


def list_journal_dates(journal_directory: Path) -> list[str]:
    """Return a sorted list of dates (``YYYY-MM-DD``) that have journal files.

    Dates are sorted chronologically (oldest first).
    """
    if not journal_directory.is_dir():
        return []
    dates: list[str] = []
    for f in journal_directory.iterdir():
        if f.suffix == ".md" and _DATE_FILENAME_RE.match(f.stem):
            dates.append(f.stem)
    dates.sort()
    return dates


def read_recent_journal(
    journal_directory: Path,
    *,
    days: int = DEFAULT_INJECTION_DAYS,
    token_budget: int = DEFAULT_INJECTION_TOKEN_BUDGET,
    exclude_session_key: str | None = None,
) -> str:
    """Read recent journal entries, dual-limited by days and tokens.

    Implements the injection algorithm: starts from the most recent day
    and works backward.  Whole files are included when they fit; when a
    file exceeds the remaining token budget, the tail-biased partial
    inclusion strategy keeps the most recent entries fully expanded
    while retaining structural headings from earlier entries.

    Entries attributed to *exclude_session_key* (if provided) are
    filtered out to avoid duplicating information already present in
    the current session's history.
    """
    all_dates = list_journal_dates(journal_directory)
    if not all_dates:
        return ""

    accumulator_parts: list[str] = []
    remaining_tokens = token_budget
    remaining_days = days

    for date in reversed(all_dates):
        if remaining_days <= 0 or remaining_tokens <= 0:
            break

        content = read_journal_day(journal_directory, date)
        if not content or not content.strip():
            continue

        if exclude_session_key:
            content = _filter_session_entries(content, exclude_session_key)
            if not content.strip():
                continue

        content_tokens = estimate_tokens(content)

        if content_tokens <= remaining_tokens:
            day_block = f"### {date}\n\n{content.strip()}"
            accumulator_parts.insert(0, day_block)
            remaining_tokens -= content_tokens
        else:
            partial = _partial_journal_content(content, remaining_tokens)
            if partial.strip():
                day_block = f"### {date}\n\n{partial.strip()}"
                accumulator_parts.insert(0, day_block)
            remaining_tokens = 0

        remaining_days -= 1

    return "\n\n---\n\n".join(accumulator_parts)


# ---------------------------------------------------------------------------
# Internal: section-level operations on journal file content
# ---------------------------------------------------------------------------


def _split_into_sections(content: str) -> list[str]:
    """Split journal content into sections at ``##`` heading boundaries.

    Each returned string is a complete section: the heading line plus
    all body lines until the next heading (or end of file).  Content
    before the first heading (if any) is returned as its own section.
    """
    lines = content.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _SECTION_HEADING_RE.match(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections


def _extract_section_heading(section: str) -> str | None:
    """Return the first ``##`` heading line from *section*, or ``None``."""
    for line in section.split("\n"):
        if _SECTION_HEADING_RE.match(line):
            return line
    return None


def _filter_session_entries(
    content: str,
    exclude_session_key: str,
) -> str:
    """Remove journal sections attributed to *exclude_session_key*."""
    sections = _split_into_sections(content)
    kept = [
        s for s in sections
        if not _section_is_attributed_to(s, exclude_session_key)
    ]
    return "\n".join(kept)


def _section_is_attributed_to(section: str, session_key: str) -> bool:
    """Check whether *section*'s heading attributes it to *session_key*.

    Matches ``-- {session_key}`` at the end of the heading line so that
    a key like ``session-A`` does not falsely match ``session-AB``.
    """
    heading = _extract_section_heading(section)
    if heading is None:
        return False
    return heading.rstrip().endswith(f"-- {session_key}")


def _partial_journal_content(content: str, token_budget: int) -> str:
    """Include journal content with a strong bias toward the tail.

    Sections from the end of the file are included fully first.  Once
    the budget is exhausted, earlier sections are represented by their
    heading line only (preserving structural context about what happened
    earlier without consuming full token cost).
    """
    sections = _split_into_sections(content)
    if not sections:
        return ""

    remaining = token_budget
    fully_included_from = len(sections)

    for i in range(len(sections) - 1, -1, -1):
        section_tokens = estimate_tokens(sections[i])
        if section_tokens <= remaining:
            remaining -= section_tokens
            fully_included_from = i
        else:
            break

    result_parts: list[str] = []

    for i in range(fully_included_from):
        heading = _extract_section_heading(sections[i])
        if heading:
            stub = heading + "\n[...content omitted...]"
            stub_tokens = estimate_tokens(stub)
            if stub_tokens <= remaining:
                result_parts.append(stub)
                remaining -= stub_tokens

    for i in range(fully_included_from, len(sections)):
        result_parts.append(sections[i])

    return "\n\n".join(result_parts)


# ---------------------------------------------------------------------------
# Scope-chain helpers for tool functions
# ---------------------------------------------------------------------------


def _resolve_session_key() -> str | None:
    """Walk the scope chain to find the nearest ``session_key`` metadata."""
    from thorn.core._context import get_context

    try:
        ctx = get_context()
    except RuntimeError:
        return None

    scope = ctx.scope
    while scope is not None:
        sk = scope.metadata.get("session_key")
        if sk is not None:
            return str(sk)
        scope = scope.outer
    return None


def _resolve_journal_directory() -> Path | None:
    """Derive the journal directory from the current agent's home."""
    from thorn.core._context import get_context

    try:
        ctx = get_context()
    except RuntimeError:
        return None

    agent = ctx.agent
    if agent is None:
        return None
    home = agent.home
    if home is None:
        return None
    return home / "journal"


# ---------------------------------------------------------------------------
# Tool functions (agent-facing, use get_context() for ambient state)
# ---------------------------------------------------------------------------


@tool
async def write_journal(content: str) -> str:
    """Append a timestamped entry to your personal journal.

    Your journal is a chronological log of activity and observations,
    stored across sessions.  Use it to record important context,
    decisions, and progress that should persist beyond the current
    conversation.

    The entry is automatically timestamped and attributed to the
    current session.

    Args:
        content: The text to write to the journal entry.
    """
    journal_dir = _resolve_journal_directory()
    if journal_dir is None:
        return "Error: no agent home directory available. Cannot write journal."

    session_key = _resolve_session_key()
    file_path = append_journal_entry(journal_dir, content, session_key=session_key)

    return f"Journal entry appended to {file_path.stem}."


@tool
async def read_journal(date: str | None = None, days: int = 1) -> str:
    """Read entries from your personal journal.

    By default returns today's entries.  Provide a specific date to
    read that day's entries, or increase ``days`` to see more history.

    Args:
        date: Specific date in YYYY-MM-DD format.  When provided,
            reads only that day's entries (``days`` is ignored).
        days: Number of most recent days to include.  Defaults to 1.
    """
    journal_dir = _resolve_journal_directory()
    if journal_dir is None:
        return "Error: no agent home directory available. Cannot read journal."

    if date is not None:
        day_content = read_journal_day(journal_dir, date)
        if day_content is None:
            return f"No journal entries found for {date}."
        return day_content

    all_dates = list_journal_dates(journal_dir)
    if not all_dates:
        return "No journal entries found."

    recent_dates = all_dates[-days:]
    parts: list[str] = []
    for d in recent_dates:
        day_content = read_journal_day(journal_dir, d)
        if day_content and day_content.strip():
            parts.append(f"### {d}\n\n{day_content.strip()}")

    if not parts:
        return "No journal entries found."

    return "\n\n---\n\n".join(parts)


JOURNAL_TOOLS: list = [write_journal, read_journal]
"""Default journal tools added to every agent via ``Agent._collect_tools``."""
