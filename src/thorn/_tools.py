"""Built-in tools for thorn agents.

Each tool is an ordinary Python function.  They are exposed to agents
via ``wrap_function()`` and can also be called directly from user code.

File-access tools (``read_file``, ``write_file``, ``list_directory``,
``search_files``) enforce the active ``FileAccessPolicy`` from the
current ``ExecutionContext``, when one is set.
"""

from __future__ import annotations

import asyncio
import fnmatch as fnmatch_mod
import os
import re
from pathlib import Path


def _enforce_access(path: str, required_name: str) -> None:
    """Check the active file-access policy, if any.

    Imports are deferred so that the module can still be loaded in
    contexts where no ``ExecutionContext`` is active (e.g. direct
    scripting use of ``read_file``).
    """
    from thorn._context import get_context
    from thorn._file_access import FileAccessLevel, check_access

    try:
        ctx = get_context()
    except RuntimeError:
        return

    policy = ctx.file_access_policy
    workspace = ctx.workspace_root
    if policy is None or workspace is None:
        return

    required = FileAccessLevel[required_name]
    check_access(path, required, policy=policy, workspace=workspace)


def _check_read_access(path: str) -> bool:
    """Return whether the active policy grants at least READ for *path*.

    Returns ``True`` when no context or policy is active, matching the
    permissive default of ``_enforce_access``.
    """
    from thorn._context import get_context
    from thorn._file_access import FileAccessLevel, resolve_for_check

    try:
        ctx = get_context()
    except RuntimeError:
        return True

    policy = ctx.file_access_policy
    workspace = ctx.workspace_root
    if policy is None or workspace is None:
        return True

    resolved = resolve_for_check(path, workspace)
    return policy.check(resolved) >= FileAccessLevel.READ


MAX_READ_LINES: int = 500
"""Hard ceiling on lines returned by a single ``read_file`` call."""

MAX_READ_CHARS: int = 50_000
"""Hard ceiling on characters returned by a single ``read_file`` call."""

OUTLINE_THRESHOLD: int = 200
"""When a file exceeds this many lines and no explicit range is
requested, return an outline view instead of a verbatim prefix.
Set below ``MAX_READ_LINES`` so that outline kicks in before hard
truncation; the agent can still request up to ``MAX_READ_LINES``
via ``offset``/``limit``."""

MAX_SEARCH_MATCHES: int = 100
"""Hard ceiling on matching lines returned by a single ``search_files`` call."""

MAX_SEARCH_CHARS: int = 50_000
"""Hard ceiling on characters returned by a single ``search_files`` call."""


def _format_lines(
    lines: list[str],
    first_line_number: int,
    *,
    width: int | None = None,
) -> str:
    """Number *lines* starting at *first_line_number*.

    Produces right-aligned line numbers with a ``|`` separator, e.g.::

          1| first line
         10| tenth line
    """
    last_number = first_line_number + len(lines) - 1
    if width is None:
        width = max(len(str(last_number)), 1)
    parts: list[str] = []
    for i, line in enumerate(lines):
        num = first_line_number + i
        parts.append(f"{num:>{width}}| {line}")
    return "\n".join(parts)


async def read_file(
    path: str,
    offset: int = 1,
    limit: int | None = None,
) -> str:
    """Read and return the contents of a file at the given path.

    Output is line-numbered.  Results are capped at a maximum number of
    lines and characters per invocation; use ``offset`` and ``limit`` to
    page through large files.

    Args:
        path: Filesystem path to read.
        offset: 1-based line number to start reading from (default: 1).
        limit: Maximum number of lines the caller wants.  The hard cap
            of MAX_READ_LINES still applies even if a larger value is
            given.
    """
    _enforce_access(path, "READ")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    all_lines = p.read_text(encoding="utf-8").splitlines()
    total_lines = len(all_lines)

    if offset == 1 and limit is None and total_lines > OUTLINE_THRESHOLD:
        from thorn._outline import outline_and_format

        return outline_and_format(
            all_lines,
            line_budget=OUTLINE_THRESHOLD,
            char_budget=MAX_READ_CHARS,
        )

    start_idx = max(offset - 1, 0)
    if start_idx >= total_lines:
        return f"[No content: offset {offset} is past end of file ({total_lines} lines).]"

    available = all_lines[start_idx:]
    requested_count = len(available)
    if limit is not None:
        requested_count = min(requested_count, limit)

    hard_cap = min(requested_count, MAX_READ_LINES)
    selected = available[:hard_cap]

    char_count = 0
    char_capped_len = len(selected)
    for i, line in enumerate(selected):
        char_count += len(line) + 1  # +1 for the newline
        if char_count > MAX_READ_CHARS:
            char_capped_len = max(i, 1)
            break

    selected = selected[:char_capped_len]
    shown_count = len(selected)
    first_line_num = start_idx + 1
    last_line_num = start_idx + shown_count

    body = _format_lines(selected, first_line_num)

    truncated = shown_count < len(available)
    if truncated:
        body += (
            f"\n[Truncated: showing lines {first_line_num}-{last_line_num}"
            f" of {total_lines} total."
            f" Use offset/limit to read more.]"
        )

    return body


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    _enforce_access(path, "WRITE")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


async def list_directory(path: str = ".") -> list[str]:
    """List entries in a directory.  Returns a list of names."""
    _enforce_access(path, "READ")
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    entries = sorted(entry.name for entry in p.iterdir())

    from thorn._context import get_context
    from thorn._file_access import FileAccessLevel, resolve_for_check

    try:
        ctx = get_context()
    except RuntimeError:
        return entries

    policy = ctx.file_access_policy
    workspace = ctx.workspace_root
    if policy is None or workspace is None:
        return entries

    return policy.filter_listing(entries, resolve_for_check(path, workspace))


# ---------------------------------------------------------------------------
# search_files helpers
# ---------------------------------------------------------------------------


def _collect_match_groups(
    all_lines: list[str],
    match_indices: list[int],
    context_lines: int,
) -> list[tuple[int, list[str]]]:
    """Merge matching line indices with surrounding context into groups.

    Adjacent or overlapping context windows are merged so that each
    group is a contiguous run of lines.  Returns ``(start_line, lines)``
    tuples where *start_line* is 1-based.
    """
    if not match_indices:
        return []

    total = len(all_lines)
    ranges: list[tuple[int, int]] = []
    for idx in match_indices:
        start = max(0, idx - context_lines)
        end = min(total - 1, idx + context_lines)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))

    return [
        (start + 1, all_lines[start : end + 1])
        for start, end in ranges
    ]


def _search_single_file(
    file_path: Path,
    compiled: re.Pattern[str],
    context_lines: int,
) -> tuple[int, list[tuple[int, list[str]]]]:
    """Search one file for *compiled*, returning ``(match_count, groups)``.

    Files that cannot be decoded as UTF-8 are silently skipped
    (returns zero matches).
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError, OSError):
        return 0, []

    all_lines = text.splitlines()
    match_indices = [
        i for i, line in enumerate(all_lines) if compiled.search(line)
    ]
    if not match_indices:
        return 0, []

    groups = _collect_match_groups(all_lines, match_indices, context_lines)
    return len(match_indices), groups


def _format_file_matches(
    file_path: str,
    groups: list[tuple[int, list[str]]],
) -> str:
    """Format one file's match groups in ripgrep style."""
    max_line = max(start + len(lines) - 1 for start, lines in groups)
    width = max(len(str(max_line)), 1)

    parts: list[str] = [f"{file_path}:"]
    for i, (start_line, lines) in enumerate(groups):
        if i > 0:
            parts.append("--")
        parts.append(_format_lines(lines, start_line, width=width))
    return "\n".join(parts)


async def search_files(
    pattern: str,
    path: str = ".",
    *,
    glob: str | None = None,
    use_regex: bool = False,
    context_lines: int = 0,
) -> str:
    """Search file contents for a pattern, returning matching lines.

    Searches a single file or recursively searches all files under a
    directory.  Results are formatted with file paths and line numbers.
    Output is capped; narrow the pattern or use ``glob`` to filter files
    if results are truncated.

    Args:
        pattern: Text to search for (literal by default, regex when
            *use_regex* is True).
        path: File or directory to search.  Defaults to the current
            directory.  Directory searches are recursive.
        glob: Optional filename filter using fnmatch syntax (e.g.
            ``"*.py"``).  Only files whose names match are searched.
        use_regex: Interpret *pattern* as a Python regular expression.
        context_lines: Number of lines to show before and after each
            match (like ``grep -C``).
    """
    try:
        compiled = re.compile(pattern if use_regex else re.escape(pattern))
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    p = Path(path)

    if p.is_file():
        _enforce_access(path, "READ")
        match_count, groups = _search_single_file(p, compiled, context_lines)
        if match_count == 0:
            return f'No matches found for "{pattern}" in {path}.'
        return _format_file_matches(path, groups)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    _enforce_access(path, "READ")

    target_files: list[Path] = sorted(
        f
        for f in p.rglob("*")
        if f.is_file()
        and (glob is None or fnmatch_mod.fnmatch(f.name, glob))
        and _check_read_access(str(f))
    )

    result_parts: list[str] = []
    shown_matches = 0
    total_matches = 0
    files_with_matches = 0
    char_count = 0
    truncated = False

    for file_path in target_files:
        file_match_count, groups = _search_single_file(
            file_path, compiled, context_lines,
        )
        if file_match_count == 0:
            continue

        total_matches += file_match_count
        files_with_matches += 1

        if truncated:
            continue

        formatted = _format_file_matches(str(file_path), groups)
        would_exceed_matches = (
            shown_matches + file_match_count > MAX_SEARCH_MATCHES
        )
        would_exceed_chars = char_count + len(formatted) > MAX_SEARCH_CHARS

        if result_parts and (would_exceed_matches or would_exceed_chars):
            truncated = True
            continue

        result_parts.append(formatted)
        shown_matches += file_match_count
        char_count += len(formatted)

    if not result_parts:
        return f'No matches found for "{pattern}" in {path}.'

    body = "\n\n".join(result_parts)

    if truncated:
        body += (
            f"\n\n[Truncated: showing {shown_matches} of {total_matches}"
            f" total matches across {files_with_matches} files."
            f" Narrow your pattern or use the glob parameter to"
            f" filter files.]"
        )

    return body


async def run_shell(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr.

    Not included in ``ALL_BUILTIN_TOOLS`` — add it explicitly to
    specific agents or ``prompt()`` calls when needed.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace") if stdout else ""
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


async def ask_user(question: str) -> str:
    """Ask the human user a question and return their response.

    In a non-interactive context this will raise an error.
    """
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, lambda: input(f"\n? {question}\n> "))
    return answer


ALL_BUILTIN_TOOLS = [read_file, write_file, list_directory, search_files, ask_user]

# Pre-packaged toolsets for common capabilities.  These are plain lists
# that can be nested inside a tools= parameter and will be flattened
# automatically by _prepare_tools / _collect_tools.

FILE_READING: list = [read_file, list_directory, search_files]
FILE_WRITING: list = [FILE_READING, write_file]
