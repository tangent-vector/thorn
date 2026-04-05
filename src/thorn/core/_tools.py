"""Built-in tools for thorn agents.

Each tool is an ordinary Python function.  They are exposed to agents
via ``wrap_function()`` and can also be called directly from user code.

File-access tools (``read_file``, ``edit_file``, ``create_file``,
``delete_file``, ``move_file``, ``list_directory``, ``find_files``,
``search_files``) enforce the active
``FileAccessPolicy`` from the current ``ExecutionContext``, when one
is set.
"""

from __future__ import annotations

import asyncio
import fnmatch as fnmatch_mod
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass


def _enforce_access(path: str, required_name: str) -> None:
    """Check the active file-access policy, if any.

    Imports are deferred so that the module can still be loaded in
    contexts where no ``ExecutionContext`` is active (e.g. direct
    scripting use of ``read_file``).
    """
    from thorn.core._context import get_context
    from thorn.core._file_access import FileAccessLevel, check_access

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
    from thorn.core._context import get_context
    from thorn.core._file_access import FileAccessLevel, resolve_for_check

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

MAX_FIND_RESULTS: int = 200
"""Hard ceiling on entries returned by a single ``find_files`` call."""

MAX_LIST_ENTRIES: int = 200
"""Hard ceiling on entries shown by ``list_directory``."""


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
        from thorn.core._outline import outline_and_format

        return outline_and_format(
            all_lines,
            line_budget=OUTLINE_THRESHOLD,
            char_budget=MAX_READ_CHARS,
            file_path=path,
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


EDIT_CONTEXT_LINES: int = 4
"""Lines of context shown around each edit region in ``edit_file`` results."""


@dataclass
class FileEdit:
    """A single find-and-replace edit within a file."""

    old_string: str = Field(
        description=(
            "Text to find. Must occur exactly once in the file at the "
            "time this edit is applied. Include enough surrounding "
            "context to ensure a unique match."
        ),
    )
    new_string: str = Field(
        description=(
            "Replacement text. Must include the same surrounding "
            "context as old_string, plus the desired change. Use an "
            "empty string to delete the matched text."
        ),
    )


async def edit_file(path: str, edits: list[FileEdit]) -> str:
    """Apply one or more find-and-replace edits to an existing file.

    Each edit replaces exactly one occurrence of ``old_string`` with
    ``new_string``.  Edits are applied sequentially; later edits match
    against the content resulting from earlier ones.

    Returns a contextual view of the file around each edited region so
    the caller can verify the changes without a separate ``read_file``.

    Args:
        path: Filesystem path to the file to edit.
        edits: Edits to apply in order.  Each edit's ``old_string``
            must match exactly once in the file at the time it is
            applied.
    """
    _enforce_access(path, "WRITE")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if not edits:
        content = p.read_text(encoding="utf-8")
        total = len(content.splitlines())
        return f"No edits to apply. {path} is unchanged ({total} lines)."

    content = p.read_text(encoding="utf-8")

    edit_regions: list[tuple[int, int]] = []

    for i, edit in enumerate(edits):
        label = f"Edit {i + 1}/{len(edits)}"

        if not edit.old_string:
            raise ValueError(
                f"{label}: old_string must not be empty. "
                f"Use create_file to write to a new or empty file."
            )

        count = content.count(edit.old_string)
        if count == 0:
            raise ValueError(
                f"{label}: old_string not found in {path}."
            )
        if count > 1:
            raise ValueError(
                f"{label}: old_string has {count} matches in {path} "
                f"(must be unique). Add more surrounding context to "
                f"disambiguate."
            )

        pos = content.index(edit.old_string)
        start_line = content[:pos].count("\n") + 1
        old_newlines = edit.old_string.count("\n")

        content = (
            content[:pos]
            + edit.new_string
            + content[pos + len(edit.old_string) :]
        )

        if edit.new_string:
            new_newlines = edit.new_string.count("\n")
            end_line = start_line + new_newlines
        else:
            end_line = start_line

        line_delta = (
            edit.new_string.count("\n") - old_newlines
        )
        if line_delta != 0:
            edit_end_in_old = start_line + old_newlines
            for j in range(len(edit_regions)):
                r_start, r_end = edit_regions[j]
                if r_start > edit_end_in_old:
                    edit_regions[j] = (
                        r_start + line_delta,
                        r_end + line_delta,
                    )

        edit_regions.append((start_line, end_line))

    p.write_text(content, encoding="utf-8")

    all_lines = content.splitlines()
    header = f"Applied {len(edits)} edit(s) to {path}."
    body = _format_file_result(all_lines, edit_regions)
    return f"{header}\n{body}"


async def create_file(path: str, content: str) -> str:
    """Create a new file with the given content.

    Parent directories are created automatically.  Raises ``FileExistsError``
    if the file already exists — use ``edit_file`` to modify existing files.

    Returns a view of the new file's content so the caller can verify
    what was written without a separate ``read_file``.

    Args:
        path: Filesystem path for the new file.
        content: Full text content to write.
    """
    _enforce_access(path, "WRITE")
    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"File already exists: {path}. "
            f"Use edit_file to modify existing files."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

    all_lines = content.splitlines()
    total = len(all_lines)
    header = f"Created {path} ({total} lines)."

    if total == 0:
        return header

    if total <= OUTLINE_THRESHOLD:
        body = _format_lines(all_lines, 1)
        return f"{header}\n{body}"

    from thorn.core._outline import outline_and_format

    body = outline_and_format(
        all_lines,
        line_budget=OUTLINE_THRESHOLD,
        char_budget=MAX_READ_CHARS,
        file_path=path,
    )
    return f"{header}\n{body}"


def _format_file_result(
    lines: list[str],
    regions: list[tuple[int, int]],
) -> str:
    """Format a view of file content highlighting specific regions.

    Shows context lines around each region with the rest collapsed,
    using the same ``OutputSpan`` / ``format_outline`` pipeline as
    ``read_file``.
    """
    total = len(lines)
    if total == 0:
        return "[empty file]"

    from thorn.core._outline import format_outline, spans_for_regions

    spans = spans_for_regions(
        total, regions, context_lines=EDIT_CONTEXT_LINES,
    )
    return format_outline(lines, spans, char_budget=MAX_READ_CHARS)


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    .. deprecated::
        Use ``edit_file`` for modifying existing files and
        ``create_file`` for new files.  ``write_file`` remains
        available for backward compatibility but is no longer
        included in the default tool sets.
    """
    _enforce_access(path, "WRITE")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


async def delete_file(path: str) -> str:
    """Delete a file at the given path.

    Only regular files can be deleted — directory deletion is not
    supported.  Raises ``FileNotFoundError`` if the file does not exist.

    Args:
        path: Filesystem path of the file to delete.
    """
    _enforce_access(path, "WRITE")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.is_dir():
        raise IsADirectoryError(
            f"Cannot delete directory: {path}. "
            f"Only individual files can be deleted."
        )
    p.unlink()
    return f"Deleted {path}."


async def move_file(source: str, destination: str) -> str:
    """Move or rename a file from *source* to *destination*.

    Parent directories for *destination* are created automatically.
    Raises ``FileExistsError`` if the destination already exists.

    Args:
        source: Current path of the file to move.
        destination: New path for the file.
    """
    _enforce_access(source, "READ")
    _enforce_access(destination, "WRITE")
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if dst.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. "
            f"Delete it first or choose a different name."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved {source} → {destination}."


def _apply_listing_filter(entries: list[str], path: str) -> list[str]:
    """Filter *entries* through the active file-access policy, if any.

    Removes HIDDEN entries so they don't appear in listings.
    """
    from thorn.core._context import get_context
    from thorn.core._file_access import resolve_for_check

    try:
        ctx = get_context()
    except RuntimeError:
        return entries

    policy = ctx.file_access_policy
    workspace = ctx.workspace_root
    if policy is None or workspace is None:
        return entries

    return policy.filter_listing(entries, resolve_for_check(path, workspace))


def _list_recursive(
    root: Path,
    max_depth: int,
) -> str:
    """Build a tree-style listing of *root* up to *max_depth* levels."""
    lines: list[str] = []
    truncated = False

    def _walk(dir_path: Path, depth: int, prefix: str) -> None:
        nonlocal truncated
        if truncated or depth > max_depth:
            return

        try:
            entries = sorted(dir_path.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if truncated:
                return
            if not _check_read_access(str(entry)):
                continue
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
            else:
                lines.append(f"{prefix}{entry.name}")
            if len(lines) >= MAX_LIST_ENTRIES:
                truncated = True
                return
            if entry.is_dir():
                _walk(entry, depth + 1, prefix + "  ")

    _walk(root, 1, "")

    if not lines:
        return "[empty directory]"
    result = "\n".join(lines)
    if truncated:
        result += (
            f"\n[{MAX_LIST_ENTRIES} entries shown."
            f" Use find_files for pattern-based search.]"
        )
    return result


async def list_directory(
    path: str = ".",
    *,
    recursive: bool = False,
    max_depth: int = 3,
) -> str:
    """List entries in a directory.

    Returns a formatted listing with directories marked by a trailing
    ``/``.  Use ``recursive=True`` for a tree-style view with depth
    capped at *max_depth*.

    Args:
        path: Directory to list.  Defaults to the current directory.
        recursive: When ``True``, recurse into subdirectories.
        max_depth: Maximum recursion depth (only applies when
            *recursive* is ``True``).  Defaults to 3.
    """
    _enforce_access(path, "READ")
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    if recursive:
        return _list_recursive(p, max_depth)

    raw_entries = sorted(p.iterdir())
    names = [e.name for e in raw_entries]
    names = _apply_listing_filter(names, path)

    dir_set = {e.name for e in raw_entries if e.is_dir()}
    formatted = [
        name + "/" if name in dir_set else name
        for name in names
    ]

    if not formatted:
        return "[empty directory]"

    total = len(formatted)
    if total > MAX_LIST_ENTRIES:
        formatted = formatted[:MAX_LIST_ENTRIES]
        result = "\n".join(formatted)
        result += (
            f"\n[{total} entries total, showing first {MAX_LIST_ENTRIES}."
            f" Use find_files for pattern-based search.]"
        )
        return result

    return "\n".join(formatted)


async def find_files(
    pattern: str,
    path: str = ".",
    *,
    type: Literal["file", "directory"] | None = None,
) -> str:
    """Find files and directories matching a glob pattern.

    Searches recursively under *path* for entries whose relative path
    matches *pattern* (using ``Path.rglob``).  Output is capped at
    ``MAX_FIND_RESULTS`` entries.

    Args:
        pattern: Glob pattern to match (e.g. ``"*.py"``, ``"test_*"``).
        path: Directory to search under.  Defaults to the current
            directory.
        type: Restrict results to ``"file"`` or ``"directory"``.
            When ``None`` (the default), both are included.
    """
    _enforce_access(path, "READ")
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    matches: list[str] = []
    for entry in sorted(p.rglob(pattern)):
        if type == "file" and not entry.is_file():
            continue
        if type == "directory" and not entry.is_dir():
            continue
        if not _check_read_access(str(entry)):
            continue
        rel = entry.relative_to(p).as_posix()
        if entry.is_dir():
            rel += "/"
        matches.append(rel)
        if len(matches) >= MAX_FIND_RESULTS:
            break

    if not matches:
        return f'No matches for pattern "{pattern}" in {path}.'

    result = "\n".join(matches)
    if len(matches) >= MAX_FIND_RESULTS:
        result += (
            f"\n[Results capped at {MAX_FIND_RESULTS}."
            f" Narrow your pattern to see more.]"
        )
    return result


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
    ignore_case: bool = False,
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
        glob: Optional file-path filter using fnmatch syntax (e.g.
            ``"*.py"`` or ``"src/**/*.py"``).  Matched against the
            path relative to *path*, not just the filename.
        use_regex: Interpret *pattern* as a Python regular expression.
        ignore_case: Perform case-insensitive matching.
        context_lines: Number of lines to show before and after each
            match (like ``grep -C``).
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        compiled = re.compile(
            pattern if use_regex else re.escape(pattern), flags,
        )
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
        and (glob is None or fnmatch_mod.fnmatch(f.relative_to(p).as_posix(), glob))
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


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants.

    Plain ``proc.kill()`` only terminates the immediate process.  When
    ``create_subprocess_shell`` is used, the immediate process is the
    shell (e.g. ``cmd.exe``), and child processes survive — keeping
    pipes open and causing ``communicate()`` to hang.  Walking the tree
    with ``psutil`` handles this reliably across platforms.
    """
    import psutil

    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


async def run_shell(
    command: str,
    working_directory: str | None = None,
    timeout: float = 120,
) -> str:
    """Run a shell command and return its combined stdout and stderr.

    Not included in ``ALL_BUILTIN_TOOLS`` — add it explicitly to
    specific agents or ``prompt()`` calls when needed.

    Args:
        command: Shell command to execute.
        working_directory: Directory to run the command in.  When
            ``None``, uses the current working directory.
        timeout: Maximum seconds to wait before killing the process.
            Defaults to 120.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=working_directory,
    )
    timed_out = False
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        _kill_process_tree(proc.pid)
        stdout, _ = await proc.communicate()
        timed_out = True

    output = stdout.decode(errors="replace") if stdout else ""

    if len(output) > MAX_READ_CHARS:
        full_len = len(output)
        output = (
            output[:MAX_READ_CHARS]
            + f"\n[output truncated: {full_len} chars total]"
        )

    if timed_out:
        return f"[timed out after {timeout}s]\n{output}"
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


async def ask_user(question: str) -> str:
    """Ask the human user a question and return their response.

    Requires an ``AskUserHandler`` to be configured on the active
    ``ExecutionContext``.  The CLI commands (``thorn run``,
    ``thorn chat``) provide a rich-console handler automatically.
    Raises ``RuntimeError`` if no handler is available.

    Args:
        question: The question to present to the user.
    """
    from thorn.core._context import get_context

    ctx = get_context()
    if ctx.ask_user_handler is None:
        raise RuntimeError(
            "ask_user is not available in this context. "
            "No user-interaction handler has been configured."
        )
    return await ctx.ask_user_handler(question)


# Register ToolCallNode subclasses on built-in tools so that
# HistoryTree records typed nodes, enabling isinstance-based
# identification (e.g. in context injection).
from thorn.core._history import DirectoryListCallNode, FileReadCallNode

read_file._thorn_call_node_class = FileReadCallNode  # type: ignore[attr-defined]
list_directory._thorn_call_node_class = DirectoryListCallNode  # type: ignore[attr-defined]


ALL_BUILTIN_TOOLS = [
    read_file,
    edit_file,
    create_file,
    delete_file,
    move_file,
    list_directory,
    find_files,
    search_files,
    ask_user,
]

# Pre-packaged toolsets for common capabilities.  These are plain lists
# that can be nested inside a tools= parameter and will be flattened
# automatically by _prepare_tools / _collect_tools.

FILE_READING: list = [read_file, list_directory, find_files, search_files]
FILE_WRITING: list = [FILE_READING, edit_file, create_file, delete_file, move_file]
