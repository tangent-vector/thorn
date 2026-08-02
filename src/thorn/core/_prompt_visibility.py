"""Prompt-render visibility facts for file content.

The types in this module record what source file content was actually
rendered into a provider prompt.  They intentionally report facts, not
policy decisions: callers decide whether a visibility comparison should
be treated as redundant, surprising, or acceptable.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Iterable, Mapping

from thorn.core._messages import Message

_LINE_NUMBERED_CONTENT_RE = re.compile(
    r"^\s*(?P<line>\d+)(?P<marker>[^|\d])?\| ?(?P<content>.*)$",
)
_SANDBOX_WORKSPACE_PREFIX = PurePosixPath("/agent/workspace")


@dataclass(frozen=True)
class FileContentKey:
    """Stable identity for file content observed by one agent session."""

    resolved_path: Path


@dataclass(frozen=True)
class FileLineContent:
    """One line of source content with its 1-based line number."""

    line_number: int
    content: str

    def __post_init__(self) -> None:
        if self.line_number < 1:
            raise ValueError("line_number must be >= 1")


@dataclass(frozen=True)
class FileLineRange:
    """A 1-based inclusive file line range."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError("start_line must be >= 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")


@dataclass(frozen=True)
class FileContentVisibilityComparison:
    """Exact-content visibility facts for a file-content query."""

    file_key: FileContentKey
    compared_line_count: int
    exact_visible_line_count: int
    visible_line_mismatch_count: int
    not_visible_line_count: int

    @property
    def exact_visible_fraction(self) -> float:
        """Fraction of compared lines that exactly matched visible content."""
        if self.compared_line_count == 0:
            return 0.0
        return self.exact_visible_line_count / self.compared_line_count


@dataclass(frozen=True)
class PromptVisibilitySnapshot:
    """Content-visibility facts for one rendered provider prompt."""

    visible_file_line_hashes_by_path: Mapping[
        FileContentKey,
        Mapping[int, str],
    ] = field(default_factory=dict)
    collapsed_file_ranges_by_path: Mapping[
        FileContentKey,
        tuple[FileLineRange, ...],
    ] = field(default_factory=dict)

    def compare_file_content(
        self,
        file_key: FileContentKey,
        lines: Iterable[FileLineContent],
    ) -> FileContentVisibilityComparison:
        """Compare source lines against exactly visible prompt content."""
        candidate_lines = tuple(lines)
        visible_lines = self.visible_file_line_hashes_by_path.get(file_key, {})
        exact_visible = 0
        visible_mismatch = 0
        not_visible = 0

        for line in candidate_lines:
            visible_hash = visible_lines.get(line.line_number)
            if visible_hash is None:
                not_visible += 1
                continue
            if visible_hash == _line_hash(line.content):
                exact_visible += 1
            else:
                visible_mismatch += 1

        return FileContentVisibilityComparison(
            file_key=file_key,
            compared_line_count=len(candidate_lines),
            exact_visible_line_count=exact_visible,
            visible_line_mismatch_count=visible_mismatch,
            not_visible_line_count=not_visible,
        )


@dataclass
class PromptVisibilitySnapshotBuilder:
    """Mutable builder for a prompt visibility snapshot."""

    _visible_file_line_hashes_by_path: dict[
        FileContentKey,
        dict[int, str],
    ] = field(default_factory=dict)
    _collapsed_file_ranges_by_path: dict[
        FileContentKey,
        list[FileLineRange],
    ] = field(default_factory=dict)

    def record_visible_file_lines(
        self,
        file_key: FileContentKey,
        lines: Iterable[FileLineContent],
    ) -> None:
        """Record file lines that were emitted verbatim into the prompt."""
        visible_lines = self._visible_file_line_hashes_by_path.setdefault(
            file_key,
            {},
        )
        for line in lines:
            visible_lines[line.line_number] = _line_hash(line.content)

    def record_collapsed_file_ranges(
        self,
        file_key: FileContentKey,
        ranges: Iterable[FileLineRange],
    ) -> None:
        """Record file ranges referenced by collapsed prompt placeholders."""
        collapsed_ranges = self._collapsed_file_ranges_by_path.setdefault(
            file_key,
            [],
        )
        collapsed_ranges.extend(ranges)

    def build(self) -> PromptVisibilitySnapshot:
        """Return an immutable snapshot of the recorded facts."""
        visible_lines = {
            key: MappingProxyType(dict(line_hashes))
            for key, line_hashes in self._visible_file_line_hashes_by_path.items()
        }
        collapsed_ranges = {
            key: tuple(ranges)
            for key, ranges in self._collapsed_file_ranges_by_path.items()
        }
        return PromptVisibilitySnapshot(
            visible_file_line_hashes_by_path=MappingProxyType(visible_lines),
            collapsed_file_ranges_by_path=MappingProxyType(collapsed_ranges),
        )


@dataclass
class PromptVisibilityStore:
    """Bounded retention for prompt visibility snapshots by render ID."""

    max_snapshots: int = 32
    _snapshots_by_render_id: OrderedDict[str, PromptVisibilitySnapshot] = field(
        default_factory=OrderedDict,
    )

    def remember(
        self,
        render_id: str,
        snapshot: PromptVisibilitySnapshot,
    ) -> None:
        """Retain visibility facts for one provider request/render."""
        if self.max_snapshots <= 0:
            return
        self._snapshots_by_render_id.pop(render_id, None)
        self._snapshots_by_render_id[render_id] = snapshot
        while len(self._snapshots_by_render_id) > self.max_snapshots:
            self._snapshots_by_render_id.popitem(last=False)

    def get(self, render_id: str) -> PromptVisibilitySnapshot | None:
        """Return retained visibility facts for *render_id*, if present."""
        return self._snapshots_by_render_id.get(render_id)

    def compare_file_content(
        self,
        *,
        render_id: str,
        raw_path: str,
        workspace_root: Path | None,
        lines: Iterable[FileLineContent],
    ) -> FileContentVisibilityComparison | None:
        """Compare source lines to content visible in a retained render."""
        snapshot = self.get(render_id)
        if snapshot is None:
            return None
        file_key = file_content_key_for_path(
            raw_path,
            workspace_root=workspace_root,
        )
        if file_key is None:
            return None
        return snapshot.compare_file_content(file_key, lines)


@dataclass(frozen=True)
class RenderedHistory:
    """Provider messages plus visibility facts from the same render."""

    messages: list[Message]
    visibility: PromptVisibilitySnapshot
    folded_file_tool_call_ids: frozenset[str] = frozenset()


def parse_line_numbered_file_content(content: str) -> tuple[FileLineContent, ...]:
    """Parse Thorn line-numbered file content from rendered tool output."""
    lines: list[FileLineContent] = []
    for rendered_line in content.splitlines():
        match = _LINE_NUMBERED_CONTENT_RE.match(rendered_line)
        if match is None:
            continue
        lines.append(FileLineContent(
            line_number=int(match.group("line")),
            content=match.group("content"),
        ))
    return tuple(lines)


def file_content_key_for_path(
    raw_path: str | Path,
    *,
    workspace_root: Path | None,
) -> FileContentKey | None:
    """Resolve a tool-visible path into a file-content visibility key."""
    raw_text = str(raw_path)
    try:
        if workspace_root is not None:
            sandbox_relative = _sandbox_workspace_relative_path(raw_text)
            if sandbox_relative is not None:
                host_relative = _relative_to_active_workspace(
                    sandbox_relative,
                    workspace_root,
                )
                return FileContentKey(
                    (workspace_root / host_relative).resolve(strict=False),
                )

        raw_file_path = Path(raw_text)
        if raw_file_path.is_absolute():
            return FileContentKey(raw_file_path.expanduser().resolve(strict=False))
        if workspace_root is not None:
            return FileContentKey(
                (workspace_root / raw_file_path).resolve(strict=False),
            )
        return FileContentKey(raw_file_path.expanduser().resolve(strict=False))
    except OSError:
        return None


def _line_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _relative_to_active_workspace(
    sandbox_relative: Path,
    workspace_root: Path,
) -> Path:
    sandbox_parts = sandbox_relative.parts
    workspace_parts = workspace_root.parts
    max_prefix_len = min(len(sandbox_parts), len(workspace_parts))
    for prefix_len in range(max_prefix_len, 0, -1):
        if sandbox_parts[:prefix_len] == workspace_parts[-prefix_len:]:
            return Path(*sandbox_parts[prefix_len:])
    return sandbox_relative


def _sandbox_workspace_relative_path(raw_path: str) -> Path | None:
    posix_path = PurePosixPath(raw_path)
    try:
        relative = posix_path.relative_to(_SANDBOX_WORKSPACE_PREFIX)
    except ValueError:
        return None
    if any(part == ".." for part in relative.parts):
        return None
    return Path(*relative.parts)
