"""Content-aware file outlining.

Provides structural outlines of files that exceed a size budget, showing
top-level structure with bodies collapsed.  Three layers:

1. **Content Hierarchy** -- a pluggable strategy parses lines into a
   ``ContentNode`` tree reflecting nesting structure.  Indentation-based
   and Markdown-aware strategies are available; selection is driven by
   file extension via ``detect_content_format``.
2. **Collapse Policy** -- a budget-driven algorithm selects which regions
   to show verbatim and which to collapse to summary lines.
3. **Display Formatting** -- renders the tree and collapse decisions into
   the same line-numbered text format that ``read_file`` uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import PurePosixPath, PureWindowsPath


# ---------------------------------------------------------------------------
# Layer 1: Content Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class ContentNode:
    """A contiguous region of lines at a particular nesting depth.

    Lines within this node's range that are at exactly ``depth`` are the
    node's *own lines*.  Contiguous runs of deeper lines form children.
    """

    start_line: int
    """1-based inclusive start."""

    end_line: int
    """1-based inclusive end."""

    depth: int
    """Nesting depth of this region.

    Semantics depend on the strategy that built the hierarchy: for
    indentation-based hierarchies this is the minimum indent level in
    spaces; for Markdown hierarchies it is the section nesting level.
    """

    children: list[ContentNode] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def own_line_count(self) -> int:
        """Lines at exactly this depth (not inside any child)."""
        return self.line_count - sum(c.line_count for c in self.children)


def compute_depths(lines: list[str], *, tab_width: int = 4) -> list[int]:
    """Assign an effective indentation depth to every line.

    Blank lines are classified using surrounding context so they don't
    create artificial nesting boundaries: each blank line receives
    ``min(prev_nonblank_indent, next_nonblank_indent)``.
    """
    n = len(lines)
    if n == 0:
        return []

    raw: list[int | None] = []
    for line in lines:
        expanded = line.expandtabs(tab_width)
        stripped = expanded.lstrip(" ")
        if stripped:
            raw.append(len(expanded) - len(stripped))
        else:
            raw.append(None)

    prev_nonblank: list[int | None] = [None] * n
    last: int | None = None
    for i in range(n):
        if raw[i] is not None:
            last = raw[i]
        prev_nonblank[i] = last

    next_nonblank: list[int | None] = [None] * n
    last = None
    for i in range(n - 1, -1, -1):
        if raw[i] is not None:
            last = raw[i]
        next_nonblank[i] = last

    depths: list[int] = []
    for i in range(n):
        if raw[i] is not None:
            depths.append(raw[i])
        elif prev_nonblank[i] is not None and next_nonblank[i] is not None:
            depths.append(min(prev_nonblank[i], next_nonblank[i]))
        elif prev_nonblank[i] is not None:
            depths.append(prev_nonblank[i])
        elif next_nonblank[i] is not None:
            depths.append(next_nonblank[i])
        else:
            depths.append(0)

    return depths


def build_hierarchy(depths: list[int]) -> ContentNode:
    """Build a ``ContentNode`` tree from per-line depth values.

    The root spans the entire file at the file's minimum depth.
    """
    if not depths:
        return ContentNode(start_line=1, end_line=0, depth=0)

    min_depth = min(depths)
    return _build_node(depths, 0, len(depths) - 1, min_depth)


def _build_node(
    depths: list[int], start_idx: int, end_idx: int, node_depth: int,
) -> ContentNode:
    children: list[ContentNode] = []
    i = start_idx
    while i <= end_idx:
        if depths[i] > node_depth:
            child_start = i
            child_min = depths[i]
            i += 1
            while i <= end_idx and depths[i] > node_depth:
                child_min = min(child_min, depths[i])
                i += 1
            children.append(
                _build_node(depths, child_start, i - 1, child_min),
            )
        else:
            i += 1

    return ContentNode(
        start_line=start_idx + 1,
        end_line=end_idx + 1,
        depth=node_depth,
        children=children,
    )


# ---------------------------------------------------------------------------
# Layer 2: Collapse Policy
# ---------------------------------------------------------------------------


@dataclass
class OutputSpan:
    """A contiguous range of file lines to either show or collapse."""

    start_line: int
    """1-based inclusive start."""

    end_line: int
    """1-based inclusive end."""

    visible: bool
    """True = show lines verbatim; False = show a one-line summary."""

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def output_line_count(self) -> int:
        """Lines this span contributes to the formatted output."""
        return self.line_count if self.visible else 1


def compute_collapse(
    root: ContentNode,
    *,
    line_budget: int,
) -> list[OutputSpan]:
    """Select which regions to show and which to collapse within budget.

    Uses a histogram-like approach: iterates over distinct depth values
    to find the deepest cutoff D where showing all lines at depth <= D
    (plus one summary line per collapsed region) fits within
    *line_budget*.  Surplus budget is distributed across collapsed
    regions for partial expansion.
    """
    total_lines = root.line_count
    if total_lines == 0:
        return []

    if total_lines <= line_budget:
        return [OutputSpan(root.start_line, root.end_line, visible=True)]

    depth_set: set[int] = set()
    _collect_depths(root, depth_set)
    sorted_depths = sorted(depth_set)

    best_spans: list[OutputSpan] | None = None

    for d in sorted_depths:
        spans = _build_spans(root, d)
        if sum(s.output_line_count for s in spans) <= line_budget:
            best_spans = spans
        else:
            break

    if best_spans is None:
        return _truncation_spans(total_lines, line_budget)

    surplus = line_budget - sum(s.output_line_count for s in best_spans)
    if surplus > 0:
        best_spans = _partially_expand(best_spans, surplus)

    return _finalize_spans(best_spans)


def _collect_depths(node: ContentNode, out: set[int]) -> None:
    out.add(node.depth)
    for child in node.children:
        _collect_depths(child, out)


def _build_spans(root: ContentNode, cutoff: int) -> list[OutputSpan]:
    """Produce spans by collapsing nodes deeper than *cutoff*."""
    raw: list[OutputSpan] = []
    _walk_for_spans(root, cutoff, raw)
    return _finalize_spans(raw)


def _walk_for_spans(
    node: ContentNode, cutoff: int, out: list[OutputSpan],
) -> None:
    if node.depth > cutoff:
        out.append(OutputSpan(node.start_line, node.end_line,
                              visible=node.line_count <= 1))
        return

    current = node.start_line
    for child in node.children:
        if current < child.start_line:
            out.append(OutputSpan(current, child.start_line - 1, visible=True))
        _walk_for_spans(child, cutoff, out)
        current = child.end_line + 1
    if current <= node.end_line:
        out.append(OutputSpan(current, node.end_line, visible=True))


def _finalize_spans(spans: list[OutputSpan]) -> list[OutputSpan]:
    """Suppress single-line collapses, then merge adjacent visible spans."""
    suppressed = [
        OutputSpan(s.start_line, s.end_line, visible=True)
        if not s.visible and s.line_count <= 1
        else s
        for s in spans
    ]
    return _merge_visible(suppressed)


def _merge_visible(spans: list[OutputSpan]) -> list[OutputSpan]:
    if not spans:
        return spans
    merged = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if span.visible and prev.visible and span.start_line == prev.end_line + 1:
            merged[-1] = OutputSpan(prev.start_line, span.end_line, visible=True)
        else:
            merged.append(span)
    return merged


def _truncation_spans(total_lines: int, line_budget: int) -> list[OutputSpan]:
    """Fallback for files with no useful hierarchical structure."""
    if total_lines <= line_budget:
        return [OutputSpan(1, total_lines, visible=True)]
    return [
        OutputSpan(1, line_budget, visible=True),
        OutputSpan(line_budget + 1, total_lines, visible=False),
    ]


def _partially_expand(
    spans: list[OutputSpan], surplus: int,
) -> list[OutputSpan]:
    """Distribute *surplus* extra visible lines across collapsed spans.

    Each collapsed span of N lines currently costs 1 output line.
    Showing M additional lines from it costs M output lines (the M
    visible lines plus one summary for the remainder, minus the
    original summary).  Full expansion costs N - 1 additional lines.
    """
    collapsed = [(i, s) for i, s in enumerate(spans) if not s.visible]
    if not collapsed:
        return spans

    capacities = [(i, s.line_count - 1) for i, s in collapsed]
    allocations = _distribute_evenly(capacities, surplus)

    result: list[OutputSpan] = []
    for i, span in enumerate(spans):
        extra = allocations.get(i, 0)
        if extra <= 0 or span.visible:
            result.append(span)
            continue

        if extra >= span.line_count - 1:
            result.append(
                OutputSpan(span.start_line, span.end_line, visible=True),
            )
        else:
            split = span.start_line + extra - 1
            result.append(OutputSpan(span.start_line, split, visible=True))
            result.append(
                OutputSpan(split + 1, span.end_line, visible=False),
            )

    return result


def _distribute_evenly(
    capacities: list[tuple[int, int]], total: int,
) -> dict[int, int]:
    """Distribute *total* units among items as evenly as possible.

    *capacities* maps item indices to their maximum allocation.
    Smaller-capacity items are filled first so that the distribution
    stays uniform.
    """
    if not capacities or total <= 0:
        return {}

    sorted_caps = sorted(capacities, key=lambda x: x[1])
    allocations: dict[int, int] = {}
    remaining = total

    for pos, (idx, cap) in enumerate(sorted_caps):
        items_left = len(sorted_caps) - pos
        per_item = remaining // items_left
        give = min(per_item, cap)
        allocations[idx] = give
        remaining -= give

    return allocations


# ---------------------------------------------------------------------------
# Layer 3: Display Formatting
# ---------------------------------------------------------------------------


def format_outline(
    lines: list[str],
    spans: list[OutputSpan],
    *,
    char_budget: int,
) -> str:
    """Render *spans* into line-numbered text.

    Visible spans use the same ``NNN| content`` format as ``read_file``.
    Collapsed spans produce a bracketed summary.  A footer indicates
    total file size.
    """
    total_lines = len(lines)
    width = max(len(str(total_lines)), 1)

    parts: list[str] = []
    char_count = 0
    hit_char_limit = False

    for span in spans:
        if hit_char_limit:
            break

        if span.visible:
            for line_num in range(span.start_line, span.end_line + 1):
                formatted = f"{line_num:>{width}}| {lines[line_num - 1]}"
                char_count += len(formatted) + 1
                if char_count > char_budget:
                    hit_char_limit = True
                    break
                parts.append(formatted)
        else:
            summary = (
                f"[lines {span.start_line}-{span.end_line},"
                f" {span.line_count} lines"
                f" -- use offset/limit to read]"
            )
            char_count += len(summary) + 1
            if char_count > char_budget:
                hit_char_limit = True
                break
            parts.append(summary)

    if hit_char_limit:
        parts.append(
            f"[Outline truncated at character limit."
            f" {total_lines} lines total."
            f" Use offset/limit to read specific regions.]"
        )
    else:
        parts.append(
            f"[Outline: {total_lines} lines total."
            f" Use offset/limit to read specific regions.]"
        )

    return "\n".join(parts)


def spans_for_regions(
    total_lines: int,
    regions: list[tuple[int, int]],
    context_lines: int = 3,
) -> list[OutputSpan]:
    """Build ``OutputSpan`` list showing context around specific line regions.

    Creates visible windows around each *region* (1-based inclusive
    ``(start, end)`` tuples) padded by *context_lines*, collapses
    everything else, and merges overlapping or adjacent windows.
    """
    if total_lines == 0:
        return []

    if not regions:
        return [OutputSpan(1, total_lines, visible=False)]

    sorted_regions = sorted(regions)

    visible: list[tuple[int, int]] = []
    for start, end in sorted_regions:
        v_start = max(1, start - context_lines)
        v_end = min(total_lines, end + context_lines)
        if visible and v_start <= visible[-1][1] + 1:
            visible[-1] = (visible[-1][0], max(visible[-1][1], v_end))
        else:
            visible.append((v_start, v_end))

    spans: list[OutputSpan] = []
    cursor = 1
    for v_start, v_end in visible:
        if cursor < v_start:
            spans.append(OutputSpan(cursor, v_start - 1, visible=False))
        spans.append(OutputSpan(v_start, v_end, visible=True))
        cursor = v_end + 1
    if cursor <= total_lines:
        spans.append(OutputSpan(cursor, total_lines, visible=False))

    return spans


# ---------------------------------------------------------------------------
# Format Detection
# ---------------------------------------------------------------------------

_MARKDOWN_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown"})


class ContentFormat(Enum):
    """File-content format used to select a hierarchy-building strategy."""

    INDENTATION = auto()
    MARKDOWN = auto()


def detect_content_format(file_path: str | None) -> ContentFormat:
    """Choose a hierarchy strategy based on *file_path*.

    Currently only inspects the file extension.  Content-based
    inference can be added later.
    """
    if file_path is not None:
        # Accept both Windows and POSIX paths.
        suffix = ""
        for cls in (PureWindowsPath, PurePosixPath):
            suffix = cls(file_path).suffix.lower()
            if suffix:
                break
        if suffix in _MARKDOWN_EXTENSIONS:
            return ContentFormat.MARKDOWN
    return ContentFormat.INDENTATION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def outline_and_format(
    lines: list[str],
    *,
    line_budget: int,
    char_budget: int,
    file_path: str | None = None,
) -> str:
    """Generate a budget-constrained outline of file content.

    Builds a content hierarchy (strategy chosen from *file_path* when
    provided, falling back to indentation), selects which regions to
    show or collapse based on the line and character budgets, and
    formats the result as line-numbered text.
    """
    fmt = detect_content_format(file_path)
    if fmt is ContentFormat.MARKDOWN:
        from thorn.core._outline_markdown import build_markdown_hierarchy

        root = build_markdown_hierarchy(lines)
    else:
        depths = compute_depths(lines)
        root = build_hierarchy(depths)
    spans = compute_collapse(root, line_budget=line_budget)
    return format_outline(lines, spans, char_budget=char_budget)
