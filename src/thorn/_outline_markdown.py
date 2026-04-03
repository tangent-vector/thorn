"""Markdown-aware content hierarchy builder.

Parses Markdown content with ``mistletoe`` and groups block-level
tokens into heading-based sections to produce a ``ContentNode`` tree.
The result plugs into the same collapse-policy and formatting layers
as the indentation-based strategy in ``_outline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mistletoe.block_token import (
    BlockToken as MdBlockToken,
    Document as MdDocument,
    Heading as MdHeading,
    ListItem as MdListItem,
    Quote as MdQuote,
    SetextHeading as MdSetextHeading,
)
from mistletoe.block_token import List as MdList

from thorn._outline import ContentNode

_HEADING_TYPES: tuple[type, ...] = (MdHeading, MdSetextHeading)


@dataclass
class _Block:
    """A parsed block token annotated with its source line range."""

    token: object
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_markdown_hierarchy(lines: list[str]) -> ContentNode:
    """Build a ``ContentNode`` tree from Markdown content.

    Parses *lines* with ``mistletoe``, groups the resulting block
    tokens into heading-based sections, and produces a tree suitable
    for the collapse-policy and display-formatting layers in
    ``_outline``.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return ContentNode(start_line=1, end_line=0, depth=0)

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    doc = MdDocument(text)

    blocks = _annotate_children(doc.children or [], parent_end_line=total_lines)
    return _build_document_section(
        blocks, start_line=1, end_line=total_lines, base_depth=0,
    )


# ---------------------------------------------------------------------------
# Block annotation
# ---------------------------------------------------------------------------


def _annotate_children(
    tokens: Sequence[object],
    parent_end_line: int,
) -> list[_Block]:
    """Attach end-line numbers to sibling block tokens.

    ``mistletoe`` provides ``line_number`` (1-based start) on each
    block token but not the end line.  End lines are inferred from the
    next sibling's start line, or from *parent_end_line* for the last
    token.
    """
    if not tokens:
        return []

    result: list[_Block] = []
    for i, token in enumerate(tokens):
        start = getattr(token, "line_number", None)
        if start is None:
            continue
        if i + 1 < len(tokens):
            next_start = getattr(tokens[i + 1], "line_number", None)
            end = (next_start - 1) if next_start is not None else parent_end_line
        else:
            end = parent_end_line
        result.append(_Block(token=token, start_line=start, end_line=max(start, end)))

    return result


# ---------------------------------------------------------------------------
# Heading helpers
# ---------------------------------------------------------------------------


def _is_heading(block: _Block) -> bool:
    return isinstance(block.token, _HEADING_TYPES)


def _heading_level(block: _Block) -> int:
    return block.token.level  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Container promotion
# ---------------------------------------------------------------------------


def _contains_heading(token: object) -> bool:
    """Return whether *token* or any block-level descendant is a heading."""
    if isinstance(token, _HEADING_TYPES):
        return True
    children = getattr(token, "children", None)
    if not children:
        return False
    return any(
        isinstance(child, MdBlockToken) and _contains_heading(child)
        for child in children
    )


def _maybe_promote_container(block: _Block, depth: int) -> ContentNode | None:
    """Build a sub-hierarchy if *block* is a container with headings inside.

    Returns ``None`` when the block has no heading descendants, leaving
    its lines as own-lines of the parent section.
    """
    token = block.token

    if isinstance(token, MdQuote):
        if not _contains_heading(token):
            return None
        inner = _annotate_children(list(token.children or []), block.end_line)
        return _build_document_section(
            inner, block.start_line, block.end_line, depth,
        )

    if isinstance(token, MdList):
        if not _contains_heading(token):
            return None
        items = _annotate_children(list(token.children or []), block.end_line)
        children: list[ContentNode] = []
        for item in items:
            if isinstance(item.token, MdListItem) and _contains_heading(item.token):
                inner = _annotate_children(
                    list(item.token.children or []), item.end_line,
                )
                children.append(
                    _build_document_section(
                        inner, item.start_line, item.end_line, depth + 1,
                    ),
                )
        if not children:
            return None
        return ContentNode(
            start_line=block.start_line,
            end_line=block.end_line,
            depth=depth,
            children=children,
        )

    return None


# ---------------------------------------------------------------------------
# Section grouping
# ---------------------------------------------------------------------------


def _build_document_section(
    blocks: list[_Block],
    start_line: int,
    end_line: int,
    base_depth: int,
) -> ContentNode:
    """Build a ``ContentNode`` for a document-level context.

    A "document" (which also models the interior of a block quote or
    list item) has an optional leading level-1 heading, then lede
    (non-heading) blocks, then child sections starting with headings
    at any level.
    """
    children: list[ContentNode] = []
    i = 0

    if blocks and _is_heading(blocks[0]) and _heading_level(blocks[0]) == 1:
        i = 1

    while i < len(blocks) and not _is_heading(blocks[i]):
        promoted = _maybe_promote_container(blocks[i], base_depth + 1)
        if promoted is not None:
            children.append(promoted)
        i += 1

    children.extend(
        _collect_child_sections(blocks, i, section_level=0, base_depth=base_depth),
    )

    return ContentNode(
        start_line=start_line,
        end_line=end_line,
        depth=base_depth,
        children=children,
    )


def _collect_child_sections(
    blocks: list[_Block],
    start: int,
    *,
    section_level: int,
    base_depth: int,
) -> list[ContentNode]:
    """Partition ``blocks[start:]`` into child sections.

    Each child section begins with a heading whose level exceeds
    *section_level* and extends until a heading at the child's own
    level or lower is reached.
    """
    children: list[ContentNode] = []
    i = start
    while i < len(blocks):
        if not _is_heading(blocks[i]):
            i += 1
            continue

        h_level = _heading_level(blocks[i])
        if h_level <= section_level:
            break

        section_start = i
        i += 1
        while i < len(blocks):
            if _is_heading(blocks[i]) and _heading_level(blocks[i]) <= h_level:
                break
            i += 1

        children.append(
            _build_ordinary_section(
                blocks[section_start:i], h_level, base_depth + 1,
            ),
        )

    return children


def _build_ordinary_section(
    blocks: list[_Block],
    section_heading_level: int,
    depth: int,
) -> ContentNode:
    """Build a ``ContentNode`` for a heading-introduced section.

    ``blocks[0]`` is the heading.  Subsequent non-heading blocks are
    lede; remaining headings (with levels above *section_heading_level*)
    form child sub-sections.
    """
    children: list[ContentNode] = []
    i = 1

    while i < len(blocks) and not _is_heading(blocks[i]):
        promoted = _maybe_promote_container(blocks[i], depth + 1)
        if promoted is not None:
            children.append(promoted)
        i += 1

    children.extend(
        _collect_child_sections(
            blocks, i, section_level=section_heading_level, base_depth=depth,
        ),
    )

    return ContentNode(
        start_line=blocks[0].start_line,
        end_line=blocks[-1].end_line,
        depth=depth,
        children=children,
    )
