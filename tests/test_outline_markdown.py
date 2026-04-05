"""Tests for thorn._outline_markdown — Markdown-aware content hierarchy."""

from __future__ import annotations

import textwrap

from thorn.core._outline import (
    ContentNode,
    compute_collapse,
    format_outline,
    outline_and_format,
)
from thorn.core._outline_markdown import build_markdown_hierarchy


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).splitlines()


# ---------------------------------------------------------------------------
# Flat document (no headings)
# ---------------------------------------------------------------------------


class TestFlatDocument:
    def test_no_headings(self):
        lines = _lines("""\
            Some text

            More text
        """)
        root = build_markdown_hierarchy(lines)
        assert root.depth == 0
        assert root.start_line == 1
        assert root.end_line == len(lines)
        assert root.children == []
        assert root.own_line_count == len(lines)

    def test_empty(self):
        root = build_markdown_hierarchy([])
        assert root.line_count == 0
        assert root.children == []

    def test_single_line(self):
        root = build_markdown_hierarchy(["hello world"])
        assert root.start_line == 1
        assert root.end_line == 1
        assert root.depth == 0
        assert root.children == []


# ---------------------------------------------------------------------------
# Document with title + sections
# ---------------------------------------------------------------------------


class TestDocumentWithTitle:
    def test_title_and_two_sections(self):
        lines = _lines("""\
            # Title

            Some intro

            ## Section A

            Content A

            ## Section B

            Content B
        """)
        root = build_markdown_hierarchy(lines)
        assert root.depth == 0
        assert root.start_line == 1
        assert root.end_line == len(lines)
        assert len(root.children) == 2

        sec_a, sec_b = root.children
        assert sec_a.depth == 1
        assert sec_b.depth == 1
        assert sec_a.end_line < sec_b.start_line or sec_a.end_line == sec_b.start_line - 1

    def test_title_only(self):
        lines = _lines("""\
            # Just a Title

            And some body text.
        """)
        root = build_markdown_hierarchy(lines)
        assert root.depth == 0
        assert root.children == []
        assert root.own_line_count == len(lines)

    def test_title_with_another_h1(self):
        """A second level-1 heading becomes a child section."""
        lines = _lines("""\
            # Title

            Lede

            # Another Title

            More text
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 1
        child = root.children[0]
        assert child.depth == 1
        # Document own lines include the first title + lede.
        assert root.own_line_count > 0


# ---------------------------------------------------------------------------
# Nested sections
# ---------------------------------------------------------------------------


class TestNestedSections:
    def test_two_subsections(self):
        lines = _lines("""\
            ## Section A

            Text A

            ### Sub A1

            Sub A1 text

            ### Sub A2

            Sub A2 text
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 1

        sec_a = root.children[0]
        assert sec_a.depth == 1
        assert len(sec_a.children) == 2

        sub1, sub2 = sec_a.children
        assert sub1.depth == 2
        assert sub2.depth == 2
        assert sub1.end_line < sub2.start_line

    def test_three_levels(self):
        lines = _lines("""\
            # Doc

            ## Section

            ### Subsection

            #### Sub-sub
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 1  # ## Section

        sec = root.children[0]
        assert sec.depth == 1
        assert len(sec.children) == 1  # ### Subsection

        subsec = sec.children[0]
        assert subsec.depth == 2
        assert len(subsec.children) == 1  # #### Sub-sub

        subsubsec = subsec.children[0]
        assert subsubsec.depth == 3


# ---------------------------------------------------------------------------
# Mixed heading levels
# ---------------------------------------------------------------------------


class TestMixedHeadingLevels:
    def test_deep_then_shallow_are_siblings(self):
        """#### Deep followed by ### Shallow are both children of ##."""
        lines = _lines("""\
            ## Section A

            Text A

            #### Deep

            Deep text

            ### Shallow

            Shallow text
        """)
        root = build_markdown_hierarchy(lines)
        sec_a = root.children[0]
        assert sec_a.depth == 1
        assert len(sec_a.children) == 2

        deep, shallow = sec_a.children
        assert deep.depth == 2
        assert shallow.depth == 2

    def test_h2_ends_h4(self):
        """A heading at level 2 ends a section started by level 4."""
        lines = _lines("""\
            ## First

            #### Deep

            Deep text

            ## Second

            Second text
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 2
        first, second = root.children
        assert first.depth == 1
        assert second.depth == 1
        # Deep is a child of First.
        assert len(first.children) == 1
        assert first.children[0].depth == 2


# ---------------------------------------------------------------------------
# Document without title
# ---------------------------------------------------------------------------


class TestDocumentWithoutTitle:
    def test_headings_at_various_levels(self):
        lines = _lines("""\
            Some intro

            ## A

            Text A

            ### B

            Text B
        """)
        root = build_markdown_hierarchy(lines)
        assert root.depth == 0
        assert len(root.children) == 1  # ## A (with ### B nested)

        sec_a = root.children[0]
        assert sec_a.depth == 1
        assert len(sec_a.children) == 1  # ### B

        sec_b = sec_a.children[0]
        assert sec_b.depth == 2

    def test_all_same_level(self):
        lines = _lines("""\
            ## A

            ## B

            ## C
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 3
        for child in root.children:
            assert child.depth == 1

    def test_starts_with_h2(self):
        """Document starting with ## has no document heading."""
        lines = _lines("""\
            ## Section

            Content
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 1
        assert root.children[0].depth == 1


# ---------------------------------------------------------------------------
# Setext headings
# ---------------------------------------------------------------------------


class TestSetextHeadings:
    def test_setext_level1_as_title(self):
        lines = _lines("""\
            Title
            =====

            Some text

            Section
            -------

            Section text
        """)
        root = build_markdown_hierarchy(lines)
        assert root.depth == 0
        # The setext level-1 heading is the document heading.
        # The setext level-2 heading starts a child section.
        assert len(root.children) == 1
        child = root.children[0]
        assert child.depth == 1

    def test_setext_level2_sections(self):
        lines = _lines("""\
            Section A
            ---------

            Text A

            Section B
            ---------

            Text B
        """)
        root = build_markdown_hierarchy(lines)
        assert len(root.children) == 2
        for child in root.children:
            assert child.depth == 1


# ---------------------------------------------------------------------------
# Block quote with headings
# ---------------------------------------------------------------------------


class TestBlockQuoteWithHeadings:
    def test_promoted_to_child(self):
        """A block quote containing headings becomes a child ContentNode."""
        lines = _lines("""\
            Some text

            > ## Quoted Heading
            >
            > Quoted text
        """)
        root = build_markdown_hierarchy(lines)
        # The block quote should be promoted because it contains a heading.
        assert len(root.children) == 1
        bq_node = root.children[0]
        assert bq_node.depth == 1
        # Inside the block quote, the heading creates a section.
        assert len(bq_node.children) == 1
        inner_sec = bq_node.children[0]
        assert inner_sec.depth == 2

    def test_quote_in_lede_of_section(self):
        """Block quote with headings in a section's lede is promoted."""
        lines = _lines("""\
            # Title

            > ## Inside BQ

            ## Regular Section

            Text
        """)
        root = build_markdown_hierarchy(lines)
        # Children: promoted block quote + regular section.
        assert len(root.children) == 2
        bq, sec = root.children
        assert bq.start_line < sec.start_line


# ---------------------------------------------------------------------------
# Block quote without headings
# ---------------------------------------------------------------------------


class TestBlockQuoteWithoutHeadings:
    def test_stays_as_own_lines(self):
        """A block quote without headings remains own lines of the parent."""
        lines = _lines("""\
            # Title

            Some text

            > Just a quote
            > with two lines

            ## Section

            Content
        """)
        root = build_markdown_hierarchy(lines)
        # Only the ## Section is a child; the block quote is not promoted.
        assert len(root.children) == 1
        assert root.children[0].depth == 1


# ---------------------------------------------------------------------------
# List items with headings
# ---------------------------------------------------------------------------


class TestListWithHeadings:
    def test_list_with_heading_in_item(self):
        lines = _lines("""\
            - item one

              ## Heading in item

              content
        """)
        root = build_markdown_hierarchy(lines)
        # The list should be promoted because a list item has a heading.
        assert len(root.children) == 1
        list_node = root.children[0]
        assert list_node.depth == 1
        # The promoted list item should have a child for the heading section.
        assert len(list_node.children) == 1
        item_node = list_node.children[0]
        assert item_node.depth == 2

    def test_list_without_heading_not_promoted(self):
        lines = _lines("""\
            - item one
            - item two
            - item three
        """)
        root = build_markdown_hierarchy(lines)
        assert root.children == []


# ---------------------------------------------------------------------------
# Line range consistency
# ---------------------------------------------------------------------------


class TestLineRanges:
    def test_root_spans_entire_file(self):
        lines = _lines("""\
            # Title

            Intro

            ## A

            Text
        """)
        root = build_markdown_hierarchy(lines)
        assert root.start_line == 1
        assert root.end_line == len(lines)

    def test_children_within_parent(self):
        lines = _lines("""\
            # Title

            ## A

            Text A

            ## B

            Text B
        """)
        root = build_markdown_hierarchy(lines)
        for child in root.children:
            assert child.start_line >= root.start_line
            assert child.end_line <= root.end_line

    def test_siblings_non_overlapping(self):
        lines = _lines("""\
            ## A

            ## B

            ## C
        """)
        root = build_markdown_hierarchy(lines)
        for i in range(len(root.children) - 1):
            assert root.children[i].end_line < root.children[i + 1].start_line

    def test_own_line_count_non_negative(self):
        lines = _lines("""\
            # Title

            Intro

            ## A

            ### A1

            ## B
        """)
        root = build_markdown_hierarchy(lines)

        def _check(node: ContentNode) -> None:
            assert node.own_line_count >= 0, (
                f"Negative own_line_count at depth {node.depth}, "
                f"lines {node.start_line}-{node.end_line}"
            )
            for child in node.children:
                _check(child)

        _check(root)


# ---------------------------------------------------------------------------
# Integration with collapse policy
# ---------------------------------------------------------------------------


class TestCollapseIntegration:
    def _make_large_md(self, sections: int = 10, body_lines: int = 20) -> list[str]:
        """Build a markdown document that exceeds typical budgets."""
        parts = ["# Big Document", "", "Introduction paragraph.", ""]
        for i in range(sections):
            parts.append(f"## Section {i + 1}")
            parts.append("")
            for j in range(body_lines):
                parts.append(f"Line {j + 1} of section {i + 1}.")
            parts.append("")
        return parts

    def test_outline_shows_headings(self):
        """With a tight budget, section headings should appear."""
        lines = self._make_large_md(sections=5, body_lines=40)
        budget = 30
        root = build_markdown_hierarchy(lines)
        spans = compute_collapse(root, line_budget=budget)
        total_output = sum(s.output_line_count for s in spans)
        assert total_output <= budget
        result = format_outline(lines, spans, char_budget=50_000)
        # The document title should be visible.
        assert "# Big Document" in result
        # At least one section heading should be visible (via partial
        # expansion, the collapse policy shows leading lines from
        # collapsed regions).
        assert "## Section" in result

    def test_outline_and_format_dispatches(self):
        """outline_and_format uses the markdown strategy for .md paths."""
        lines = self._make_large_md(sections=3, body_lines=30)
        result = outline_and_format(
            lines,
            line_budget=20,
            char_budget=50_000,
            file_path="docs/README.md",
        )
        assert "[Outline:" in result
        assert "# Big Document" in result

    def test_outline_and_format_with_markdown_extension(self):
        """Both .md and .markdown trigger the markdown strategy."""
        lines = self._make_large_md(sections=3, body_lines=30)
        for ext in (".md", ".markdown"):
            result = outline_and_format(
                lines,
                line_budget=20,
                char_budget=50_000,
                file_path=f"file{ext}",
            )
            assert "[Outline:" in result

    def test_small_file_shown_fully(self):
        lines = _lines("""\
            # Title

            Some text

            ## Section

            Content
        """)
        root = build_markdown_hierarchy(lines)
        spans = compute_collapse(root, line_budget=100)
        assert len(spans) == 1
        assert spans[0].visible
        assert spans[0].line_count == len(lines)

    def test_budget_respected(self):
        lines = self._make_large_md(sections=10, body_lines=50)
        budget = 40
        result = outline_and_format(
            lines,
            line_budget=budget,
            char_budget=50_000,
            file_path="test.md",
        )
        content_lines = result.split("\n")
        non_footer = [l for l in content_lines if not l.startswith("[Outline")]
        assert len(non_footer) <= budget
