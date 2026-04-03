"""Tests for thorn._outline — content-aware file outlining."""

from __future__ import annotations

import textwrap

from thorn._outline import (
    ContentFormat,
    ContentNode,
    OutputSpan,
    build_hierarchy,
    compute_collapse,
    compute_depths,
    detect_content_format,
    format_outline,
    outline_and_format,
    spans_for_regions,
    _distribute_evenly,
)


# ---------------------------------------------------------------------------
# Layer 1: compute_depths
# ---------------------------------------------------------------------------


class TestComputeDepths:
    def test_empty(self):
        assert compute_depths([]) == []

    def test_single_line(self):
        assert compute_depths(["hello"]) == [0]

    def test_indented_lines(self):
        lines = ["top", "    nested", "        deep", "    nested2", "top2"]
        assert compute_depths(lines) == [0, 4, 8, 4, 0]

    def test_blank_lines_classified_by_context(self):
        lines = ["top", "    body", "", "    body2", "top2"]
        depths = compute_depths(lines)
        assert depths[2] == min(4, 4)  # blank between two depth-4 lines

    def test_blank_line_between_different_depths(self):
        lines = ["top", "    nested", "", "top2"]
        depths = compute_depths(lines)
        assert depths[2] == min(4, 0)  # blank between depth 4 and depth 0

    def test_blank_line_at_start(self):
        lines = ["", "top", "    nested"]
        depths = compute_depths(lines)
        assert depths[0] == 0  # no prev, uses next nonblank indent

    def test_blank_line_at_end(self):
        lines = ["top", "    nested", ""]
        depths = compute_depths(lines)
        assert depths[2] == 4  # no next, uses prev nonblank indent

    def test_all_blank(self):
        assert compute_depths(["", "", ""]) == [0, 0, 0]

    def test_tabs_expanded(self):
        lines = ["\ttabbed", "\t\tdeep"]
        depths = compute_depths(lines, tab_width=4)
        assert depths == [4, 8]

    def test_custom_tab_width(self):
        lines = ["\ttabbed"]
        assert compute_depths(lines, tab_width=2) == [2]

    def test_whitespace_only_treated_as_blank(self):
        lines = ["top", "   ", "top2"]
        depths = compute_depths(lines)
        assert depths[1] == 0  # whitespace-only → blank → min(0, 0)


# ---------------------------------------------------------------------------
# Layer 1: build_hierarchy
# ---------------------------------------------------------------------------


class TestBuildHierarchy:
    def test_empty(self):
        root = build_hierarchy([])
        assert root.line_count == 0
        assert root.children == []

    def test_flat(self):
        root = build_hierarchy([0, 0, 0])
        assert root.start_line == 1
        assert root.end_line == 3
        assert root.depth == 0
        assert root.children == []
        assert root.own_line_count == 3

    def test_single_child(self):
        root = build_hierarchy([0, 4, 4, 0])
        assert root.depth == 0
        assert root.own_line_count == 2  # lines 1 and 4
        assert len(root.children) == 1
        child = root.children[0]
        assert child.start_line == 2
        assert child.end_line == 3
        assert child.depth == 4
        assert child.line_count == 2

    def test_multiple_children(self):
        root = build_hierarchy([0, 4, 0, 4, 4, 0])
        assert len(root.children) == 2
        assert root.children[0].start_line == 2
        assert root.children[0].end_line == 2
        assert root.children[1].start_line == 4
        assert root.children[1].end_line == 5

    def test_nested_children(self):
        root = build_hierarchy([0, 4, 8, 4, 0])
        assert len(root.children) == 1
        child = root.children[0]
        assert child.depth == 4
        assert len(child.children) == 1
        grandchild = child.children[0]
        assert grandchild.depth == 8
        assert grandchild.start_line == 3
        assert grandchild.end_line == 3

    def test_depth_gap(self):
        """Depth skips (e.g., 0 → 8 with no depth-4 lines) still work."""
        root = build_hierarchy([0, 8, 0])
        assert len(root.children) == 1
        assert root.children[0].depth == 8


# ---------------------------------------------------------------------------
# Layer 2: _distribute_evenly
# ---------------------------------------------------------------------------


class TestDistributeEvenly:
    def test_empty(self):
        assert _distribute_evenly([], 10) == {}

    def test_zero_total(self):
        assert _distribute_evenly([(0, 5)], 0) == {}

    def test_single_item(self):
        assert _distribute_evenly([(0, 10)], 5) == {0: 5}

    def test_single_item_capped(self):
        assert _distribute_evenly([(0, 3)], 10) == {0: 3}

    def test_even_split(self):
        result = _distribute_evenly([(0, 10), (1, 10)], 8)
        assert result[0] == 4
        assert result[1] == 4

    def test_uneven_capacity(self):
        result = _distribute_evenly([(0, 2), (1, 10)], 8)
        assert result[0] == 2  # capped at capacity
        assert result[1] == 6  # gets the remainder

    def test_all_items_filled(self):
        result = _distribute_evenly([(0, 3), (1, 3)], 100)
        assert result[0] == 3
        assert result[1] == 3

    def test_preserves_indices(self):
        result = _distribute_evenly([(5, 10), (2, 10)], 6)
        assert 5 in result
        assert 2 in result
        assert result[5] + result[2] == 6


# ---------------------------------------------------------------------------
# spans_for_regions
# ---------------------------------------------------------------------------


class TestSpansForRegions:
    def test_empty_file(self):
        assert spans_for_regions(0, [(1, 1)]) == []

    def test_no_regions_collapses_all(self):
        spans = spans_for_regions(10, [])
        assert len(spans) == 1
        assert not spans[0].visible
        assert spans[0].line_count == 10

    def test_single_region_with_context(self):
        spans = spans_for_regions(20, [(10, 10)], context_lines=2)
        visible_spans = [s for s in spans if s.visible]
        collapsed_spans = [s for s in spans if not s.visible]
        assert len(visible_spans) == 1
        assert visible_spans[0].start_line == 8
        assert visible_spans[0].end_line == 12
        assert len(collapsed_spans) == 2

    def test_region_at_file_start(self):
        spans = spans_for_regions(20, [(1, 1)], context_lines=3)
        assert spans[0].visible
        assert spans[0].start_line == 1
        assert spans[0].end_line == 4

    def test_region_at_file_end(self):
        spans = spans_for_regions(20, [(20, 20)], context_lines=3)
        visible = [s for s in spans if s.visible]
        assert visible[-1].end_line == 20
        assert visible[-1].start_line == 17

    def test_overlapping_regions_merged(self):
        spans = spans_for_regions(30, [(5, 5), (8, 8)], context_lines=3)
        visible = [s for s in spans if s.visible]
        assert len(visible) == 1
        assert visible[0].start_line == 2
        assert visible[0].end_line == 11

    def test_distant_regions_stay_separate(self):
        spans = spans_for_regions(50, [(5, 5), (45, 45)], context_lines=2)
        visible = [s for s in spans if s.visible]
        assert len(visible) == 2
        assert visible[0].end_line < visible[1].start_line

    def test_region_spanning_multiple_lines(self):
        spans = spans_for_regions(30, [(10, 15)], context_lines=2)
        visible = [s for s in spans if s.visible]
        assert len(visible) == 1
        assert visible[0].start_line == 8
        assert visible[0].end_line == 17

    def test_unsorted_regions_handled(self):
        spans_ordered = spans_for_regions(30, [(5, 5), (25, 25)], context_lines=2)
        spans_reversed = spans_for_regions(30, [(25, 25), (5, 5)], context_lines=2)
        assert spans_ordered == spans_reversed

    def test_adjacent_regions_merged(self):
        # Regions with context_lines=2: region (5,5) -> visible 3-7,
        # region (8,8) -> visible 6-10.  These overlap so should merge.
        spans = spans_for_regions(20, [(5, 5), (8, 8)], context_lines=2)
        visible = [s for s in spans if s.visible]
        assert len(visible) == 1

    def test_whole_file_covered(self):
        spans = spans_for_regions(5, [(1, 5)], context_lines=0)
        assert len(spans) == 1
        assert spans[0].visible
        assert spans[0].start_line == 1
        assert spans[0].end_line == 5


# ---------------------------------------------------------------------------
# Layer 2: compute_collapse
# ---------------------------------------------------------------------------


class TestComputeCollapse:
    def _make_root(self, depths: list[int]) -> ContentNode:
        return build_hierarchy(depths)

    def test_fits_in_budget(self):
        root = self._make_root([0, 0, 0])
        spans = compute_collapse(root, line_budget=10)
        assert len(spans) == 1
        assert spans[0].visible
        assert spans[0].line_count == 3

    def test_flat_file_truncation(self):
        root = self._make_root([0] * 20)
        spans = compute_collapse(root, line_budget=10)
        assert spans[0].visible
        assert spans[0].line_count == 10
        assert not spans[1].visible
        assert spans[1].line_count == 10

    def test_basic_collapse(self):
        depths = [0, 4, 4, 4, 0]
        root = self._make_root(depths)
        spans = compute_collapse(root, line_budget=4)
        visible_lines = sum(s.line_count for s in spans if s.visible)
        collapsed_count = sum(1 for s in spans if not s.visible)
        total_output = visible_lines + collapsed_count
        assert total_output <= 4
        assert collapsed_count >= 1

    def test_single_line_not_collapsed(self):
        depths = [0, 4, 0]
        root = self._make_root(depths)
        spans = compute_collapse(root, line_budget=3)
        assert all(s.visible for s in spans)

    def test_partial_expansion(self):
        """With surplus budget, collapsed regions get extra visible lines."""
        depths = [0] + [4] * 20 + [0]
        root = self._make_root(depths)
        # Budget of 10: depth-0 lines are 2, collapsed region is 1 = 3.
        # Surplus of 7 should partially expand the collapsed region.
        spans = compute_collapse(root, line_budget=10)
        total_output = sum(s.output_line_count for s in spans)
        assert total_output <= 10
        # There should be some visible lines from the collapsed region.
        visible_lines = sum(s.line_count for s in spans if s.visible)
        assert visible_lines > 2  # more than just the depth-0 lines

    def test_multiple_collapsed_regions_get_even_distribution(self):
        depths = [0] + [4] * 10 + [0] + [4] * 10 + [0]
        root = self._make_root(depths)
        # Depth-0: 3 lines. Two collapsed regions of 10 each = 2 summaries.
        # Base output = 5. Budget = 15 → surplus = 10.
        spans = compute_collapse(root, line_budget=15)
        total_output = sum(s.output_line_count for s in spans)
        assert total_output <= 15

    def test_deeply_nested(self):
        depths = [0, 2, 4, 6, 8, 6, 4, 2, 0]
        root = self._make_root(depths)
        spans = compute_collapse(root, line_budget=5)
        total_output = sum(s.output_line_count for s in spans)
        assert total_output <= 5


# ---------------------------------------------------------------------------
# Layer 3: format_outline
# ---------------------------------------------------------------------------


class TestFormatOutline:
    def test_all_visible(self):
        lines = ["alpha", "beta", "gamma"]
        spans = [OutputSpan(1, 3, visible=True)]
        result = format_outline(lines, spans, char_budget=10_000)
        assert "1| alpha" in result
        assert "2| beta" in result
        assert "3| gamma" in result
        assert "[Outline: 3 lines total." in result

    def test_collapsed_span(self):
        lines = ["a"] * 10
        spans = [
            OutputSpan(1, 3, visible=True),
            OutputSpan(4, 8, visible=False),
            OutputSpan(9, 10, visible=True),
        ]
        result = format_outline(lines, spans, char_budget=10_000)
        assert "[lines 4-8, 5 lines -- use offset/limit to read]" in result
        assert " 9| a" in result

    def test_line_number_width_consistent(self):
        lines = ["x"] * 1000
        spans = [
            OutputSpan(1, 2, visible=True),
            OutputSpan(3, 999, visible=False),
            OutputSpan(1000, 1000, visible=True),
        ]
        result = format_outline(lines, spans, char_budget=100_000)
        assert "   1| x" in result
        assert "1000| x" in result

    def test_char_budget_truncation(self):
        lines = ["x" * 100] * 50
        spans = [OutputSpan(1, 50, visible=True)]
        result = format_outline(lines, spans, char_budget=300)
        assert "Outline truncated at character limit." in result

    def test_footer_present(self):
        lines = ["a", "b"]
        spans = [OutputSpan(1, 2, visible=True)]
        result = format_outline(lines, spans, char_budget=10_000)
        assert result.endswith(
            "[Outline: 2 lines total. Use offset/limit to read specific regions.]",
        )


# ---------------------------------------------------------------------------
# Integration: outline_and_format
# ---------------------------------------------------------------------------


class TestOutlineAndFormat:
    def test_python_like(self):
        code = textwrap.dedent("""\
            class Foo:
                def bar(self):
                    x = 1
                    y = 2
                    return x + y

                def baz(self):
                    return 42

            class Qux:
                pass
        """).splitlines()
        result = outline_and_format(code, line_budget=8, char_budget=50_000)
        assert "[Outline:" in result
        # Top-level class lines should be visible.
        assert "class Foo:" in result
        assert "class Qux:" in result

    def test_cpp_like(self):
        code = textwrap.dedent("""\
            #include <string>

            enum Color {
                Red,
                Green,
                Blue,
            };

            class Widget {
            public:
                Widget();
                void draw();
            private:
                int x_;
                int y_;
            };
        """).splitlines()
        result = outline_and_format(code, line_budget=8, char_budget=50_000)
        assert "[Outline:" in result
        assert "#include <string>" in result

    def test_flat_file_degrades_to_truncation(self):
        lines = [f"line {i}" for i in range(50)]
        result = outline_and_format(lines, line_budget=10, char_budget=50_000)
        assert "[Outline:" in result
        assert "line 0" in result
        assert "line 9" in result
        assert "line 10" not in result.split("[lines")[0]

    def test_small_file_shown_fully(self):
        lines = ["a", "b", "c"]
        result = outline_and_format(lines, line_budget=10, char_budget=50_000)
        assert "1| a" in result
        assert "2| b" in result
        assert "3| c" in result
        assert "[Outline: 3 lines total." in result

    def test_budget_respected(self):
        """Output line count (excluding footer) stays within budget."""
        code = textwrap.dedent("""\
            def f1():
                a = 1
                b = 2
                c = 3
                d = 4
                e = 5
            def f2():
                a = 1
                b = 2
                c = 3
                d = 4
                e = 5
            def f3():
                a = 1
                b = 2
                c = 3
                d = 4
                e = 5
        """).splitlines()
        budget = 10
        result = outline_and_format(code, line_budget=budget, char_budget=50_000)
        content_lines = result.split("\n")
        # Last line is the footer.
        non_footer = [l for l in content_lines if not l.startswith("[Outline")]
        assert len(non_footer) <= budget


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectContentFormat:
    def test_none_path(self):
        assert detect_content_format(None) is ContentFormat.INDENTATION

    def test_python_file(self):
        assert detect_content_format("src/foo.py") is ContentFormat.INDENTATION

    def test_md_extension(self):
        assert detect_content_format("docs/README.md") is ContentFormat.MARKDOWN

    def test_markdown_extension(self):
        assert detect_content_format("notes.markdown") is ContentFormat.MARKDOWN

    def test_case_insensitive(self):
        assert detect_content_format("FILE.MD") is ContentFormat.MARKDOWN
        assert detect_content_format("FILE.Markdown") is ContentFormat.MARKDOWN

    def test_no_extension(self):
        assert detect_content_format("Makefile") is ContentFormat.INDENTATION

    def test_windows_path(self):
        assert detect_content_format("C:\\docs\\file.md") is ContentFormat.MARKDOWN

    def test_posix_path(self):
        assert detect_content_format("/home/user/file.md") is ContentFormat.MARKDOWN


# ---------------------------------------------------------------------------
# outline_and_format file_path dispatch
# ---------------------------------------------------------------------------


class TestOutlineAndFormatDispatch:
    def test_no_path_uses_indentation(self):
        code = textwrap.dedent("""\
            def foo():
                return 1
            def bar():
                return 2
        """).splitlines()
        result_no_path = outline_and_format(
            code, line_budget=10, char_budget=50_000,
        )
        result_py_path = outline_and_format(
            code, line_budget=10, char_budget=50_000, file_path="test.py",
        )
        assert result_no_path == result_py_path

    def test_md_path_uses_markdown_strategy(self):
        """Verify .md triggers a different hierarchy than indentation.

        Flat (unindented) content with headings spread through the file:
        the indentation strategy sees everything at depth 0 and falls
        back to prefix truncation, while the markdown strategy groups
        by headings and shows section titles from across the file.
        """
        parts: list[str] = ["# Document Title", ""]
        for i in range(1, 11):
            parts.append(f"## Section {i}")
            parts.append("")
            for j in range(20):
                parts.append(f"body line {j} of section {i}")
            parts.append("")
        md = parts
        result_indent = outline_and_format(
            md, line_budget=30, char_budget=50_000,
        )
        result_md = outline_and_format(
            md, line_budget=30, char_budget=50_000, file_path="doc.md",
        )
        # Indentation strategy treats everything as flat (depth 0) so
        # it shows a prefix; the markdown strategy collapses section
        # bodies and can show headings from deeper in the file.
        assert result_indent != result_md
        # The markdown outline should mention later sections that the
        # flat prefix truncation cannot reach.
        assert "## Section 1" in result_md
