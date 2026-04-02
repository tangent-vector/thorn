"""Tests for thorn._tools — built-in file/shell tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from thorn._tools import (
    MAX_READ_CHARS,
    MAX_READ_LINES,
    MAX_SEARCH_CHARS,
    MAX_SEARCH_MATCHES,
    _collect_match_groups,
    _format_lines,
    list_directory,
    read_file,
    search_files,
    write_file,
)


# ---------------------------------------------------------------------------
# _format_lines
# ---------------------------------------------------------------------------

class TestFormatLines:
    def test_single_line(self):
        assert _format_lines(["hello"], 1) == "1| hello"

    def test_multiple_lines(self):
        result = _format_lines(["a", "b", "c"], 1)
        assert result == "1| a\n2| b\n3| c"

    def test_pads_to_widest_number(self):
        result = _format_lines(["first", "last"], 9)
        assert result == " 9| first\n10| last"

    def test_high_offset(self):
        result = _format_lines(["x"], 1000)
        assert result == "1000| x"

    def test_explicit_width(self):
        result = _format_lines(["a", "b"], 1, width=4)
        assert result == "   1| a\n   2| b"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def _make_file(tmp_path: Path, name: str, num_lines: int) -> Path:
    """Write a file with predictable line content: ``line 1``, ``line 2``, ..."""
    p = tmp_path / name
    p.write_text(
        "\n".join(f"line {i}" for i in range(1, num_lines + 1)),
        encoding="utf-8",
    )
    return p


class TestReadFile:
    async def test_reads_existing_file(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("world", encoding="utf-8")
        result = await read_file(str(p))
        assert "world" in result
        assert result.strip() == "1| world"

    async def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await read_file(str(tmp_path / "nope.txt"))

    async def test_line_numbers_included(self, tmp_path):
        p = _make_file(tmp_path, "numbered.txt", 3)
        result = await read_file(str(p))
        assert "1| line 1" in result
        assert "2| line 2" in result
        assert "3| line 3" in result

    async def test_offset(self, tmp_path):
        p = _make_file(tmp_path, "off.txt", 10)
        result = await read_file(str(p), offset=8)
        assert " 8| line 8" in result
        assert "10| line 10" in result
        assert "line 7" not in result

    async def test_limit(self, tmp_path):
        p = _make_file(tmp_path, "lim.txt", 10)
        result = await read_file(str(p), limit=3)
        assert "1| line 1" in result
        assert "3| line 3" in result
        assert "line 4" not in result
        assert "[Truncated:" in result

    async def test_offset_and_limit(self, tmp_path):
        p = _make_file(tmp_path, "both.txt", 20)
        result = await read_file(str(p), offset=5, limit=3)
        assert "5| line 5" in result
        assert "7| line 7" in result
        assert "line 4" not in result
        assert "line 8" not in result

    async def test_offset_past_end(self, tmp_path):
        p = _make_file(tmp_path, "past.txt", 5)
        result = await read_file(str(p), offset=100)
        assert "No content" in result
        assert "5 lines" in result

    async def test_truncated_at_max_lines(self, tmp_path):
        n = MAX_READ_LINES + 200
        p = _make_file(tmp_path, "big.txt", n)
        result = await read_file(str(p))
        lines = result.split("\n")
        numbered_lines = [l for l in lines if "| line " in l]
        assert len(numbered_lines) == MAX_READ_LINES
        assert "[Truncated:" in result
        assert f"of {n} total" in result

    async def test_limit_larger_than_max_still_capped(self, tmp_path):
        n = MAX_READ_LINES + 200
        p = _make_file(tmp_path, "overcap.txt", n)
        result = await read_file(str(p), limit=n)
        numbered_lines = [l for l in result.split("\n") if "| line " in l]
        assert len(numbered_lines) == MAX_READ_LINES
        assert "[Truncated:" in result

    async def test_limit_within_cap_no_truncation(self, tmp_path):
        p = _make_file(tmp_path, "small.txt", 10)
        result = await read_file(str(p), limit=10)
        assert "[Truncated:" not in result

    async def test_char_limit_truncates(self, tmp_path):
        long_line = "x" * (MAX_READ_CHARS + 100)
        p = tmp_path / "longline.txt"
        p.write_text(f"short\n{long_line}\nafter", encoding="utf-8")
        result = await read_file(str(p))
        assert "[Truncated:" in result
        assert "after" not in result

    async def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        result = await read_file(str(p))
        assert "No content" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    async def test_writes_content(self, tmp_path):
        p = tmp_path / "out.txt"
        result = await write_file(str(p), "data")
        assert p.read_text(encoding="utf-8") == "data"
        assert "4" in result  # mentions byte count

    async def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "a" / "b" / "c.txt"
        await write_file(str(p), "nested")
        assert p.read_text(encoding="utf-8") == "nested"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    async def test_lists_entries(self, tmp_path):
        (tmp_path / "alpha.txt").touch()
        (tmp_path / "beta.txt").touch()
        (tmp_path / "gamma").mkdir()
        result = await list_directory(str(tmp_path))
        assert result == ["alpha.txt", "beta.txt", "gamma"]

    async def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(NotADirectoryError):
            await list_directory(str(f))


# ---------------------------------------------------------------------------
# _collect_match_groups
# ---------------------------------------------------------------------------

class TestCollectMatchGroups:
    def test_single_match_no_context(self):
        lines = ["a", "b", "c", "d"]
        groups = _collect_match_groups(lines, [1], context_lines=0)
        assert groups == [(2, ["b"])]

    def test_single_match_with_context(self):
        lines = ["a", "b", "c", "d", "e"]
        groups = _collect_match_groups(lines, [2], context_lines=1)
        assert groups == [(2, ["b", "c", "d"])]

    def test_context_clamped_at_boundaries(self):
        lines = ["a", "b", "c"]
        groups = _collect_match_groups(lines, [0], context_lines=5)
        assert groups == [(1, ["a", "b", "c"])]

    def test_separate_groups(self):
        lines = [f"line{i}" for i in range(10)]
        groups = _collect_match_groups(lines, [1, 8], context_lines=0)
        assert len(groups) == 2
        assert groups[0] == (2, ["line1"])
        assert groups[1] == (9, ["line8"])

    def test_overlapping_groups_merged(self):
        lines = [f"L{i}" for i in range(10)]
        groups = _collect_match_groups(lines, [2, 4], context_lines=1)
        assert len(groups) == 1
        assert groups[0] == (2, ["L1", "L2", "L3", "L4", "L5"])

    def test_adjacent_groups_merged(self):
        lines = ["a", "b", "c", "d", "e"]
        # Matches at 1 and 3: with context=0, ranges are [1,1] and [3,3]
        # These are not adjacent (gap at index 2), so they stay separate.
        groups = _collect_match_groups(lines, [1, 3], context_lines=0)
        assert len(groups) == 2
        # With context=1: ranges [0,2] and [2,4] overlap at index 2.
        groups = _collect_match_groups(lines, [1, 3], context_lines=1)
        assert len(groups) == 1
        assert groups[0] == (1, ["a", "b", "c", "d", "e"])

    def test_empty_match_indices(self):
        assert _collect_match_groups(["a", "b"], [], context_lines=0) == []


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

class TestSearchFiles:
    async def test_literal_match_single_file(self, tmp_path):
        p = tmp_path / "hello.py"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        result = await search_files("beta", str(p))
        assert str(p) in result
        assert "2| beta" in result

    async def test_no_matches_single_file(self, tmp_path):
        p = tmp_path / "hello.py"
        p.write_text("alpha\nbeta\n", encoding="utf-8")
        result = await search_files("zzz", str(p))
        assert "No matches" in result

    async def test_regex_mode(self, tmp_path):
        p = tmp_path / "code.py"
        p.write_text("foo123\nbar456\nfoo789\n", encoding="utf-8")
        result = await search_files(
            r"foo\d+", str(p), use_regex=True,
        )
        assert "foo123" in result
        assert "foo789" in result
        assert "bar456" not in result

    async def test_literal_escapes_regex_chars(self, tmp_path):
        p = tmp_path / "special.txt"
        p.write_text("price is $100\nprice is 100\n", encoding="utf-8")
        result = await search_files("$100", str(p))
        assert "price is $100" in result
        # "$100" as regex would match "100" at end of line — verify it doesn't.
        lines = result.strip().split("\n")
        match_lines = [l for l in lines if "| " in l]
        assert len(match_lines) == 1

    async def test_recursive_directory_search(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "two.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text("nothing\n", encoding="utf-8")
        result = await search_files("needle", str(tmp_path))
        assert "one.txt" in result
        assert "two.txt" in result
        assert "c.txt" not in result

    async def test_glob_filter(self, tmp_path):
        (tmp_path / "code.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "data.txt").write_text("needle\n", encoding="utf-8")
        result = await search_files("needle", str(tmp_path), glob="*.py")
        assert "code.py" in result
        assert "data.txt" not in result

    async def test_context_lines(self, tmp_path):
        p = tmp_path / "ctx.txt"
        p.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        result = await search_files("c", str(p), context_lines=1)
        assert "b" in result
        assert "c" in result
        assert "d" in result
        assert "| a" not in result
        assert "| e" not in result

    async def test_context_merging(self, tmp_path):
        p = tmp_path / "merge.txt"
        lines = [f"line {i}" for i in range(1, 11)]
        p.write_text("\n".join(lines), encoding="utf-8")
        result = await search_files(
            r"line [35]", str(p), use_regex=True, context_lines=1,
        )
        assert "--" not in result
        assert "line 2" in result
        assert "line 3" in result
        assert "line 4" in result
        assert "line 5" in result
        assert "line 6" in result

    async def test_non_contiguous_groups_separated(self, tmp_path):
        p = tmp_path / "sep.txt"
        lines = [f"line {i}" for i in range(1, 11)]
        p.write_text("\n".join(lines), encoding="utf-8")
        result = await search_files(
            r"line [19]", str(p), use_regex=True, context_lines=0,
        )
        assert "--" in result

    async def test_truncation_at_max_matches(self, tmp_path):
        n_files = MAX_SEARCH_MATCHES + 50
        for i in range(n_files):
            (tmp_path / f"file_{i:04d}.txt").write_text(
                f"needle in file {i}\n", encoding="utf-8",
            )
        result = await search_files("needle", str(tmp_path))
        assert "[Truncated:" in result
        assert f"of {n_files}" in result

    async def test_truncation_at_max_chars(self, tmp_path):
        long_line = "needle " + "x" * 10_000
        for i in range(20):
            (tmp_path / f"big_{i:02d}.txt").write_text(
                long_line + "\n", encoding="utf-8",
            )
        result = await search_files("needle", str(tmp_path))
        assert "[Truncated:" in result

    async def test_no_matches_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
        result = await search_files("zzz", str(tmp_path))
        assert "No matches" in result

    async def test_binary_file_skipped(self, tmp_path):
        binary = tmp_path / "binary.bin"
        binary.write_bytes(b"needle\xff\xfe invalid utf-8")
        txt = tmp_path / "text.txt"
        txt.write_text("needle\n", encoding="utf-8")
        result = await search_files("needle", str(tmp_path))
        assert "text.txt" in result
        assert "binary.bin" not in result

    async def test_path_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await search_files("needle", str(tmp_path / "nonexistent"))

    async def test_invalid_regex_raises(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("data\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid regex"):
            await search_files("[unclosed", str(p), use_regex=True)
