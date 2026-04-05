"""Tests for thorn._tools — built-in file/shell tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from thorn.core._history import DirectoryListCallNode, FileReadCallNode
from thorn.core._tools import (
    EDIT_CONTEXT_LINES,
    MAX_FIND_RESULTS,
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    MAX_READ_LINES,
    MAX_SEARCH_CHARS,
    MAX_SEARCH_MATCHES,
    OUTLINE_THRESHOLD,
    FileEdit,
    _collect_match_groups,
    _format_lines,
    ask_user,
    create_file,
    delete_file,
    edit_file,
    find_files,
    list_directory,
    move_file,
    read_file,
    run_shell,
    search_files,
    write_file,
)


# ---------------------------------------------------------------------------
# call_node_class registration on built-in tools
# ---------------------------------------------------------------------------

class TestBuiltinToolCallNodeClass:
    def test_read_file_has_file_read_class(self):
        assert getattr(read_file, "_thorn_call_node_class", None) is FileReadCallNode

    def test_list_directory_has_directory_list_class(self):
        assert getattr(list_directory, "_thorn_call_node_class", None) is DirectoryListCallNode

    def test_other_tools_have_no_class(self):
        for fn in [edit_file, create_file, delete_file, move_file, find_files, search_files]:
            assert not hasattr(fn, "_thorn_call_node_class"), (
                f"{fn.__name__} should not have _thorn_call_node_class"
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

    async def test_large_file_gets_outlined(self, tmp_path):
        """Files exceeding OUTLINE_THRESHOLD get an outline, not raw truncation."""
        n = MAX_READ_LINES + 200
        p = _make_file(tmp_path, "big.txt", n)
        result = await read_file(str(p))
        assert "[Outline:" in result
        assert f"{n} lines total" in result

    async def test_outline_bypass_with_explicit_offset(self, tmp_path):
        """Explicit offset skips outlining and uses verbatim + truncation."""
        n = MAX_READ_LINES + 200
        p = _make_file(tmp_path, "big.txt", n)
        result = await read_file(str(p), offset=1, limit=MAX_READ_LINES)
        numbered_lines = [l for l in result.split("\n") if "| line " in l]
        assert len(numbered_lines) == MAX_READ_LINES
        assert "[Truncated:" in result

    async def test_truncated_at_max_lines_with_limit(self, tmp_path):
        """With explicit limit, old truncation behavior applies at MAX_READ_LINES."""
        n = MAX_READ_LINES + 200
        p = _make_file(tmp_path, "big.txt", n)
        result = await read_file(str(p), limit=n)
        numbered_lines = [l for l in result.split("\n") if "| line " in l]
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

    async def test_file_at_outline_threshold_not_outlined(self, tmp_path):
        """Files with exactly OUTLINE_THRESHOLD lines are returned verbatim."""
        p = _make_file(tmp_path, "exact.txt", OUTLINE_THRESHOLD)
        result = await read_file(str(p))
        assert "[Outline:" not in result
        assert f"1| line 1" in result

    async def test_file_just_above_threshold_outlined(self, tmp_path):
        n = OUTLINE_THRESHOLD + 1
        p = _make_file(tmp_path, "over.txt", n)
        result = await read_file(str(p))
        assert "[Outline:" in result
        assert f"{n} lines total" in result

    async def test_outline_preserves_explicit_range_read(self, tmp_path):
        """After seeing an outline, the agent can read specific ranges."""
        n = OUTLINE_THRESHOLD + 100
        p = _make_file(tmp_path, "big.txt", n)
        result = await read_file(str(p), offset=50, limit=10)
        assert "50| line 50" in result
        assert "59| line 59" in result
        assert "[Outline:" not in result


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
# edit_file
# ---------------------------------------------------------------------------


def _write(tmp_path, name: str, content: str):
    """Helper: write a file and return its string path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestEditFile:
    async def test_single_edit(self, tmp_path):
        path = _write(tmp_path, "f.txt", "hello world\n")
        result = await edit_file(path, [
            FileEdit(old_string="hello", new_string="goodbye"),
        ])
        assert (tmp_path / "f.txt").read_text("utf-8") == "goodbye world\n"
        assert "Applied 1 edit(s)" in result
        assert "goodbye" in result

    async def test_multi_edit_sequential(self, tmp_path):
        path = _write(tmp_path, "f.txt", "aaa\nbbb\nccc\n")
        result = await edit_file(path, [
            FileEdit(old_string="aaa", new_string="AAA"),
            FileEdit(old_string="ccc", new_string="CCC"),
        ])
        assert (tmp_path / "f.txt").read_text("utf-8") == "AAA\nbbb\nCCC\n"
        assert "Applied 2 edit(s)" in result

    async def test_later_edit_sees_earlier_changes(self, tmp_path):
        path = _write(tmp_path, "f.txt", "foo bar\n")
        await edit_file(path, [
            FileEdit(old_string="foo", new_string="baz"),
            FileEdit(old_string="baz bar", new_string="done"),
        ])
        assert (tmp_path / "f.txt").read_text("utf-8") == "done\n"

    async def test_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await edit_file(str(tmp_path / "nope.txt"), [
                FileEdit(old_string="x", new_string="y"),
            ])

    async def test_no_match_raises(self, tmp_path):
        path = _write(tmp_path, "f.txt", "hello\n")
        with pytest.raises(ValueError, match="not found"):
            await edit_file(path, [
                FileEdit(old_string="zzz", new_string="aaa"),
            ])

    async def test_ambiguous_match_raises(self, tmp_path):
        path = _write(tmp_path, "f.txt", "aaa\naaa\n")
        with pytest.raises(ValueError, match="2 matches"):
            await edit_file(path, [
                FileEdit(old_string="aaa", new_string="bbb"),
            ])

    async def test_empty_old_string_raises(self, tmp_path):
        path = _write(tmp_path, "f.txt", "hello\n")
        with pytest.raises(ValueError, match="old_string must not be empty"):
            await edit_file(path, [
                FileEdit(old_string="", new_string="stuff"),
            ])

    async def test_deletion_via_empty_new_string(self, tmp_path):
        path = _write(tmp_path, "f.txt", "keep\nremove\nkeep\n")
        await edit_file(path, [
            FileEdit(old_string="remove\n", new_string=""),
        ])
        assert (tmp_path / "f.txt").read_text("utf-8") == "keep\nkeep\n"

    async def test_multiline_replacement(self, tmp_path):
        original = "line1\nline2\nline3\nline4\nline5\n"
        path = _write(tmp_path, "f.txt", original)
        await edit_file(path, [
            FileEdit(
                old_string="line2\nline3\nline4",
                new_string="NEW2\nNEW3",
            ),
        ])
        assert (tmp_path / "f.txt").read_text("utf-8") == (
            "line1\nNEW2\nNEW3\nline5\n"
        )

    async def test_no_edits_returns_unchanged(self, tmp_path):
        path = _write(tmp_path, "f.txt", "content\n")
        result = await edit_file(path, [])
        assert "unchanged" in result.lower()
        assert (tmp_path / "f.txt").read_text("utf-8") == "content\n"

    async def test_result_shows_context(self, tmp_path):
        lines = [f"line {i}" for i in range(1, 21)]
        path = _write(tmp_path, "f.txt", "\n".join(lines))
        result = await edit_file(path, [
            FileEdit(old_string="line 10", new_string="EDITED 10"),
        ])
        assert "EDITED 10" in result
        for nearby in range(
            max(1, 10 - EDIT_CONTEXT_LINES),
            min(20, 10 + EDIT_CONTEXT_LINES) + 1,
        ):
            assert f"line {nearby}" in result or "EDITED" in result

    async def test_result_collapses_distant_lines(self, tmp_path):
        lines = [f"line {i}" for i in range(1, 51)]
        path = _write(tmp_path, "f.txt", "\n".join(lines))
        result = await edit_file(path, [
            FileEdit(old_string="line 25", new_string="CHANGED"),
        ])
        assert "CHANGED" in result
        assert "[lines" in result

    async def test_error_label_includes_edit_index(self, tmp_path):
        path = _write(tmp_path, "f.txt", "abc\n")
        with pytest.raises(ValueError, match="Edit 1/2"):
            await edit_file(path, [
                FileEdit(old_string="zzz", new_string="x"),
                FileEdit(old_string="abc", new_string="y"),
            ])

    async def test_multi_edit_region_adjustment(self, tmp_path):
        """Later edit that adds lines shifts earlier recorded regions."""
        lines = "\n".join(f"line {i}" for i in range(1, 31))
        path = _write(tmp_path, "f.txt", lines)
        result = await edit_file(path, [
            FileEdit(old_string="line 25", new_string="EDIT_A"),
            FileEdit(
                old_string="line 5",
                new_string="EDIT_B_1\nEDIT_B_2\nEDIT_B_3",
            ),
        ])
        content = (tmp_path / "f.txt").read_text("utf-8")
        assert "EDIT_A" in content
        assert "EDIT_B_1" in content
        assert "Applied 2 edit(s)" in result


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------


class TestCreateFile:
    async def test_creates_new_file(self, tmp_path):
        p = tmp_path / "new.txt"
        result = await create_file(str(p), "hello\nworld\n")
        assert p.read_text("utf-8") == "hello\nworld\n"
        assert "Created" in result
        assert "hello" in result

    async def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "a" / "b" / "c.txt"
        await create_file(str(p), "nested\n")
        assert p.read_text("utf-8") == "nested\n"

    async def test_existing_file_raises(self, tmp_path):
        p = tmp_path / "exists.txt"
        p.write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError, match="already exists"):
            await create_file(str(p), "new")

    async def test_empty_content(self, tmp_path):
        p = tmp_path / "empty.txt"
        result = await create_file(str(p), "")
        assert p.read_text("utf-8") == ""
        assert "0 lines" in result

    async def test_small_file_shown_fully(self, tmp_path):
        p = tmp_path / "small.txt"
        content = "alpha\nbeta\ngamma\n"
        result = await create_file(str(p), content)
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result

    async def test_large_file_outlined(self, tmp_path):
        p = tmp_path / "big.txt"
        lines = [f"line {i}" for i in range(OUTLINE_THRESHOLD + 50)]
        content = "\n".join(lines)
        result = await create_file(str(p), content)
        assert "[Outline:" in result


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    async def test_lists_entries_with_dir_markers(self, tmp_path):
        (tmp_path / "alpha.txt").touch()
        (tmp_path / "beta.txt").touch()
        (tmp_path / "gamma").mkdir()
        result = await list_directory(str(tmp_path))
        assert isinstance(result, str)
        assert "alpha.txt" in result
        assert "beta.txt" in result
        assert "gamma/" in result

    async def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(NotADirectoryError):
            await list_directory(str(f))

    async def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = await list_directory(str(empty))
        assert result == "[empty directory]"

    async def test_recursive_listing(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "src" / "lib").mkdir()
        (tmp_path / "src" / "lib" / "util.py").touch()
        (tmp_path / "README.md").touch()
        result = await list_directory(str(tmp_path), recursive=True)
        assert "README.md" in result
        assert "src/" in result
        assert "main.py" in result
        assert "lib/" in result
        assert "util.py" in result

    async def test_recursive_respects_max_depth(self, tmp_path):
        d = tmp_path
        for name in ["a", "b", "c", "d"]:
            d = d / name
            d.mkdir()
            (d / "file.txt").touch()
        result = await list_directory(str(tmp_path), recursive=True, max_depth=2)
        assert "a/" in result
        assert "b/" in result
        # depth 3 and beyond should be excluded
        lines = result.split("\n")
        deep_entries = [l for l in lines if "d/" in l or "d/file.txt" in l]
        assert not deep_entries

    async def test_recursive_truncation(self, tmp_path):
        for i in range(MAX_LIST_ENTRIES + 50):
            (tmp_path / f"file_{i:04d}.txt").touch()
        result = await list_directory(str(tmp_path), recursive=True)
        assert f"{MAX_LIST_ENTRIES} entries shown" in result


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

    async def test_ignore_case(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("Hello World\ngoodbye world\n", encoding="utf-8")
        result = await search_files("hello", str(p), ignore_case=True)
        assert "Hello World" in result

    async def test_ignore_case_not_set(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("Hello World\ngoodbye world\n", encoding="utf-8")
        result = await search_files("hello", str(p))
        assert "No matches" in result

    async def test_glob_matches_relative_path(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "code.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").write_text("needle\n", encoding="utf-8")
        result = await search_files(
            "needle", str(tmp_path), glob="src/*.py",
        )
        assert "code.py" in result
        assert "test.py" not in result


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------

class TestFindFiles:
    async def test_basic_glob(self, tmp_path):
        (tmp_path / "foo.py").touch()
        (tmp_path / "bar.py").touch()
        (tmp_path / "baz.txt").touch()
        result = await find_files("*.py", str(tmp_path))
        assert "foo.py" in result
        assert "bar.py" in result
        assert "baz.txt" not in result

    async def test_recursive_glob(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "src" / "lib").mkdir()
        (tmp_path / "src" / "lib" / "util.py").touch()
        result = await find_files("*.py", str(tmp_path))
        assert "main.py" in result
        assert "util.py" in result

    async def test_type_filter_file(self, tmp_path):
        (tmp_path / "file.py").touch()
        (tmp_path / "package").mkdir()
        result = await find_files("*", str(tmp_path), type="file")
        assert "file.py" in result
        assert "package" not in result

    async def test_type_filter_directory(self, tmp_path):
        (tmp_path / "file.py").touch()
        (tmp_path / "package").mkdir()
        result = await find_files("*", str(tmp_path), type="directory")
        assert "package/" in result
        assert "file.py" not in result

    async def test_no_matches(self, tmp_path):
        (tmp_path / "file.txt").touch()
        result = await find_files("*.py", str(tmp_path))
        assert "No matches" in result

    async def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(NotADirectoryError):
            await find_files("*", str(f))

    async def test_cap_at_max_results(self, tmp_path):
        for i in range(MAX_FIND_RESULTS + 50):
            (tmp_path / f"file_{i:04d}.txt").touch()
        result = await find_files("*.txt", str(tmp_path))
        assert f"Results capped at {MAX_FIND_RESULTS}" in result

    async def test_directory_entries_have_trailing_slash(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        result = await find_files("mydir", str(tmp_path))
        assert "mydir/" in result


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------

class TestDeleteFile:
    async def test_deletes_existing_file(self, tmp_path):
        p = tmp_path / "doomed.txt"
        p.write_text("bye", encoding="utf-8")
        result = await delete_file(str(p))
        assert "Deleted" in result
        assert not p.exists()

    async def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await delete_file(str(tmp_path / "nope.txt"))

    async def test_directory_raises(self, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        with pytest.raises(IsADirectoryError):
            await delete_file(str(d))


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------

class TestMoveFile:
    async def test_moves_file(self, tmp_path):
        src = tmp_path / "old.txt"
        src.write_text("content", encoding="utf-8")
        dst = tmp_path / "new.txt"
        result = await move_file(str(src), str(dst))
        assert "Moved" in result
        assert not src.exists()
        assert dst.read_text("utf-8") == "content"

    async def test_creates_parent_directories(self, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("data", encoding="utf-8")
        dst = tmp_path / "a" / "b" / "file.txt"
        await move_file(str(src), str(dst))
        assert dst.read_text("utf-8") == "data"

    async def test_source_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await move_file(
                str(tmp_path / "nope.txt"),
                str(tmp_path / "dest.txt"),
            )

    async def test_destination_exists_raises(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("a", encoding="utf-8")
        dst = tmp_path / "dst.txt"
        dst.write_text("b", encoding="utf-8")
        with pytest.raises(FileExistsError, match="already exists"):
            await move_file(str(src), str(dst))


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------

class TestRunShell:
    async def test_basic_command(self, tmp_path):
        result = await run_shell("echo hello")
        assert "hello" in result

    async def test_exit_code_reported(self):
        exit_cmd = "exit /b 42" if os.name == "nt" else "exit 42"
        result = await run_shell(exit_cmd)
        assert "[exit code 42]" in result

    async def test_working_directory(self, tmp_path):
        cwd_cmd = "cd" if os.name == "nt" else "pwd"
        result = await run_shell(cwd_cmd, working_directory=str(tmp_path))
        assert tmp_path.name in result

    async def test_timeout_kills_process(self):
        long_cmd = (
            "ping -n 60 127.0.0.1 > nul" if os.name == "nt"
            else "sleep 60"
        )
        result = await run_shell(long_cmd, timeout=0.1)
        assert "timed out" in result

    async def test_output_truncation(self):
        result = await run_shell(
            f'python -c "print(\'x\' * {MAX_READ_CHARS + 1000})"',
        )
        assert "[output truncated:" in result


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------

class TestAskUser:
    async def test_raises_without_handler(self):
        """ask_user raises RuntimeError when no handler is configured."""
        from thorn.core._context import ExecutionContext, set_context, reset_context
        from thorn.core._provider import MockProvider

        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            with pytest.raises(RuntimeError, match="not available"):
                await ask_user("anything")
        finally:
            reset_context(token)

    async def test_delegates_to_handler(self):
        """ask_user delegates to the configured handler."""
        from thorn.core._context import ExecutionContext, set_context, reset_context
        from thorn.core._provider import MockProvider

        async def fake_handler(question: str) -> str:
            return f"answer to: {question}"

        ctx = ExecutionContext(
            provider=MockProvider(),
            ask_user_handler=fake_handler,
        )
        token = set_context(ctx)
        try:
            result = await ask_user("what?")
            assert result == "answer to: what?"
        finally:
            reset_context(token)
