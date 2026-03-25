"""Tests for thorn._tools — built-in file/shell tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from thorn._tools import list_directory, read_file, write_file


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    async def test_reads_existing_file(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("world", encoding="utf-8")
        result = await read_file(str(p))
        assert result == "world"

    async def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await read_file(str(tmp_path / "nope.txt"))


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
