"""Tests for thorn._discovery and the @tool decorator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from thorn._discovery import discover_tools, find_thorn_dirs, load_module_tools
from thorn._func import tool


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

class TestToolDecorator:
    def test_marks_function(self):
        @tool
        def my_fn(x: int) -> int:
            """Double x."""
            return x * 2

        assert getattr(my_fn, "_thorn_tool", False) is True

    def test_preserves_behavior(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(3, 4) == 7

    def test_preserves_metadata(self):
        @tool
        def named_fn(x: str) -> str:
            """A docstring."""
            return x

        assert named_fn.__name__ == "named_fn"
        assert named_fn.__doc__ == "A docstring."

    async def test_works_with_async(self):
        @tool
        async def async_fn(x: int) -> int:
            """Async double."""
            return x * 2

        assert getattr(async_fn, "_thorn_tool", False) is True
        assert await async_fn(5) == 10


# ---------------------------------------------------------------------------
# find_thorn_dirs
# ---------------------------------------------------------------------------

class TestFindThornDirs:
    def test_finds_thorn_dir_in_start(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        result = find_thorn_dirs(start=tmp_path)
        assert thorn_dir in result

    def test_finds_thorn_dir_in_ancestor(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        child = tmp_path / "a" / "b" / "c"
        child.mkdir(parents=True)

        result = find_thorn_dirs(start=child)
        assert thorn_dir in result

    def test_deepest_first_ordering(self, tmp_path: Path):
        parent_thorn = tmp_path / ".thorn"
        parent_thorn.mkdir()
        child = tmp_path / "project"
        child.mkdir()
        child_thorn = child / ".thorn"
        child_thorn.mkdir()

        result = find_thorn_dirs(start=child)
        parent_idx = result.index(parent_thorn)
        child_idx = result.index(child_thorn)
        assert child_idx < parent_idx

    def test_no_thorn_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        result = find_thorn_dirs(start=deep)
        # May include ~/.thorn if it exists, but no dirs from tmp_path
        for d in result:
            assert not str(d).startswith(str(tmp_path))

    def test_ignores_thorn_file_not_dir(self, tmp_path: Path):
        (tmp_path / ".thorn").write_text("not a dir")
        result = find_thorn_dirs(start=tmp_path)
        for d in result:
            assert not str(d).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# load_module_tools
# ---------------------------------------------------------------------------

class TestLoadModuleTools:
    def test_loads_tool_decorated_function(self, tmp_path: Path):
        py = tmp_path / "my_tools.py"
        py.write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def greet(name: str) -> str:
                \"\"\"Say hi.\"\"\"
                return f"hello {name}"
        """))
        result = load_module_tools(py)
        assert len(result) == 1
        assert result[0].__name__ == "greet"
        assert result[0]("world") == "hello world"

    def test_loads_skill_decorated_function(self, tmp_path: Path):
        py = tmp_path / "my_skills.py"
        py.write_text(textwrap.dedent("""\
            from thorn import skill

            @skill
            async def summarize(text: str) -> str:
                \"\"\"Summarize: {text}\"\"\"
        """))
        result = load_module_tools(py)
        assert len(result) == 1
        assert result[0].__name__ == "summarize"
        assert getattr(result[0], "_thorn_skill", False) is True

    def test_ignores_undecorated_functions(self, tmp_path: Path):
        py = tmp_path / "mixed.py"
        py.write_text(textwrap.dedent("""\
            from thorn import tool

            def helper():
                return 42

            @tool
            def exposed(x: int) -> int:
                \"\"\"Do something.\"\"\"
                return helper() + x
        """))
        result = load_module_tools(py)
        assert len(result) == 1
        assert result[0].__name__ == "exposed"

    def test_ignores_private_functions(self, tmp_path: Path):
        py = tmp_path / "private.py"
        py.write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def _internal(x: int) -> int:
                \"\"\"Internal.\"\"\"
                return x
        """))
        result = load_module_tools(py)
        assert len(result) == 0

    def test_syntax_error_returns_empty(self, tmp_path: Path):
        py = tmp_path / "broken.py"
        py.write_text("def oops(\n")
        result = load_module_tools(py)
        assert result == []

    def test_import_error_returns_empty(self, tmp_path: Path):
        py = tmp_path / "bad_import.py"
        py.write_text("import nonexistent_module_xyzzy_12345\n")
        result = load_module_tools(py)
        assert result == []


# ---------------------------------------------------------------------------
# discover_tools (integration)
# ---------------------------------------------------------------------------

class TestDiscoverTools:
    def test_discovers_from_thorn_dir(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        py = thorn_dir / "tools.py"
        py.write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def ping() -> str:
                \"\"\"Ping.\"\"\"
                return "pong"
        """))

        result = discover_tools(start=tmp_path)
        names = [fn.__name__ for fn in result]
        assert "ping" in names

    def test_deduplicates_by_name(self, tmp_path: Path):
        parent_thorn = tmp_path / ".thorn"
        parent_thorn.mkdir()
        child = tmp_path / "sub"
        child.mkdir()
        child_thorn = child / ".thorn"
        child_thorn.mkdir()

        for d in [parent_thorn, child_thorn]:
            (d / "tools.py").write_text(textwrap.dedent("""\
                from thorn import tool

                @tool
                def ping() -> str:
                    \"\"\"Ping.\"\"\"
                    return "pong"
            """))

        result = discover_tools(start=child)
        ping_fns = [fn for fn in result if fn.__name__ == "ping"]
        assert len(ping_fns) == 1

    def test_multiple_files(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()

        (thorn_dir / "a.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def alpha() -> str:
                \"\"\"Alpha.\"\"\"
                return "a"
        """))
        (thorn_dir / "b.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def beta() -> str:
                \"\"\"Beta.\"\"\"
                return "b"
        """))

        result = discover_tools(start=tmp_path)
        names = {fn.__name__ for fn in result}
        assert "alpha" in names
        assert "beta" in names

    def test_skips_broken_files_gracefully(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()

        (thorn_dir / "good.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def works() -> str:
                \"\"\"Works.\"\"\"
                return "yes"
        """))
        (thorn_dir / "broken.py").write_text("this is not valid python {{{{")

        result = discover_tools(start=tmp_path)
        names = [fn.__name__ for fn in result]
        assert "works" in names

    def test_empty_thorn_dir(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        result = discover_tools(start=tmp_path)
        # Only things from ~/.thorn might show up, nothing from tmp_path
        for fn in result:
            source = getattr(fn, "__module__", "")
            assert "thorn_user" not in source or tmp_path.name not in source
