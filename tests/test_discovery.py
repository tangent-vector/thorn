"""Tests for thorn.core._discovery and the @tool decorator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from thorn.core._discovery import (
    discover_tools,
    find_agents_thorn_dirs,
    find_thorn_dirs,
    load_agent_memory,
    load_workspace_instructions,
)
from thorn.core._func import tool


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
# find_thorn_dirs (agency state)
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
        for d in result:
            assert not str(d).startswith(str(tmp_path))

    def test_ignores_thorn_file_not_dir(self, tmp_path: Path):
        (tmp_path / ".thorn").write_text("not a dir")
        result = find_thorn_dirs(start=tmp_path)
        for d in result:
            assert not str(d).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# find_agents_thorn_dirs (project tool definitions)
# ---------------------------------------------------------------------------

class TestFindAgentsThornDirs:
    def test_finds_agents_thorn_dir(self, tmp_path: Path):
        agents_thorn = tmp_path / ".agents" / "thorn"
        agents_thorn.mkdir(parents=True)
        (agents_thorn / "tools.py").write_text("x = 1\n")

        result = find_agents_thorn_dirs(start=tmp_path)
        assert agents_thorn in result

    def test_ignores_empty_thorn_dir(self, tmp_path: Path):
        agents_thorn = tmp_path / ".agents" / "thorn"
        agents_thorn.mkdir(parents=True)

        result = find_agents_thorn_dirs(start=tmp_path)
        assert agents_thorn not in result

    def test_finds_in_ancestor(self, tmp_path: Path):
        agents_thorn = tmp_path / ".agents" / "thorn"
        agents_thorn.mkdir(parents=True)
        (agents_thorn / "tools.py").write_text("x = 1\n")
        child = tmp_path / "a" / "b" / "c"
        child.mkdir(parents=True)

        result = find_agents_thorn_dirs(start=child)
        assert agents_thorn in result

    def test_deepest_first_ordering(self, tmp_path: Path):
        parent_agents = tmp_path / ".agents" / "thorn"
        parent_agents.mkdir(parents=True)
        (parent_agents / "a.py").write_text("x = 1\n")

        child = tmp_path / "project"
        child.mkdir()
        child_agents = child / ".agents" / "thorn"
        child_agents.mkdir(parents=True)
        (child_agents / "b.py").write_text("x = 2\n")

        result = find_agents_thorn_dirs(start=child)
        parent_idx = result.index(parent_agents)
        child_idx = result.index(child_agents)
        assert child_idx < parent_idx

    def test_ignores_non_directory(self, tmp_path: Path):
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "thorn").write_text("not a directory")

        result = find_agents_thorn_dirs(start=tmp_path)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# load_workspace_instructions
# ---------------------------------------------------------------------------

class TestLoadWorkspaceInstructions:
    def test_returns_content_when_file_exists(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("Follow these rules.", encoding="utf-8")
        assert load_workspace_instructions(tmp_path) == "Follow these rules."

    def test_returns_none_when_absent(self, tmp_path: Path):
        assert load_workspace_instructions(tmp_path) is None

    def test_returns_none_when_agents_md_is_directory(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").mkdir()
        assert load_workspace_instructions(tmp_path) is None

    def test_preserves_multiline_content(self, tmp_path: Path):
        content = "# Rules\n\n- Be concise\n- Use types\n"
        (tmp_path / "AGENTS.md").write_text(content, encoding="utf-8")
        assert load_workspace_instructions(tmp_path) == content


# ---------------------------------------------------------------------------
# load_agent_memory
# ---------------------------------------------------------------------------

class TestLoadAgentMemory:
    def test_returns_content_when_file_exists(self, tmp_path: Path):
        memory_md = tmp_path / "MEMORY.md"
        memory_md.write_text("The repository URL is https://example.com/repo.git", encoding="utf-8")
        assert load_agent_memory(tmp_path) == "The repository URL is https://example.com/repo.git"

    def test_returns_none_when_absent(self, tmp_path: Path):
        assert load_agent_memory(tmp_path) is None

    def test_returns_none_when_memory_md_is_directory(self, tmp_path: Path):
        (tmp_path / "MEMORY.md").mkdir()
        assert load_agent_memory(tmp_path) is None

    def test_preserves_multiline_content(self, tmp_path: Path):
        content = "# Agent Memory\n\n- Project URL: https://example.com\n- Default branch: main\n"
        (tmp_path / "MEMORY.md").write_text(content, encoding="utf-8")
        assert load_agent_memory(tmp_path) == content


# ---------------------------------------------------------------------------
# discover_tools (integration, from .agents/thorn/)
# ---------------------------------------------------------------------------

class TestDiscoverTools:
    def test_discovers_from_agents_thorn_dir(self, tmp_path: Path):
        tool_dir = tmp_path / ".agents" / "thorn"
        tool_dir.mkdir(parents=True)
        (tool_dir / "tools.py").write_text(textwrap.dedent("""\
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
        parent_tools = tmp_path / ".agents" / "thorn"
        parent_tools.mkdir(parents=True)
        child = tmp_path / "sub"
        child.mkdir()
        child_tools = child / ".agents" / "thorn"
        child_tools.mkdir(parents=True)

        for d in [parent_tools, child_tools]:
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
        tool_dir = tmp_path / ".agents" / "thorn"
        tool_dir.mkdir(parents=True)

        (tool_dir / "a.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def alpha() -> str:
                \"\"\"Alpha.\"\"\"
                return "a"
        """))
        (tool_dir / "b.py").write_text(textwrap.dedent("""\
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
        tool_dir = tmp_path / ".agents" / "thorn"
        tool_dir.mkdir(parents=True)

        (tool_dir / "good.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def works() -> str:
                \"\"\"Works.\"\"\"
                return "yes"
        """))
        (tool_dir / "broken.py").write_text("this is not valid python {{{{")

        result = discover_tools(start=tmp_path)
        names = [fn.__name__ for fn in result]
        assert "works" in names

    def test_empty_agents_thorn_dir_yields_nothing(self, tmp_path: Path):
        tool_dir = tmp_path / ".agents" / "thorn"
        tool_dir.mkdir(parents=True)
        result = discover_tools(start=tmp_path)
        for fn in result:
            source = getattr(fn, "__module__", "")
            assert "thorn_user" not in source or tmp_path.name not in source

    def test_sibling_relative_imports(self, tmp_path: Path):
        tool_dir = tmp_path / ".agents" / "thorn"
        tool_dir.mkdir(parents=True)

        (tool_dir / "helpers.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def helper_add(a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b
        """))
        (tool_dir / "main_tools.py").write_text(textwrap.dedent("""\
            from thorn import tool
            from .helpers import helper_add

            @tool
            def add_and_double(a: int, b: int) -> int:
                \"\"\"Add two numbers and double the result.\"\"\"
                return helper_add(a, b) * 2
        """))

        result = discover_tools(start=tmp_path)
        by_name = {fn.__name__: fn for fn in result}
        assert "helper_add" in by_name
        assert "add_and_double" in by_name
        assert by_name["add_and_double"](3, 4) == 14

    def test_does_not_discover_from_dot_thorn(self, tmp_path: Path):
        """Tool discovery no longer scans .thorn/ directories."""
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "tools.py").write_text(textwrap.dedent("""\
            from thorn import tool

            @tool
            def old_tool() -> str:
                \"\"\"Should not be found.\"\"\"
                return "nope"
        """))

        result = discover_tools(start=tmp_path)
        names = [fn.__name__ for fn in result]
        assert "old_tool" not in names
