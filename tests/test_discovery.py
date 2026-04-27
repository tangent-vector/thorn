"""Tests for thorn.core._discovery and the @tool decorator.

Coverage is limited to the ``@tool`` marker decorator and to
``discover_tools`` (the project-level ``.agents/thorn/*.py`` Python
tool collector).  The rest of the module's previous surface
(``find_thorn_dirs``, ``load_workspace_instructions``,
``load_agent_memory``) was retired alongside the unified
context-gathering refactor; equivalent behavior now lives in
``thorn.runtime._context_layers`` and is exercised by
``tests/test_context_layers.py``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from thorn.core._discovery import discover_tools
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
# discover_tools (integration, from .agents/thorn/)
#
# This is the only `_discovery` surface that survived the
# unified-context-gathering refactor: ``find_thorn_dirs``,
# ``load_workspace_instructions``, and ``load_agent_memory`` were
# retired in favor of the per-prompt context-gathering pipeline.  The
# bulk of the previous test coverage moved with them; what remains
# here exercises the deepest-first walker indirectly via the public
# ``discover_tools`` entry point.
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
