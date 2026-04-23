"""Tests for the per-tool venue annotations introduced in Phase A.

Locks in two contracts:

* Built-in tools carry a ``_thorn_venue`` attribute that drives where
  the brain dispatches them; ``ask_user`` and the inbox tools stay
  in-process, everything else flips to ``SANDBOX``.
* :func:`_prepare_tools` rejects bare callables that aren't on the
  known-built-in list, while still accepting them when wrapped via
  :func:`wrap_function` (the explicit opt-in route).
"""

from __future__ import annotations

import pytest

from thorn.core._executor import ToolVenue
from thorn.core._func import _prepare_tools, wrap_function
from thorn.core._journal import read_journal, write_journal
from thorn.core._tools import (
    ALL_BUILTIN_TOOLS,
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
from thorn.runtime._inbox_tools import (
    list_inbox_items,
    read_inbox_item,
    update_inbox_item,
)


SANDBOX_TOOLS = [
    read_file,
    write_file,
    edit_file,
    create_file,
    delete_file,
    move_file,
    list_directory,
    find_files,
    search_files,
    run_shell,
    write_journal,
    read_journal,
]


IN_PROCESS_TOOLS = [
    ask_user,
    list_inbox_items,
    read_inbox_item,
    update_inbox_item,
]


class TestVenueAnnotations:
    @pytest.mark.parametrize("fn", SANDBOX_TOOLS)
    def test_sandbox_tools_marked_sandbox(self, fn):
        assert getattr(fn, "_thorn_venue", None) is ToolVenue.SANDBOX, fn.__name__

    @pytest.mark.parametrize("fn", IN_PROCESS_TOOLS)
    def test_in_process_tools_marked_in_process(self, fn):
        assert getattr(fn, "_thorn_venue", None) is ToolVenue.IN_PROCESS, fn.__name__

    def test_wrap_function_propagates_venue(self):
        wrapped = wrap_function(read_file)
        assert wrapped.venue is ToolVenue.SANDBOX

        wrapped_inbox = wrap_function(list_inbox_items)
        assert wrapped_inbox.venue is ToolVenue.IN_PROCESS

    def test_wrap_function_unannotated_defaults_to_sandbox(self):
        async def custom() -> str:
            """Custom."""
            return "x"

        wrapped = wrap_function(custom)
        assert wrapped.venue is ToolVenue.SANDBOX


class TestPrepareToolsAllowList:
    def test_known_builtins_accepted_bare(self):
        result = _prepare_tools(ALL_BUILTIN_TOOLS)
        assert {w.schema["function"]["name"] for w in result} == {
            fn.__name__ for fn in ALL_BUILTIN_TOOLS
        }

    def test_run_shell_accepted_bare(self):
        result = _prepare_tools([run_shell])
        assert result[0].schema["function"]["name"] == "run_shell"

    def test_unknown_callable_rejected(self):
        async def custom() -> str:
            """Custom."""
            return "x"

        with pytest.raises(TypeError, match="not a registered Thorn tool"):
            _prepare_tools([custom])

    def test_unknown_callable_accepted_via_wrap_function(self):
        async def custom() -> str:
            """Custom."""
            return "x"

        result = _prepare_tools([wrap_function(custom)])
        assert result[0].schema["function"]["name"] == "custom"

    def test_known_callable_in_nested_iterable_accepted(self):
        result = _prepare_tools([[read_file, [list_directory]]])
        names = {w.schema["function"]["name"] for w in result}
        assert names == {"read_file", "list_directory"}


class TestForgeAndGitVenues:
    def test_git_tools_default_to_sandbox(self):
        from thorn.tools.git import GIT_TOOLS

        for fn in GIT_TOOLS:
            wrapped = wrap_function(fn)
            assert wrapped.venue is ToolVenue.SANDBOX, getattr(fn, "__name__", fn)

    def test_forge_tools_default_to_sandbox(self):
        from thorn.tools.forge import FORGE_TOOLS

        for fn in FORGE_TOOLS:
            wrapped = wrap_function(fn)
            assert wrapped.venue is ToolVenue.SANDBOX, getattr(fn, "__name__", fn)
