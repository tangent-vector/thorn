"""Tests for per-tool venue annotations and the catalog parity contract.

The brain (``thorn.core._func._known_builtin_tools``) and the toolhost
daemon (``thorn.toolhost._server.build_default_registry``) both read
the canonical tool lists from :mod:`thorn.tools._catalog`.  Drift
between the two views is what allowed the original bug -- forge / git
tools tagged ``SANDBOX`` but never registered with the daemon -- to
hide silently until first invocation in production.

This module locks in three contracts:

* Every tool in :data:`thorn.tools._catalog.SANDBOXED_TOOLS` carries
  ``_thorn_venue == ToolVenue.SANDBOX`` (and similarly for
  ``IN_PROCESS_TOOLS``).
* ``wrap_function`` and ``@tool`` both refuse to assign a venue
  silently -- there is no default, the caller must always pick.
* The brain's allowlist (the union of both catalog lists), the
  daemon's registry (the SANDBOXED slice), and the catalog itself
  agree on which tools exist where.
"""

from __future__ import annotations

import pytest

from thorn.core._executor import ToolVenue
from thorn.core._func import _prepare_tools, tool, wrap_function
from thorn.tools._catalog import (
    ALL_BUILTIN_TOOL_FUNCTIONS,
    IN_PROCESS_TOOLS,
    SANDBOXED_TOOLS,
)

# ---------------------------------------------------------------------------
# Per-tool venue annotations
# ---------------------------------------------------------------------------


class TestVenueAnnotations:
    @pytest.mark.parametrize("fn", SANDBOXED_TOOLS)
    def test_sandboxed_tools_marked_sandbox(self, fn):
        assert getattr(fn, "_thorn_venue", None) is ToolVenue.SANDBOX, (
            fn.__name__
        )

    @pytest.mark.parametrize("fn", IN_PROCESS_TOOLS)
    def test_in_process_tools_marked_in_process(self, fn):
        assert getattr(fn, "_thorn_venue", None) is ToolVenue.IN_PROCESS, (
            fn.__name__
        )

    def test_wrap_function_propagates_venue_from_decoration(self):
        # When the function was decorated with @tool(venue=...), the
        # venue propagates through wrap_function without a redundant
        # keyword argument.
        from thorn.core._tools import read_file
        from thorn.runtime._inbox_tools import list_inbox_items

        assert wrap_function(read_file).venue is ToolVenue.SANDBOX
        assert wrap_function(list_inbox_items).venue is ToolVenue.IN_PROCESS

    def test_wrap_function_explicit_venue_override(self):
        # Direct callers may pass venue= explicitly.  Useful for tests
        # and for the (rare) case of wrapping a third-party callable
        # that does not carry a Thorn venue annotation.
        async def custom() -> str:
            """Custom."""
            return "x"

        assert (
            wrap_function(custom, venue=ToolVenue.SANDBOX).venue
            is ToolVenue.SANDBOX
        )
        assert (
            wrap_function(custom, venue=ToolVenue.IN_PROCESS).venue
            is ToolVenue.IN_PROCESS
        )

    def test_wrap_function_no_venue_raises(self):
        # No silent default: a function with no @tool decoration and
        # no explicit venue= argument is rejected.  The error message
        # points the author at the right fix instead of letting the
        # call quietly succeed with whatever default we picked.
        async def custom() -> str:
            """Custom."""
            return "x"

        with pytest.raises(TypeError, match="has no venue"):
            wrap_function(custom)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecoratorVenueRequired:
    def test_bare_decorator_rejected(self):
        # @tool with no parens used to set up a marker-only decorator.
        # Now venue is mandatory, so @tool (no parens) raises.
        with pytest.raises(TypeError):
            @tool
            def my_fn() -> str:
                """No venue."""
                return ""

    def test_no_venue_keyword_rejected(self):
        # @tool() with no venue= keyword is also rejected.
        with pytest.raises(TypeError):
            @tool()
            def my_fn() -> str:
                """No venue."""
                return ""

    def test_explicit_venue_accepted(self):
        @tool(venue=ToolVenue.SANDBOX)
        def my_fn() -> str:
            """Sandbox."""
            return ""

        assert my_fn._thorn_venue is ToolVenue.SANDBOX
        assert my_fn._thorn_tool is True


# ---------------------------------------------------------------------------
# Catalog parity: brain allowlist vs daemon registry vs catalog
# ---------------------------------------------------------------------------


class TestCatalogParity:
    """The catalog is the single source of truth.

    The brain's allowlist must be the union of both catalog lists, and
    the daemon's registry must be exactly the sandboxed slice.  These
    tests are the regression guard against the original drift bug.
    """

    def test_catalog_union_is_complete(self):
        # ALL_BUILTIN_TOOL_FUNCTIONS is the full union.
        assert set(ALL_BUILTIN_TOOL_FUNCTIONS) == (
            set(IN_PROCESS_TOOLS) | set(SANDBOXED_TOOLS)
        )

    def test_catalog_lists_disjoint(self):
        # A tool can only live in one venue at a time.  Overlap would
        # mean the brain and daemon both think they own dispatch.
        assert not (set(IN_PROCESS_TOOLS) & set(SANDBOXED_TOOLS))

    def test_brain_allowlist_matches_catalog(self):
        # Reset the lazy cache so the test sees a fresh build.
        import thorn.core._func as _func_mod
        from thorn.core._func import _known_builtin_tools

        _func_mod._KNOWN_BUILTIN_TOOLS = None
        try:
            allowed = _known_builtin_tools()
            assert allowed == set(ALL_BUILTIN_TOOL_FUNCTIONS)
        finally:
            _func_mod._KNOWN_BUILTIN_TOOLS = None

    def test_daemon_registry_matches_sandboxed_slice(self):
        from thorn.toolhost._server import build_default_registry

        _registry, table = build_default_registry()
        registered_names = set(table.keys())
        expected_names = {fn.__name__ for fn in SANDBOXED_TOOLS}
        assert registered_names == expected_names

    def test_write_file_is_not_registered_anywhere(self):
        from thorn.core._func import _known_builtin_tools
        from thorn.toolhost._server import build_default_registry

        _registry, table = build_default_registry()
        known_names = {fn.__name__ for fn in _known_builtin_tools()}

        assert "write_file" not in known_names
        assert "write_file" not in table
        assert "write_file" not in {fn.__name__ for fn in SANDBOXED_TOOLS}

    def test_in_process_tools_absent_from_daemon_registry(self):
        # The daemon must not see in-process tools: those need brain
        # state it doesn't have, and registering them there would mask
        # a routing bug as a successful dispatch with wrong behaviour.
        from thorn.toolhost._server import build_default_registry

        _registry, table = build_default_registry()
        in_process_names = {fn.__name__ for fn in IN_PROCESS_TOOLS}
        assert not (in_process_names & set(table.keys())), (
            "Daemon registry leaked in-process tools: "
            f"{in_process_names & set(table.keys())}"
        )


# ---------------------------------------------------------------------------
# _prepare_tools allowlist behaviour
# ---------------------------------------------------------------------------


class TestPrepareToolsAllowList:
    def test_known_builtins_accepted_bare(self):
        from thorn.core._tools import ALL_BUILTIN_TOOLS

        result = _prepare_tools(ALL_BUILTIN_TOOLS)
        assert {w.schema["function"]["name"] for w in result} == {
            fn.__name__ for fn in ALL_BUILTIN_TOOLS
        }

    def test_run_shell_accepted_bare(self):
        from thorn.core._tools import run_shell

        result = _prepare_tools([run_shell])
        assert result[0].schema["function"]["name"] == "run_shell"

    def test_peer_tools_accepted_bare(self):
        # The original P0 regression: PEER_TOOLS were missing from the
        # brain allowlist and ProjectCoordinator._collect_tools() failed
        # at _prepare_tools time.  This is the direct guard against that
        # specific class of regression.
        from thorn.tools.peers import PEER_TOOLS

        result = _prepare_tools(PEER_TOOLS)
        assert {w.schema["function"]["name"] for w in result} == {
            fn.__name__ for fn in PEER_TOOLS
        }

    def test_forge_tools_accepted_bare(self):
        from thorn.tools.forge import FORGE_TOOLS

        result = _prepare_tools(FORGE_TOOLS)
        assert {w.schema["function"]["name"] for w in result} == {
            fn.__name__ for fn in FORGE_TOOLS
        }

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

        result = _prepare_tools(
            [wrap_function(custom, venue=ToolVenue.SANDBOX)]
        )
        assert result[0].schema["function"]["name"] == "custom"

    def test_known_callable_in_nested_iterable_accepted(self):
        from thorn.core._tools import list_directory, read_file

        result = _prepare_tools([[read_file, [list_directory]]])
        names = {w.schema["function"]["name"] for w in result}
        assert names == {"read_file", "list_directory"}


# ---------------------------------------------------------------------------
# ProjectCoordinator end-to-end smoke
# ---------------------------------------------------------------------------


class TestProjectCoordinatorTools:
    """Direct guard against the original P0 regression.

    The reviewing agent that flagged the underlying bug had to bypass
    ``_prepare_tools()`` to enumerate ProjectCoordinator's tools at
    all -- the call was failing because PEER_TOOLS were missing from
    the allowlist.  This test exercises the end-to-end path that was
    broken.
    """

    def test_collect_tools_succeeds(self):
        from thorn.gateway._agents import ProjectCoordinator

        tools = ProjectCoordinator._collect_tools()
        assert len(tools) > 0
        names = {getattr(t, "__name__", str(t)) for t in tools}
        # Spot-check: peer, forge, file, shell, and inbox tools are
        # all present.
        assert "peer_by_account" in names
        assert "forge_read_issue" in names
        assert "read_file" in names
        assert "run_shell" in names
        assert "list_inbox_items" in names

    def test_no_dedicated_git_tools(self):
        # Git operations now go through run_shell; ProjectCoordinator
        # should carry no git_* @tool functions.
        from thorn.gateway._agents import ProjectCoordinator

        tools = ProjectCoordinator._collect_tools()
        names = {getattr(t, "__name__", str(t)) for t in tools}
        assert not any(name.startswith("git_") for name in names)

    def test_lean_coordinator_collect_tools_succeeds(self):
        from thorn.gateway._agents import LeanProjectCoordinator

        tools = LeanProjectCoordinator._collect_tools()
        prepared = _prepare_tools(tools)
        names = {tool.schema["function"]["name"] for tool in prepared}

        assert "run_shell" in names
        assert "forge_create_change_request" in names
        assert "peer_by_account" not in names
