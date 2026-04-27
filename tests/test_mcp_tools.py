"""Tests for ``thorn.runtime._mcp_tools.discover_mcp_tools``.

The function under test owns Phase C.1's brain-side bridge between the
per-prompt context-gathering pipeline (which produces
``MCPServerConfig`` instances) and the agent loop (which expects
``_WrappedTool`` instances).  We exercise:

* the success path, where each server's tools land in the result
  with the right schema name, venue, and ``mcp_*`` metadata;
* per-server inventory failure (warn-and-skip);
* the executor missing the ``list_mcp_server_tools`` surface
  (skip-everything);
* collision-prefix policy (built-in vs MCP, MCP vs MCP);
* the true-ambiguity case (two surviving servers with the same
  name and tool) -- error-and-skip the colliders, surviving tools
  still register;
* schema isolation: the daemon-cached schema dict is not mutated by
  the relabel step.
"""

from __future__ import annotations

from typing import Any

import pytest

from thorn.core._executor import ToolVenue
from thorn.core._mcp_config import MCPServerConfig
from thorn.runtime._mcp_tools import discover_mcp_tools


class _FakeExecutor:
    """Stand-in for ``DaemonToolExecutor.list_mcp_server_tools``.

    Stores per-server tool schemas (or ``Exception`` instances) keyed
    by the config the brain passes in; reads use object identity so
    the test cases stay decoupled from ``MCPServerConfig`` equality
    semantics.
    """

    def __init__(
        self,
        responses: dict[int, list[dict[str, Any]] | Exception] | None = None,
    ) -> None:
        self._responses = responses or {}
        self.calls: list[MCPServerConfig] = []

    async def list_mcp_server_tools(
        self, config: MCPServerConfig,
    ) -> list[dict[str, Any]]:
        self.calls.append(config)
        result = self._responses.get(id(config))
        if isinstance(result, Exception):
            raise result
        return result or []


def _schema(name: str, *, description: str = "") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _name(wrapped: Any) -> str:
    return wrapped.schema["function"]["name"]


@pytest.mark.asyncio
async def test_no_configs_returns_empty():
    """Trivial early-out: nothing to ask the daemon for."""
    result = await discover_mcp_tools(
        sandbox_executor=_FakeExecutor(),
        mcp_configs=[],
        builtin_tool_names=set(),
    )
    assert result == []


@pytest.mark.asyncio
async def test_executor_without_list_method_skips():
    """Old executors / non-daemon executors trigger warn-and-skip."""
    cfg = MCPServerConfig(name="x", command="echo")

    class _Bare:
        pass

    result = await discover_mcp_tools(
        sandbox_executor=_Bare(),
        mcp_configs=[cfg],
        builtin_tool_names=set(),
    )
    assert result == []


@pytest.mark.asyncio
async def test_unique_names_pass_through_unprefixed():
    """No collisions => exposed name == server-side tool name."""
    cfg = MCPServerConfig(name="github", command="github-mcp")
    schema_a = _schema("issue_list")
    schema_b = _schema("pr_create")

    executor = _FakeExecutor({id(cfg): [schema_a, schema_b]})
    result = await discover_mcp_tools(
        sandbox_executor=executor,
        mcp_configs=[cfg],
        builtin_tool_names={"read_file", "write_file"},
    )

    assert sorted(_name(w) for w in result) == ["issue_list", "pr_create"]
    for wrapped in result:
        assert wrapped.venue is ToolVenue.SANDBOX
        assert wrapped.mcp_server_config is cfg
        # Unprefixed: the daemon-side name equals the registry name.
        assert wrapped.mcp_tool_name == _name(wrapped)


@pytest.mark.asyncio
async def test_builtin_collision_forces_prefix():
    """Built-ins always win the bare slot; MCP version gets prefixed."""
    cfg = MCPServerConfig(name="fs", command="fs-mcp")
    executor = _FakeExecutor({id(cfg): [_schema("read_file"), _schema("ls")]})

    result = await discover_mcp_tools(
        sandbox_executor=executor,
        mcp_configs=[cfg],
        builtin_tool_names={"read_file"},
    )

    by_exposed = {_name(w): w for w in result}
    assert "fs__read_file" in by_exposed
    assert "ls" in by_exposed  # no collision; stays unprefixed
    # Daemon-side name is always the unprefixed server-side name.
    assert by_exposed["fs__read_file"].mcp_tool_name == "read_file"
    assert by_exposed["ls"].mcp_tool_name == "ls"


@pytest.mark.asyncio
async def test_cross_server_collision_prefixes_both_sides():
    """Two MCP servers exposing ``search`` => both become ``<srv>__search``."""
    a = MCPServerConfig(name="github", command="github-mcp")
    b = MCPServerConfig(name="sentry", command="sentry-mcp")
    executor = _FakeExecutor({
        id(a): [_schema("search"), _schema("github_only")],
        id(b): [_schema("search"), _schema("sentry_only")],
    })

    result = await discover_mcp_tools(
        sandbox_executor=executor,
        mcp_configs=[a, b],
        builtin_tool_names=set(),
    )

    exposed_names = sorted(_name(w) for w in result)
    assert exposed_names == [
        "github__search",
        "github_only",
        "sentry__search",
        "sentry_only",
    ]
    # The daemon-side name remains the unprefixed ``search`` for both.
    by_exposed = {_name(w): w for w in result}
    assert by_exposed["github__search"].mcp_tool_name == "search"
    assert by_exposed["github__search"].mcp_server_config is a
    assert by_exposed["sentry__search"].mcp_tool_name == "search"
    assert by_exposed["sentry__search"].mcp_server_config is b


@pytest.mark.asyncio
async def test_per_server_failure_warns_and_skips_only_that_server(caplog):
    """One broken server must not block the rest of the prompt."""
    good = MCPServerConfig(name="good", command="good-mcp")
    bad = MCPServerConfig(name="bad", command="bad-mcp")
    executor = _FakeExecutor({
        id(good): [_schema("good_tool")],
        id(bad): RuntimeError("connect refused"),
    })

    with caplog.at_level("WARNING", logger="thorn.runtime._mcp_tools"):
        result = await discover_mcp_tools(
            sandbox_executor=executor,
            mcp_configs=[good, bad],
            builtin_tool_names=set(),
        )

    assert [_name(w) for w in result] == ["good_tool"]
    assert any(
        "MCP server 'bad'" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_duplicate_tool_within_one_server_is_collapsed():
    """A buggy server listing the same tool twice must not double-register."""
    cfg = MCPServerConfig(name="srv", command="srv-mcp")
    executor = _FakeExecutor({
        id(cfg): [_schema("dup"), _schema("dup"), _schema("unique")],
    })

    result = await discover_mcp_tools(
        sandbox_executor=executor,
        mcp_configs=[cfg],
        builtin_tool_names=set(),
    )

    assert sorted(_name(w) for w in result) == ["dup", "unique"]


@pytest.mark.asyncio
async def test_true_ambiguity_after_prefixing_is_dropped(caplog):
    """Two configs share name 'srv' and tool 'go' -> errors-and-skips both."""
    a = MCPServerConfig(name="srv", command="srv-a")
    b = MCPServerConfig(name="srv", command="srv-b")
    executor = _FakeExecutor({
        id(a): [_schema("go"), _schema("only_a")],
        id(b): [_schema("go"), _schema("only_b")],
    })

    with caplog.at_level("ERROR", logger="thorn.runtime._mcp_tools"):
        result = await discover_mcp_tools(
            sandbox_executor=executor,
            mcp_configs=[a, b],
            builtin_tool_names=set(),
        )

    exposed = sorted(_name(w) for w in result)
    # First 'go' wins; second 'go' is dropped.  Non-colliding tools
    # from both servers still register.
    assert exposed == ["only_a", "only_b", "srv__go"]
    assert any(
        "even after server-name prefixing" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_schema_relabel_does_not_mutate_input():
    """Daemon-cached schemas must not be mutated by the relabel step."""
    cfg_a = MCPServerConfig(name="a", command="a")
    cfg_b = MCPServerConfig(name="b", command="b")
    shared_schema = _schema("collide")

    # Both servers return literally the *same* dict object so a
    # mutation by the relabel step would be observable on the second
    # iteration via the first server's view.  This mimics the
    # daemon's tool-list cache returning the same dict on subsequent
    # prompts.
    executor = _FakeExecutor({
        id(cfg_a): [shared_schema],
        id(cfg_b): [shared_schema],
    })

    result = await discover_mcp_tools(
        sandbox_executor=executor,
        mcp_configs=[cfg_a, cfg_b],
        builtin_tool_names=set(),
    )

    # The original schema dict is untouched.
    assert shared_schema["function"]["name"] == "collide"
    # And both collided entries are exposed under prefixed names.
    assert sorted(_name(w) for w in result) == ["a__collide", "b__collide"]


@pytest.mark.asyncio
async def test_unsupported_in_process_execute_raises():
    """The wrapped tool's local execute path must error loudly."""
    cfg = MCPServerConfig(name="srv", command="srv")
    executor = _FakeExecutor({id(cfg): [_schema("only_tool")]})

    [wrapped] = await discover_mcp_tools(
        sandbox_executor=executor,
        mcp_configs=[cfg],
        builtin_tool_names=set(),
    )

    with pytest.raises(RuntimeError, match="daemon-backed sandbox executor"):
        await wrapped.execute(some_arg=1)
