"""Unit tests for :mod:`thorn.core._executor`.

Exercises the tool-execution seam independently of the agent loop:
registry lookups, in-process execution, routing by venue, and the
shared helpers used by ``run_agent_loop`` to build a default router.
"""

from __future__ import annotations

import pytest

from thorn.core._executor import (
    ExecutorRouter,
    InProcessToolExecutor,
    ToolInvocation,
    ToolInvocationResult,
    ToolRegistry,
    ToolRegistryEntry,
    ToolVenue,
    build_default_router,
    build_registry_from_wrapped_tools,
)
from thorn.core._loop import _WrappedTool


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        entry = ToolRegistryEntry(
            name="foo", schema=_schema("foo"), venue=ToolVenue.IN_PROCESS,
        )
        registry.register(entry)

        assert "foo" in registry
        assert registry.get("foo") is entry
        assert registry.get("missing") is None

    def test_duplicate_registration_rejected(self):
        entry = ToolRegistryEntry(
            name="foo", schema=_schema("foo"), venue=ToolVenue.IN_PROCESS,
        )
        registry = ToolRegistry([entry])
        with pytest.raises(ValueError, match="duplicate tool registration"):
            registry.register(entry)

    def test_schemas_preserve_registration_order(self):
        entries = [
            ToolRegistryEntry(
                name="a", schema=_schema("a"), venue=ToolVenue.IN_PROCESS,
            ),
            ToolRegistryEntry(
                name="b", schema=_schema("b"), venue=ToolVenue.SANDBOX,
            ),
        ]
        registry = ToolRegistry(entries)
        assert [s["function"]["name"] for s in registry.schemas()] == ["a", "b"]


class TestInProcessToolExecutor:
    @pytest.mark.asyncio
    async def test_invokes_underlying_wrapped_tool(self):
        async def run(**kwargs):
            return f"hello {kwargs['name']}"

        wrapped = _WrappedTool(schema=_schema("greet"), execute=run)
        executor = InProcessToolExecutor({"greet": wrapped})
        result = await executor.invoke(
            ToolInvocation(call_id="c1", tool_name="greet", arguments={"name": "x"}),
        )

        assert isinstance(result, ToolInvocationResult)
        assert result.content == "hello x"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_unknown_tool_is_error(self):
        executor = InProcessToolExecutor({})
        result = await executor.invoke(
            ToolInvocation(call_id="c2", tool_name="missing", arguments={}),
        )
        assert result.is_error is True
        assert result.error_kind == "unknown_tool"

    @pytest.mark.asyncio
    async def test_non_string_result_is_serialized(self):
        async def run(**kwargs):
            return {"ok": True, "value": 42}

        wrapped = _WrappedTool(schema=_schema("report"), execute=run)
        executor = InProcessToolExecutor({"report": wrapped})
        result = await executor.invoke(
            ToolInvocation(call_id="c3", tool_name="report", arguments={}),
        )
        assert result.is_error is False
        assert "42" in result.content

    @pytest.mark.asyncio
    async def test_cancel_and_aclose_are_safe(self):
        executor = InProcessToolExecutor({})
        await executor.cancel("c1")
        await executor.aclose()


class TestExecutorRouter:
    @pytest.mark.asyncio
    async def test_for_venue_returns_bound_executor(self):
        in_proc = InProcessToolExecutor({})
        router = ExecutorRouter({ToolVenue.IN_PROCESS: in_proc})
        assert router.for_venue(ToolVenue.IN_PROCESS) is in_proc

    def test_unbound_venue_raises(self):
        router = ExecutorRouter({})
        with pytest.raises(KeyError, match="no executor registered"):
            router.for_venue(ToolVenue.SANDBOX)

    @pytest.mark.asyncio
    async def test_bind_replaces_existing(self):
        first = InProcessToolExecutor({})
        second = InProcessToolExecutor({})
        router = ExecutorRouter({ToolVenue.IN_PROCESS: first})
        router.bind(ToolVenue.IN_PROCESS, second)
        assert router.for_venue(ToolVenue.IN_PROCESS) is second

    @pytest.mark.asyncio
    async def test_aclose_deduplicates_shared_executors(self):
        calls: list[str] = []

        class Counting(InProcessToolExecutor):
            async def aclose(self) -> None:
                calls.append("aclose")

        shared = Counting({})
        router = ExecutorRouter(
            {ToolVenue.IN_PROCESS: shared, ToolVenue.SANDBOX: shared}
        )
        await router.aclose()
        assert calls == ["aclose"]


class TestBuildHelpers:
    def test_registry_uses_wrapped_tool_venue(self):
        a = _WrappedTool(schema=_schema("a"), execute=lambda: None)
        b = _WrappedTool(
            schema=_schema("b"),
            execute=lambda: None,
            venue=ToolVenue.SANDBOX,
        )
        registry = build_registry_from_wrapped_tools([a, b])
        assert registry.get("a").venue is ToolVenue.IN_PROCESS
        assert registry.get("b").venue is ToolVenue.SANDBOX

    def test_registry_skips_tools_without_names(self):
        weird = _WrappedTool(schema={}, execute=lambda: None)
        registry = build_registry_from_wrapped_tools([weird])
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_default_router_shares_in_process_executor(self):
        async def run(**kwargs):
            return "ok"

        wrapped = _WrappedTool(schema=_schema("x"), execute=run)
        router = build_default_router([wrapped])
        in_proc = router.for_venue(ToolVenue.IN_PROCESS)
        sandbox = router.for_venue(ToolVenue.SANDBOX)
        assert in_proc is sandbox
        result = await in_proc.invoke(
            ToolInvocation(call_id="c1", tool_name="x", arguments={}),
        )
        assert result.content == "ok"
