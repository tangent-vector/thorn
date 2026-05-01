"""Tool execution seam: ``ToolExecutor`` protocol, registry, and router.

This module defines the interface the agent loop uses to invoke tool
calls.  It is the staging area for the Phase A sandbox split:

* :class:`ToolRegistry` is a per-agent description of the tools the
  model may call.  Each entry carries the tool's schema, its OpenAI-style
  name, a :class:`ToolVenue` saying where the tool executes, and an
  optional ``call_node_class`` used by history recording.
* :class:`ToolExecutor` is the runtime interface.  The registry says
  *which* tool to run; the executor says *how*.  Different venues are
  served by different executors (``InProcessToolExecutor`` today; in
  Phase A a ``DaemonToolExecutor`` will back the ``SANDBOX`` venue).
* :class:`ExecutorRouter` picks the executor for a given venue so that
  the agent loop does not need to know the concrete implementations.

In Phase A both the in-process and sandbox venues are implemented by
:class:`InProcessToolExecutor`; the sandbox binding is rebound to
:class:`DaemonToolExecutor` as that work lands.  The seam is designed so
that change requires no edits to the agent loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from thorn.core._history import ToolCallNode
    from thorn.core._loop import _WrappedTool
    from thorn.core._mcp_config import MCPServerConfig


class ToolVenue(str, Enum):
    """Where a tool runs relative to the brain.

    ``IN_PROCESS`` tools execute in the same Python process as the
    agent loop and can touch brain-owned state (e.g. the session
    inbox).  ``SANDBOX`` tools are ultimately
    intended to run inside the per-agent ``thorn-toolhost`` daemon; in
    Phase A they still run in-process but are tracked separately so
    that later wiring the daemon does not require touching the loop.
    """

    IN_PROCESS = "in_process"
    SANDBOX = "sandbox"


@dataclass(frozen=True)
class ToolInvocation:
    """A single tool call the loop wants executed.

    ``call_id`` is the provider-supplied identifier; the router uses it
    only for cancellation bookkeeping.  ``tool_name`` is the registered
    name (post-normalization) and ``arguments`` is the pre-parsed
    keyword payload.

    ``per_call_context`` carries runtime metadata that varies per tool
    call but is not part of the model-visible tool arguments.  The
    daemon uses it to reconstruct the effective execution context for
    sandboxed calls, including the session workspace subdirectory and
    session-key scope tag.

    ``mcp_server_config`` carries an MCP server's connection
    parameters when the call should be routed through the daemon's
    :class:`~thorn.toolhost._mcp_host.MCPHost` rather than the static
    built-in registry.  ``None`` (the default) means "this is a
    built-in tool, dispatch normally"; brain-side tool resolution
    populates the field for tools that came from an MCP server.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    per_call_context: dict[str, Any] = field(default_factory=dict)
    mcp_server_config: "MCPServerConfig | None" = None


@dataclass(frozen=True)
class ToolOutputChunk:
    """Streaming chunk emitted by a tool during execution.

    Phase A defines the shape but no built-in tool emits chunks yet; the
    ``stream`` label mirrors the wire protocol (``stdout``, ``stderr``,
    or ``structured``) so downstream consumers can route chunks without
    having to infer intent.
    """

    call_id: str
    stream: str
    data: str


@dataclass(frozen=True)
class ToolInvocationResult:
    """Terminal result of a tool call.

    ``content`` is always a string (the stringified tool result or error
    message); ``is_error`` distinguishes an error that should surface as
    a failed tool-result message to the model from a successful result.
    ``error_kind`` is an opaque classifier reserved for future daemon-
    side wiring (e.g. ``timeout``, ``cancelled``, ``sandbox_crashed``).
    """

    content: str
    is_error: bool = False
    error_kind: str | None = None


OnChunkCallback = Callable[[ToolOutputChunk], Awaitable[None]]


class ToolExecutor(Protocol):
    """Executes tool calls for one or more :class:`ToolVenue` values.

    Implementations are free to treat every call as synchronous (the
    in-process executor) or as a multiplexed RPC over a long-lived
    transport (the daemon executor).  The loop only ever awaits
    :meth:`invoke`; cancellation is best-effort and ``aclose`` is
    invoked exactly once when the agent / runtime shuts down.
    """

    async def invoke(
        self,
        invocation: ToolInvocation,
        *,
        on_chunk: OnChunkCallback | None = None,
    ) -> ToolInvocationResult:
        ...

    async def cancel(self, call_id: str) -> None:
        ...

    async def aclose(self) -> None:
        ...


@dataclass(frozen=True)
class ToolRegistryEntry:
    """Static description of a tool registered for an agent.

    The registry entry is the single source of truth for *where* a tool
    runs: the loop reads ``venue``, the router maps ``venue`` to an
    executor, and nothing else in the loop needs to know which
    implementation services the call.

    ``mcp_server_config`` and ``mcp_tool_name`` are non-``None`` only
    for tools sourced from an MCP server.  They flow from the brain's
    per-prompt MCP discovery (see
    :mod:`thorn.runtime._mcp_tools`) through to the
    :class:`ToolInvocation` the loop hands to the daemon executor: the
    daemon needs the ``MCPServerConfig`` to identify which server to
    route through and the ``mcp_tool_name`` (the unprefixed name the
    server itself exposes) so it can call ``ClientSession.call_tool``
    correctly even when the brain has prefixed the schema name to
    avoid a collision.
    """

    name: str
    schema: dict[str, Any]
    venue: ToolVenue
    call_node_class: type[ToolCallNode] | None = None
    mcp_server_config: "MCPServerConfig | None" = None
    mcp_tool_name: str | None = None


class ToolRegistry:
    """Map of tool name -> :class:`ToolRegistryEntry`.

    The registry owns the tool schemas exposed to the model and the
    per-tool venue classification.  It intentionally does *not* hold
    executable callables; those live in the executors.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: list[ToolRegistryEntry] | None = None) -> None:
        self._entries: dict[str, ToolRegistryEntry] = {}
        if entries:
            for entry in entries:
                self.register(entry)

    def register(self, entry: ToolRegistryEntry) -> None:
        """Add *entry*, rejecting duplicate names."""
        if entry.name in self._entries:
            raise ValueError(f"duplicate tool registration: {entry.name!r}")
        self._entries[entry.name] = entry

    def get(self, name: str) -> ToolRegistryEntry | None:
        return self._entries.get(name)

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def entries(self) -> list[ToolRegistryEntry]:
        return list(self._entries.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Schemas in registration order."""
        return [entry.schema for entry in self._entries.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class InProcessToolExecutor:
    """Executor that runs tools in the agent's Python process.

    The executor wraps the legacy ``_WrappedTool.execute`` callables
    directly: they already run in whatever ``ExecutionContext`` the
    caller has installed via contextvars, so no per-call setup is
    needed.  Cancellation is intentionally a no-op in Phase A -- the
    current tool callables do not react to task cancellation, and the
    scheduler never cancels a tool call mid-flight for in-process
    tools.  The :class:`DaemonToolExecutor` is what actually exercises
    the cancel path.
    """

    __slots__ = ("_tools",)

    def __init__(self, tools: dict[str, _WrappedTool] | None = None) -> None:
        self._tools: dict[str, _WrappedTool] = dict(tools or {})

    def register(self, name: str, tool: _WrappedTool) -> None:
        if name in self._tools:
            raise ValueError(f"duplicate in-process tool: {name!r}")
        self._tools[name] = tool

    async def invoke(
        self,
        invocation: ToolInvocation,
        *,
        on_chunk: OnChunkCallback | None = None,
    ) -> ToolInvocationResult:
        tool = self._tools.get(invocation.tool_name)
        if tool is None:
            return ToolInvocationResult(
                content=f"Unknown tool: {invocation.tool_name!r}",
                is_error=True,
                error_kind="unknown_tool",
            )
        result = await tool.execute(**invocation.arguments)
        if isinstance(result, str):
            content = result
        else:
            from thorn.core._schema import serialize_for_tool_result
            content = serialize_for_tool_result(result)
        return ToolInvocationResult(content=content)

    async def cancel(self, call_id: str) -> None:
        return None

    async def aclose(self) -> None:
        self._tools.clear()


class ExecutorRouter:
    """Route invocations to executors based on :class:`ToolVenue`.

    The router is owned per-agent by the loop's caller; the loop reads
    the venue off the registry entry and asks the router for the
    executor to dispatch to.  In Phase A both venues typically map to
    a single :class:`InProcessToolExecutor`.
    """

    __slots__ = ("_executors",)

    def __init__(
        self,
        executors: dict[ToolVenue, ToolExecutor] | None = None,
    ) -> None:
        self._executors: dict[ToolVenue, ToolExecutor] = dict(executors or {})

    def bind(self, venue: ToolVenue, executor: ToolExecutor) -> None:
        """Register *executor* for *venue*, replacing any existing binding."""
        self._executors[venue] = executor

    def for_venue(self, venue: ToolVenue) -> ToolExecutor:
        try:
            return self._executors[venue]
        except KeyError as exc:
            raise KeyError(
                f"no executor registered for tool venue {venue!r}",
            ) from exc

    def venues(self) -> list[ToolVenue]:
        return list(self._executors.keys())

    async def aclose(self) -> None:
        """Close each bound executor exactly once, even if duplicated."""
        seen: set[int] = set()
        for executor in self._executors.values():
            if id(executor) in seen:
                continue
            seen.add(id(executor))
            await executor.aclose()


def build_registry_from_wrapped_tools(
    tools: list[_WrappedTool],
) -> ToolRegistry:
    """Derive a :class:`ToolRegistry` from a list of ``_WrappedTool``.

    Helper used by ``run_agent_loop`` for callers that still pass tools
    the old way.  The venue is pulled from each wrapped tool
    (defaulting to ``IN_PROCESS``); the registry entry name is the
    OpenAI-style function name inside the schema.
    """
    entries: list[ToolRegistryEntry] = []
    for tool in tools:
        name = tool.schema.get("function", {}).get("name", "")
        if not name:
            continue
        entries.append(
            ToolRegistryEntry(
                name=name,
                schema=tool.schema,
                venue=getattr(tool, "venue", ToolVenue.IN_PROCESS),
                call_node_class=tool.call_node_class,
                mcp_server_config=getattr(tool, "mcp_server_config", None),
                mcp_tool_name=getattr(tool, "mcp_tool_name", None),
            )
        )
    return ToolRegistry(entries)


def build_default_router(
    tools: list[_WrappedTool],
) -> ExecutorRouter:
    """Build an :class:`ExecutorRouter` covering every venue in *tools*.

    For Phase A both ``IN_PROCESS`` and ``SANDBOX`` map to a single
    :class:`InProcessToolExecutor` so the split exists in shape even
    though no daemon is wired yet.  As ``DaemonToolExecutor`` lands,
    callers will build the router explicitly instead of using this
    helper, and bind each venue to its real executor.
    """
    dispatch: dict[str, _WrappedTool] = {}
    for tool in tools:
        name = tool.schema.get("function", {}).get("name", "")
        if name:
            dispatch[name] = tool

    in_process = InProcessToolExecutor(dispatch)
    return ExecutorRouter(
        {
            ToolVenue.IN_PROCESS: in_process,
            ToolVenue.SANDBOX: in_process,
        }
    )


def build_split_router(
    tools: list[_WrappedTool],
    sandbox_executor: ToolExecutor,
) -> ExecutorRouter:
    """Build a router that sends ``SANDBOX`` calls to *sandbox_executor*.

    Used by the gateway and CLI when a real daemon-backed executor is
    available.  The returned router still owns an
    :class:`InProcessToolExecutor` for the in-process tools (e.g. the
    inbox tools) -- those need brain-side state and never go
    through the daemon.  The supplied *sandbox_executor* is *not*
    closed by :meth:`ExecutorRouter.aclose`'s normal dedup logic when
    it is shared across calls; callers manage its lifecycle directly.
    """
    in_process_dispatch: dict[str, _WrappedTool] = {}
    for tool in tools:
        venue = getattr(tool, "venue", ToolVenue.IN_PROCESS)
        if venue is not ToolVenue.IN_PROCESS:
            continue
        name = tool.schema.get("function", {}).get("name", "")
        if name:
            in_process_dispatch[name] = tool

    in_process = InProcessToolExecutor(in_process_dispatch)
    return ExecutorRouter(
        {
            ToolVenue.IN_PROCESS: in_process,
            ToolVenue.SANDBOX: sandbox_executor,
        }
    )
