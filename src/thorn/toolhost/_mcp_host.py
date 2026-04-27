"""Daemon-side MCP server lifecycle for ``thorn-toolhost``.

Phase C.1 moves MCP traffic out of the brain and into the toolhost
daemon.  :class:`MCPHost` owns:

* the inventory of MCP servers actually running inside this daemon's
  process tree, keyed by the
  :func:`~thorn.core._mcp_config.mcp_server_config_identity` of each
  :class:`~thorn.core._mcp_config.MCPServerConfig`;
* lazy connect-on-first-reference, so a server only starts when the
  brain actually wants its tools or to call one of them;
* per-server serialization (one in-flight call per server at a time)
  via a per-entry :class:`asyncio.Lock`, matching the typical MCP
  server's "I am a single-threaded subprocess" assumption;
* a cached tool list per server, populated on first
  :meth:`list_tools` and reused for subsequent calls in the same
  daemon lifetime;
* clean shutdown of every server through a single
  :class:`contextlib.AsyncExitStack` so :meth:`aclose` is the only
  teardown path the rest of the daemon needs to know about.

The host is ``mcp``-import-tolerant: if the optional ``mcp`` Python
package is unavailable the host still constructs successfully but
:attr:`mcp_available` reports ``False``, every operation raises
:class:`MCPUnavailableError`, and the surrounding
:class:`~thorn.toolhost._server.ToolhostServer` does *not* advertise
the ``"mcp"`` feature flag in its :class:`~thorn.toolhost._protocol.Hello`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from thorn.core._mcp_config import (
    MCPServerConfig,
    mcp_server_config_identity,
)

logger = logging.getLogger(__name__)


class MCPUnavailableError(RuntimeError):
    """Raised when an MCP operation is attempted without ``mcp`` installed.

    Surfaces from inside the daemon as an ``error_kind="mcp_unavailable"``
    response so the brain can render a clear "rebuild the sandbox image
    with [mcp] installed" message instead of a stack trace.
    """


@dataclass
class _MCPEntry:
    """Per-server runtime state inside :class:`MCPHost`.

    ``call_lock`` serializes ``initialize`` / ``list_tools`` / ``call_tool``
    against a single :class:`mcp.ClientSession`; the underlying server
    is typically single-threaded and parallel calls would race even if
    the SDK itself is reentrant.  ``tools_cache`` is populated by the
    first :meth:`MCPHost.list_tools` call for this entry and reused
    thereafter; the cache lives only as long as the daemon process,
    which is short enough that staleness is not a concern in v1.
    """

    config: MCPServerConfig
    call_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: Any | None = None
    tools_cache: list[dict[str, Any]] | None = None


class MCPHost:
    """Daemon-side fleet manager for MCP server connections.

    One instance per :class:`~thorn.toolhost._server.ToolhostServer`.
    Caller flow::

        host = MCPHost()                      # cheap; does not import mcp
        if host.mcp_available:
            tools = await host.list_tools(cfg)
            text = await host.call_tool(cfg, "do_thing", {"x": 1})
        await host.aclose()

    Concurrency model:

    * A single ``_creation_lock`` guards the registration of new
      entries so that two concurrent requests for the same server
      collapse to one running process even when they arrive in the
      same event-loop tick.
    * Each entry's ``call_lock`` then guards every operation against
      that server, so per-server traffic is naturally sequential.
    * ``aclose`` waits for any in-flight operation by acquiring each
      ``call_lock`` before tearing the stack down; a single
      :class:`AsyncExitStack` owns every transport / session created
      by the host so teardown is the inverse of construction.
    """

    def __init__(self) -> None:
        self._entries: dict[
            tuple[Any, ...], _MCPEntry
        ] = {}
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._creation_lock: asyncio.Lock = asyncio.Lock()
        self._closed: bool = False

        # Probe ``mcp`` once at construction so the feature-flag
        # decision is stable for the lifetime of this host.  Importing
        # here (and not lazily) means ``ToolhostServer.__init__`` can
        # synchronously decide whether to advertise ``"mcp"`` in its
        # handshake.
        try:
            import mcp  # noqa: F401

            self._mcp_available = True
        except ImportError:
            self._mcp_available = False

    @property
    def mcp_available(self) -> bool:
        """``True`` if the ``mcp`` package is importable in this process."""
        return self._mcp_available

    async def list_tools(
        self,
        config: MCPServerConfig,
    ) -> list[dict[str, Any]]:
        """Return the OpenAI-style tool schemas served by *config*.

        First call for a given server identity connects, initializes
        the session, asks the server for its tool list, and caches the
        translated schemas.  Subsequent calls return the cache.  The
        per-server lock keeps a slow ``initialize`` from blocking
        unrelated servers but does serialize concurrent listings of
        the same server.
        """
        self._require_open()
        if not self._mcp_available:
            raise MCPUnavailableError(
                "MCP support is not installed in this toolhost daemon; "
                "rebuild the sandbox image with thorn[mcp].",
            )
        entry = await self._get_or_create_entry(config)
        async with entry.call_lock:
            await self._ensure_connected(entry)
            if entry.tools_cache is not None:
                return list(entry.tools_cache)
            tools_result = await entry.session.list_tools()
            schemas = [
                _mcp_tool_to_openai_schema(tool) for tool in tools_result.tools
            ]
            entry.tools_cache = schemas
            logger.info(
                "MCPHost: server %r exposes %d tools (%s)",
                config.name,
                len(schemas),
                ", ".join(t.name for t in tools_result.tools),
            )
            return list(schemas)

    async def call_tool(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Invoke *tool_name* on the server identified by *config*.

        Returns the concatenated text-content of the MCP result.
        Raises whatever the underlying ``ClientSession.call_tool``
        raises (timeout, transport error, server-reported error);
        callers wrap those into protocol-level error frames.
        """
        self._require_open()
        if not self._mcp_available:
            raise MCPUnavailableError(
                "MCP support is not installed in this toolhost daemon; "
                "rebuild the sandbox image with thorn[mcp].",
            )
        entry = await self._get_or_create_entry(config)
        async with entry.call_lock:
            await self._ensure_connected(entry)
            result = await entry.session.call_tool(tool_name, arguments)
            return _mcp_result_to_string(result)

    async def aclose(self) -> None:
        """Tear down every connected server.

        Acquires each entry's call lock first so an in-flight call
        finishes (or its caller's surrounding cancel propagates) before
        the underlying stream closes.  Idempotent.
        """
        if self._closed:
            return
        self._closed = True

        for entry in list(self._entries.values()):
            async with entry.call_lock:
                entry.session = None
                entry.tools_cache = None

        try:
            await self._exit_stack.aclose()
        except Exception:
            logger.exception("MCPHost: error during exit-stack teardown")
        self._entries.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MCPHost is closed")

    async def _get_or_create_entry(
        self,
        config: MCPServerConfig,
    ) -> _MCPEntry:
        key = mcp_server_config_identity(config)
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        async with self._creation_lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry
            new_entry = _MCPEntry(config=config)
            self._entries[key] = new_entry
            return new_entry

    async def _ensure_connected(self, entry: _MCPEntry) -> None:
        """Open transport + ClientSession for *entry* if not already.

        Both the transport context manager and the session context
        manager are pushed onto the host's single
        :class:`AsyncExitStack`; :meth:`aclose` is the only thing that
        unwinds them.  Calling this on an already-connected entry is a
        no-op.
        """
        if entry.session is not None:
            return

        from mcp import ClientSession

        cfg = entry.config
        if cfg.url:
            from mcp.client.streamable_http import streamable_http_client

            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(cfg.url),
            )
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=cfg.command,  # type: ignore[arg-type]
                args=list(cfg.args),
                env=dict(cfg.env) if cfg.env is not None else None,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params),
            )

        session: ClientSession = await self._exit_stack.enter_async_context(
            ClientSession(read, write),
        )
        await session.initialize()
        entry.session = session


def _mcp_tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Convert an MCP ``Tool`` object to an OpenAI-style tool schema.

    Kept duplicated from :mod:`thorn.core._mcp` (rather than imported)
    so the daemon does not pull in the brain-side ``MCPToolSource`` /
    ``serve_tools`` plumbing that lives there.  The brain-side copy
    will go away with :class:`MCPToolSource` retirement (Phase C.1
    item 7); until then both paths happen to agree on the conversion.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def _mcp_result_to_string(result: Any) -> str:
    """Extract text from an MCP ``CallToolResult``.

    See :func:`_mcp_tool_to_openai_schema` for why this is duplicated
    rather than imported.
    """
    parts: list[str] = []
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else ""


__all__ = ["MCPHost", "MCPUnavailableError"]
