"""MCP integration: consume external MCP servers and serve thorn tools.

Client side
-----------
``MCPToolSource`` connects to one or more MCP servers (stdio or HTTP),
discovers their tools, and wraps each one as a ``_WrappedTool`` that
the agent loop can dispatch like any other tool.  Configurations are
sourced from the unified context-gathering pipeline; see
:func:`thorn.runtime._context_layers.collect_mcp_configs_for_directory`
and :class:`~thorn.runtime._prompt_assembly.AssembledPromptContext`.

Server side
-----------
``serve_tools`` takes a list of ``_WrappedTool`` instances (from
discovery or manual wrapping) and exposes them as an MCP server.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from thorn.core._loop import _WrappedTool
from thorn.core._mcp_config import MCPServerConfig
from thorn.core._schema import serialize_for_tool_result

logger = logging.getLogger(__name__)


def _require_mcp(feature: str = "MCP support") -> None:
    """Raise a clear error if the ``mcp`` package is not installed."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            f"{feature} requires the 'mcp' package. "
            "Install it with:  pip install thorn[mcp]"
        ) from None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# ``MCPServerConfig`` lives in :mod:`thorn.core._mcp_config` so it can be
# imported by callers (the runtime context-gathering pipeline today, the
# toolhost daemon's MCP client tomorrow) without dragging in this module's
# brain-side ``MCPToolSource`` / ``serve_tools`` surface.  Re-exported here
# for the existing ``from thorn.core._mcp import MCPServerConfig`` callers.


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------

def _mcp_tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Convert an MCP ``Tool`` object to an OpenAI-style tool schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def _openai_schema_to_mcp_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style tool schema to the dict form expected by
    ``FastMCP.add_tool`` / the low-level MCP server."""
    func = schema.get("function", {})
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "inputSchema": func.get("parameters", {}),
    }


def _mcp_result_to_string(result: Any) -> str:
    """Extract text from an MCP ``CallToolResult``."""
    parts: list[str] = []
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Client: MCPToolSource
# ---------------------------------------------------------------------------

class MCPToolSource:
    """Async context manager that connects to MCP servers and provides
    their tools as ``_WrappedTool`` instances.

    Usage::

        async with MCPToolSource(configs) as src:
            all_tools = local_tools + src.tools
            await run_agent_loop(..., tools=all_tools)

    *configs* is typically the ``mcp_configs`` field of an
    :class:`~thorn.runtime._prompt_assembly.AssembledPromptContext`,
    i.e. the deduplicated list produced by the per-prompt
    context-gathering pipeline.
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        _require_mcp("MCPToolSource")
        self._configs = configs
        self._tools: list[_WrappedTool] = []
        self._exit_stack: AsyncExitStack | None = None

    @property
    def tools(self) -> list[_WrappedTool]:
        return list(self._tools)

    async def __aenter__(self) -> MCPToolSource:
        from mcp import ClientSession

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for cfg in self._configs:
            try:
                session = await self._connect(cfg)
                tools_result = await session.list_tools()
                for mcp_tool in tools_result.tools:
                    schema = _mcp_tool_to_openai_schema(mcp_tool)
                    tool_name = mcp_tool.name

                    bound_session = session

                    async def _execute(
                        _sess: ClientSession = bound_session,
                        _name: str = tool_name,
                        **kwargs: Any,
                    ) -> str:
                        result = await _sess.call_tool(_name, kwargs)
                        return _mcp_result_to_string(result)

                    self._tools.append(_WrappedTool(
                        schema=schema,
                        execute=_execute,
                    ))

                names = [t.name for t in tools_result.tools]
                logger.info(
                    "MCP server %r: %d tools (%s)",
                    cfg.name, len(names), ", ".join(names),
                )
            except Exception:
                logger.warning(
                    "failed to connect to MCP server %r", cfg.name,
                    exc_info=True,
                )

        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(*exc)
            self._exit_stack = None
        self._tools.clear()

    async def _connect(self, cfg: MCPServerConfig) -> Any:
        """Open transport + session for *cfg* and register both on the
        exit stack so they are cleaned up in ``__aexit__``."""
        from mcp import ClientSession

        assert self._exit_stack is not None

        if cfg.url:
            from mcp.client.streamable_http import streamable_http_client

            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(cfg.url)
            )
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=cfg.command,  # type: ignore[arg-type]
                args=cfg.args,
                env=cfg.env,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )

        session: ClientSession = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session


# ---------------------------------------------------------------------------
# Server: serve thorn tools via MCP
# ---------------------------------------------------------------------------

async def serve_tools(
    tools: list[_WrappedTool],
    *,
    name: str = "thorn",
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start an MCP server that exposes the given *tools*.

    Parameters
    ----------
    tools:
        ``_WrappedTool`` instances (from ``_prepare_tools``, discovery,
        or manual wrapping).
    name:
        Server name reported to MCP clients.
    transport:
        ``"stdio"`` (default) or ``"streamable-http"``.
    host / port:
        Bind address for HTTP transport.
    """
    _require_mcp("thorn serve")

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(name)

    for wt in tools:
        info = _openai_schema_to_mcp_tool(wt.schema)
        tool_name: str = info["name"]
        if not tool_name:
            continue

        bound_tool = wt

        async def _handler(
            _wt: _WrappedTool = bound_tool,
            **kwargs: Any,
        ) -> str:
            return await _wt.execute(**kwargs)

        _handler.__name__ = tool_name
        _handler.__doc__ = info.get("description", "")

        mcp.tool(name=tool_name, description=info.get("description", ""))(_handler)

    mcp.run(transport=transport, host=host, port=port)
