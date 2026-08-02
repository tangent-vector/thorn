"""Thorn-as-MCP-server: expose Thorn tools via the MCP protocol.

This module hosts the *server-side* MCP integration -- the
:func:`serve_tools` entry point that takes a list of Thorn
``_WrappedTool`` instances and runs them as a FastMCP server (used by
``thorn serve``).  The complementary *client-side* path -- consuming
external MCP servers from inside the per-agent ``thorn-toolhost``
daemon -- lives at :class:`thorn.toolhost._mcp_host.MCPHost` since
Phase C.1.

History
-------
Earlier phases of Thorn shipped a brain-side ``MCPToolSource`` here
that opened ``ClientSession`` connections in the same process as the
agent loop.  Phase C.1 retired it: brain-side per-prompt MCP discovery
now goes through :func:`thorn.runtime._mcp_tools.discover_mcp_tools`
which routes every list/call through the daemon, keeping MCP traffic
inside the sandbox boundary.  ``MCPServerConfig`` itself lives in its
own module ([src/thorn/core/_mcp_config.py](src/thorn/core/_mcp_config.py))
and is re-exported from here for the existing
``from thorn.core._mcp import MCPServerConfig`` callers.
"""

from __future__ import annotations

import logging
from typing import Any

from thorn.core._loop import _WrappedTool
from thorn.core._mcp_config import MCPServerConfig

logger = logging.getLogger(__name__)


def _require_mcp(feature: str = "MCP support") -> None:
    """Raise a clear error if the ``mcp`` package is not installed."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            f"{feature} requires the 'mcp' package. "
            "Install it with: uv pip install 'thorn-agent[mcp]'"
        ) from None


# ---------------------------------------------------------------------------
# Schema conversion: OpenAI-style -> MCP ``Tool`` info dict
# ---------------------------------------------------------------------------
#
# The reverse direction (MCP ``Tool`` -> OpenAI schema, used when the
# brain *consumes* an external MCP server) lives at
# :func:`thorn.toolhost._mcp_host._mcp_tool_to_openai_schema`.

def _openai_schema_to_mcp_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style tool schema to the dict form expected by
    ``FastMCP.add_tool`` / the low-level MCP server."""
    func = schema.get("function", {})
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "inputSchema": func.get("parameters", {}),
    }


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


__all__ = ["MCPServerConfig", "serve_tools"]
