"""Tests for ``thorn.core._mcp`` (Thorn-as-MCP-server side).

After Phase C.1 the brain-side MCP *client* (``MCPToolSource``) is
retired; the daemon's ``MCPHost`` is the only Thorn component that
opens an MCP ``ClientSession``.  What remains in ``thorn.core._mcp``
is the *server* side: the ``serve_tools`` entry point that backs
``thorn serve``, plus the OpenAI-schema -> MCP-tool helper it uses.

Direct ``MCPServerConfig`` validation also lives here for historical
reasons -- the dataclass is re-exported from
``thorn.core._mcp`` for back-compat with pre-C.1 callers and that
re-export is the public seam these tests exercise.

Daemon-side conversion helpers (``_mcp_tool_to_openai_schema``,
``_mcp_result_to_string``) moved to ``thorn.toolhost._mcp_host``;
their tests now live in ``tests/test_mcp_host.py``.
"""

from __future__ import annotations

import pytest

from thorn.core._mcp import MCPServerConfig, _openai_schema_to_mcp_tool


# ---------------------------------------------------------------------------
# MCPServerConfig (re-exported from thorn.core._mcp_config)
# ---------------------------------------------------------------------------

class TestMCPServerConfig:
    def test_stdio_config(self):
        cfg = MCPServerConfig(name="test", command="echo", args=["hello"])
        assert cfg.command == "echo"
        assert cfg.url is None

    def test_http_config(self):
        cfg = MCPServerConfig(name="test", url="http://localhost:8080/mcp")
        assert cfg.command is None
        assert cfg.url == "http://localhost:8080/mcp"

    def test_requires_command_or_url(self):
        with pytest.raises(ValueError, match="must specify either"):
            MCPServerConfig(name="bad")


# ---------------------------------------------------------------------------
# OpenAI schema -> MCP tool dict (still consumed by ``serve_tools``)
# ---------------------------------------------------------------------------

class TestOpenAISchemaToMcpTool:
    def test_translates_function_block(self):
        openai_schema = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }

        mcp_info = _openai_schema_to_mcp_tool(openai_schema)

        assert mcp_info["name"] == "read_file"
        assert mcp_info["description"] == "Read a file."
        assert mcp_info["inputSchema"] == openai_schema["function"]["parameters"]

    def test_missing_function_block_yields_empty_strings(self):
        # Defensive: a malformed schema must not blow up; ``serve_tools``
        # iterates this output and skips entries with empty names.
        mcp_info = _openai_schema_to_mcp_tool({})
        assert mcp_info == {"name": "", "description": "", "inputSchema": {}}


# ---------------------------------------------------------------------------
# Regression guard: MCPToolSource is gone for good
# ---------------------------------------------------------------------------

class TestMCPToolSourceRetired:
    """``MCPToolSource`` was retired in Phase C.1.

    All brain-side MCP discovery now goes through
    :func:`thorn.runtime._mcp_tools.discover_mcp_tools`, which routes
    list/call traffic through the per-agent toolhost daemon.  Anything
    importing ``MCPToolSource`` is reaching for code that no longer
    exists; making that an import error here keeps a future
    well-meaning revert from silently re-introducing the brain-side
    client.
    """

    def test_not_importable_from_thorn_core_mcp(self):
        import thorn.core._mcp as mcp_module

        assert not hasattr(mcp_module, "MCPToolSource"), (
            "MCPToolSource was retired in Phase C.1; reintroducing it "
            "would re-create a brain-side MCP client outside the "
            "sandbox boundary."
        )

    def test_not_importable_from_thorn_top_level(self):
        import thorn

        assert not hasattr(thorn, "MCPToolSource")

    def test_not_importable_from_thorn_core(self):
        import thorn.core as core

        assert not hasattr(core, "MCPToolSource")
