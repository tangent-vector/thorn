"""Tests for thorn.core._mcp (MCP integration).

Per-directory MCP discovery (``mcp.json`` walking, dedup, env
expansion) lives in the unified context-gathering pipeline -- see
``tests/test_context_layers.py`` and ``tests/test_prompt_assembly.py``
for that coverage.  This module focuses on the bits of ``_mcp`` that
remain after the refactor: ``MCPServerConfig`` validation, schema
conversion helpers, and the ``MCPToolSource`` / ``serve_tools``
client/server seams.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thorn.core._mcp import (
    MCPServerConfig,
    _mcp_result_to_string,
    _mcp_tool_to_openai_schema,
    _openai_schema_to_mcp_tool,
)


# ---------------------------------------------------------------------------
# MCPServerConfig
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
# Schema conversion
# ---------------------------------------------------------------------------

class TestSchemaConversion:
    def test_mcp_to_openai(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "search"
        mcp_tool.description = "Search for things."
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

        schema = _mcp_tool_to_openai_schema(mcp_tool)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for things."
        assert schema["function"]["parameters"] == mcp_tool.inputSchema

    def test_mcp_to_openai_no_description(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "noop"
        mcp_tool.description = None
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        schema = _mcp_tool_to_openai_schema(mcp_tool)
        assert schema["function"]["description"] == ""

    def test_openai_to_mcp(self):
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

    def test_round_trip(self):
        """OpenAI -> MCP -> OpenAI should preserve the important fields."""
        original = {
            "type": "function",
            "function": {
                "name": "greet",
                "description": "Say hello.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }

        mcp_info = _openai_schema_to_mcp_tool(original)

        # Simulate an MCP Tool object from the mcp_info dict
        mock_tool = MagicMock()
        mock_tool.name = mcp_info["name"]
        mock_tool.description = mcp_info["description"]
        mock_tool.inputSchema = mcp_info["inputSchema"]

        restored = _mcp_tool_to_openai_schema(mock_tool)

        assert restored == original


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

class TestMcpResultToString:
    def test_single_text_block(self):
        result = MagicMock()
        block = MagicMock()
        block.text = "hello world"
        result.content = [block]

        assert _mcp_result_to_string(result) == "hello world"

    def test_multiple_text_blocks(self):
        result = MagicMock()
        b1, b2 = MagicMock(), MagicMock()
        b1.text = "line 1"
        b2.text = "line 2"
        result.content = [b1, b2]

        assert _mcp_result_to_string(result) == "line 1\nline 2"

    def test_non_text_blocks_skipped(self):
        result = MagicMock()
        text_block = MagicMock()
        text_block.text = "good"
        image_block = MagicMock(spec=[])  # no .text attribute
        result.content = [image_block, text_block]

        assert _mcp_result_to_string(result) == "good"

    def test_empty_content(self):
        result = MagicMock()
        result.content = []

        assert _mcp_result_to_string(result) == ""
