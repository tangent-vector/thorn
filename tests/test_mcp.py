"""Tests for thorn.core._mcp (MCP integration)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thorn.core._mcp import (
    MCPServerConfig,
    _mcp_result_to_string,
    _mcp_tool_to_openai_schema,
    _openai_schema_to_mcp_tool,
    load_mcp_configs,
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
# load_mcp_configs
# ---------------------------------------------------------------------------

class TestLoadMcpConfigs:
    def test_loads_stdio_config(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "tok_123"},
                }
            }
        }))

        configs = load_mcp_configs([thorn_dir])
        assert len(configs) == 1
        assert configs[0].name == "github"
        assert configs[0].command == "npx"
        assert configs[0].args == ["-y", "@modelcontextprotocol/server-github"]
        assert configs[0].env == {"GITHUB_TOKEN": "tok_123"}

    def test_loads_http_config(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "remote": {"url": "http://localhost:9090/mcp"}
            }
        }))

        configs = load_mcp_configs([thorn_dir])
        assert len(configs) == 1
        assert configs[0].name == "remote"
        assert configs[0].url == "http://localhost:9090/mcp"

    def test_multiple_servers(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "a": {"command": "a-cmd"},
                "b": {"url": "http://b"},
            }
        }))

        configs = load_mcp_configs([thorn_dir])
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"a", "b"}

    def test_deduplicates_across_dirs(self, tmp_path: Path):
        d1 = tmp_path / "d1"
        d1.mkdir()
        d2 = tmp_path / "d2"
        d2.mkdir()

        for d in [d1, d2]:
            (d / "mcp.json").write_text(json.dumps({
                "mcpServers": {"dup": {"command": "test"}}
            }))

        configs = load_mcp_configs([d1, d2])
        assert len(configs) == 1

    def test_no_mcp_json(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        configs = load_mcp_configs([thorn_dir])
        assert configs == []

    def test_invalid_json(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "mcp.json").write_text("NOT JSON {{{")

        configs = load_mcp_configs([thorn_dir])
        assert configs == []

    def test_invalid_server_entry(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "bad": {},
                "good": {"command": "ok"},
            }
        }))

        configs = load_mcp_configs([thorn_dir])
        assert len(configs) == 1
        assert configs[0].name == "good"


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
