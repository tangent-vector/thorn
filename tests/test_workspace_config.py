"""Tests for thorn.core._workspace_config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thorn.core._workspace_config import (
    MissingEnvVarError,
    WorkspaceConfig,
    _expand_env_vars,
    _load_agents_mcp_configs,
    load_workspace_config,
)


class TestExpandEnvVars:
    def test_expands_string_reference(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        assert _expand_env_vars("$MY_TOKEN") == "secret123"

    def test_passthrough_plain_string(self):
        assert _expand_env_vars("hello") == "hello"

    def test_raises_on_missing_var(self):
        with pytest.raises(MissingEnvVarError) as exc_info:
            _expand_env_vars("$DEFINITELY_NOT_SET_XYZ_42")
        assert "DEFINITELY_NOT_SET_XYZ_42" in str(exc_info.value)

    def test_expands_in_dict(self, monkeypatch):
        monkeypatch.setenv("KEY", "val")
        result = _expand_env_vars({"a": "$KEY", "b": "plain"})
        assert result == {"a": "val", "b": "plain"}

    def test_expands_in_list(self, monkeypatch):
        monkeypatch.setenv("X", "y")
        result = _expand_env_vars(["$X", "z"])
        assert result == ["y", "z"]

    def test_passthrough_non_string(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(None) is None
        assert _expand_env_vars(True) is True


class TestLoadAgentsMcpConfigs:
    def test_loads_valid_config(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TEST_TOKEN", "tok123")
        agents = tmp_path / ".agents"
        agents.mkdir()
        (agents / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "test-server": {
                    "command": "test-cmd",
                    "args": ["--token", "$TEST_TOKEN"],
                }
            }
        }))

        configs = _load_agents_mcp_configs(agents)
        assert len(configs) == 1
        assert configs[0].name == "test-server"
        assert configs[0].command == "test-cmd"
        assert configs[0].args == ["--token", "tok123"]

    def test_skips_server_with_missing_env_var(self, tmp_path: Path):
        agents = tmp_path / ".agents"
        agents.mkdir()
        (agents / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "needs-secret": {
                    "command": "cmd",
                    "env": {"TOKEN": "$VERY_UNLIKELY_ENV_VAR_NAME"},
                },
                "no-secret": {
                    "url": "http://localhost:8080/mcp",
                }
            }
        }))

        configs = _load_agents_mcp_configs(agents)
        assert len(configs) == 1
        assert configs[0].name == "no-secret"

    def test_returns_empty_when_no_file(self, tmp_path: Path):
        agents = tmp_path / ".agents"
        agents.mkdir()
        assert _load_agents_mcp_configs(agents) == []

    def test_returns_empty_on_invalid_json(self, tmp_path: Path):
        agents = tmp_path / ".agents"
        agents.mkdir()
        (agents / "mcp.json").write_text("not json {{{")
        assert _load_agents_mcp_configs(agents) == []


class TestLoadWorkspaceConfig:
    def test_loads_agents_md(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("# Rules\nBe nice.\n")
        config = load_workspace_config(tmp_path)
        assert len(config.system_prompts) == 1
        assert "Be nice." in config.system_prompts[0]

    def test_loads_mcp_from_agents_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TOK", "abc")
        agents = tmp_path / ".agents"
        agents.mkdir()
        (agents / "mcp.json").write_text(json.dumps({
            "mcpServers": {
                "s1": {"command": "c", "env": {"T": "$TOK"}},
            }
        }))

        config = load_workspace_config(tmp_path)
        assert len(config.mcp_configs) == 1

    def test_finds_thorn_tool_paths(self, tmp_path: Path):
        thorn_dir = tmp_path / ".agents" / "thorn"
        thorn_dir.mkdir(parents=True)
        (thorn_dir / "tools.py").write_text("x = 1\n")
        (thorn_dir / "helpers.py").write_text("y = 2\n")

        config = load_workspace_config(tmp_path)
        assert len(config.thorn_tool_paths) == 2
        names = [p.name for p in config.thorn_tool_paths]
        assert "helpers.py" in names
        assert "tools.py" in names

    def test_empty_config_without_agents_dir(self, tmp_path: Path):
        config = load_workspace_config(tmp_path)
        assert config.mcp_configs == []
        assert config.thorn_tool_paths == []

    def test_agents_dir_suppresses_fallback(self, tmp_path: Path):
        """When .agents/ exists, external dirs should not be scanned.

        This is a design contract test -- the implementation simply
        does not look at .claude/ or .cursor/ when .agents/ is present.
        """
        (tmp_path / ".agents").mkdir()
        (tmp_path / ".cursor").mkdir()
        (tmp_path / ".cursor" / "mcp.json").write_text(json.dumps({
            "mcpServers": {"cursor-server": {"url": "http://x"}}
        }))

        config = load_workspace_config(tmp_path)
        server_names = [c.name for c in config.mcp_configs]
        assert "cursor-server" not in server_names
