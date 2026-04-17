"""Workspace configuration surface: ``.agents/`` directory loading.

The ``.agents/`` directory is project-level, checked-in configuration
that tells any agent tool about the project's agent-facing customization.

This module provides:

- ``load_workspace_config(workspace_root)`` -- scans ``.agents/`` for
  MCP server configs and Thorn-native tool paths.
- ``WorkspaceConfig`` -- the loaded result, consumed by the CLI and
  Runtime.

Fallback rule (simple): if ``.agents/`` exists in the workspace root,
**only** ``.agents/`` is scanned.  External directories (``.claude/``,
``.cursor/``) are not scanned.  Best-effort adaptation of those external
directories is future work.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    """Loaded configuration from a project's ``.agents/`` directory.

    Assembled by :func:`load_workspace_config` and consumed by the CLI
    and Runtime to wire up MCP servers and Thorn-native tools.
    """

    system_prompts: list[str] = field(default_factory=list)
    """System prompt fragments (from ``AGENTS.md``; future: from skills)."""

    mcp_configs: list[Any] = field(default_factory=list)
    """Loaded :class:`~thorn.core._mcp.MCPServerConfig` instances."""

    thorn_tool_paths: list[Path] = field(default_factory=list)
    """``.py`` files in ``.agents/thorn/`` to load for tool discovery."""


# ---------------------------------------------------------------------------
# Environment variable expansion (with graceful skip)
# ---------------------------------------------------------------------------

class MissingEnvVarError(Exception):
    """Raised internally when an ``$ENV_VAR`` reference cannot be resolved."""

    def __init__(self, var_name: str, reference: str) -> None:
        self.var_name = var_name
        self.reference = reference
        super().__init__(
            f"Environment variable {var_name!r} "
            f"(referenced as {reference!r}) is not set"
        )


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ``$ENV_VAR`` references in string values.

    Raises :class:`MissingEnvVarError` when a referenced variable is
    not set.  Callers can catch this to skip individual entries.
    """
    if isinstance(data, str):
        if data.startswith("$"):
            var_name = data[1:]
            value = os.environ.get(var_name)
            if value is None:
                raise MissingEnvVarError(var_name, data)
            return value
        return data
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# MCP config loading from .agents/mcp.json
# ---------------------------------------------------------------------------

def _load_agents_mcp_configs(agents_dir: Path) -> list[Any]:
    """Load MCP server definitions from ``.agents/mcp.json``.

    Uses the same ``mcpServers`` format as Claude Desktop and Cursor.
    Environment variables referenced via ``$VAR`` are expanded; servers
    whose config references a missing variable are **skipped** (logged,
    not fatal).
    """
    from thorn.core._mcp import MCPServerConfig

    mcp_json = agents_dir / "mcp.json"
    if not mcp_json.is_file():
        return []

    try:
        raw = json.loads(mcp_json.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("failed to parse %s", mcp_json, exc_info=True)
        return []

    servers = raw.get("mcpServers", {})
    if not isinstance(servers, dict):
        logger.warning("'mcpServers' in %s is not a mapping", mcp_json)
        return []

    configs: list[MCPServerConfig] = []
    for name, spec in servers.items():
        try:
            expanded = _expand_env_vars(spec)
        except MissingEnvVarError as exc:
            logger.info(
                "Skipping MCP server %r from %s: %s",
                name, mcp_json, exc,
            )
            continue

        try:
            configs.append(MCPServerConfig(
                name=name,
                command=expanded.get("command"),
                args=expanded.get("args", []),
                env=expanded.get("env"),
                url=expanded.get("url"),
            ))
        except (ValueError, TypeError) as exc:
            logger.warning(
                "invalid MCP server config %r in %s: %s",
                name, mcp_json, exc,
            )

    return configs


# ---------------------------------------------------------------------------
# Thorn-native tool paths from .agents/thorn/
# ---------------------------------------------------------------------------

def _find_thorn_tool_paths(agents_dir: Path) -> list[Path]:
    """Return ``.py`` files in ``.agents/thorn/``, if the directory exists."""
    thorn_dir = agents_dir / "thorn"
    if not thorn_dir.is_dir():
        return []
    return sorted(thorn_dir.glob("*.py"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_workspace_config(workspace_root: Path) -> WorkspaceConfig:
    """Load workspace configuration from ``.agents/`` (if present).

    If no ``.agents/`` directory exists, returns an empty config.
    When ``.agents/`` is present, external directories (``.claude/``,
    ``.cursor/``) are **not** scanned (simple fallback rule).
    """
    from thorn.core._discovery import load_workspace_instructions

    config = WorkspaceConfig()

    instructions = load_workspace_instructions(workspace_root)
    if instructions:
        config.system_prompts.append(instructions)

    agents_dir = workspace_root / ".agents"
    if not agents_dir.is_dir():
        return config

    config.mcp_configs = _load_agents_mcp_configs(agents_dir)
    config.thorn_tool_paths = _find_thorn_tool_paths(agents_dir)

    return config


__all__ = [
    "MissingEnvVarError",
    "WorkspaceConfig",
    "load_workspace_config",
]
