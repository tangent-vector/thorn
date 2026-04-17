"""Tests for thorn.runtime._paths.AgencyPaths."""

from __future__ import annotations

from pathlib import Path

from thorn.runtime._paths import AgencyPaths
from thorn.runtime._session import AgentID, SessionKey


class TestAgencyPaths:
    def test_for_cli(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        assert paths.home_root == tmp_path / ".thorn"
        assert paths.workspace_root == tmp_path

    def test_for_gateway(self, tmp_path: Path):
        agency = tmp_path / "agency"
        workspace = tmp_path / "workspace"
        paths = AgencyPaths.for_gateway(agency, workspace)
        assert paths.home_root == agency
        assert paths.workspace_root == workspace

    def test_agent_home(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        home = paths.agent_home(AgentID("bot"))
        assert home == tmp_path / ".thorn" / "agents" / "bot"

    def test_agents_root(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        assert paths.agents_root == tmp_path / ".thorn" / "agents"

    def test_session_workspace_cli(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        ws = paths.session_workspace(AgentID("bot"), SessionKey("issue-42"))
        assert ws == tmp_path / "bot" / "issue-42"

    def test_session_workspace_gateway(self, tmp_path: Path):
        paths = AgencyPaths.for_gateway(
            tmp_path / "home",
            tmp_path / "work",
        )
        ws = paths.session_workspace(AgentID("coord"), SessionKey("proj/issue-7"))
        assert ws == tmp_path / "work" / "coord" / "proj/issue-7"

    def test_frozen(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        import pytest
        with pytest.raises(AttributeError):
            paths.home_root = tmp_path / "other"  # type: ignore[misc]


class TestRuntimeWithPaths:
    """Verify that Runtime honors explicit AgencyPaths."""

    def test_runtime_uses_paths_for_session_store(self, tmp_path: Path):
        from unittest.mock import MagicMock

        from thorn.runtime import Runtime

        paths = AgencyPaths.for_cli(tmp_path)
        runtime = Runtime(
            provider=MagicMock(),
            workspace_root=tmp_path,
            paths=paths,
        )
        assert runtime.paths is paths
        assert runtime.sessions.root == tmp_path / ".thorn" / "agents"

    def test_runtime_default_paths(self, tmp_path: Path):
        from unittest.mock import MagicMock

        from thorn.runtime import Runtime

        runtime = Runtime(
            provider=MagicMock(),
            workspace_root=tmp_path,
        )
        assert runtime.paths.home_root == tmp_path / ".thorn"
        assert runtime.paths.workspace_root == tmp_path
