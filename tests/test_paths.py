"""Tests for thorn.runtime._paths.AgencyPaths."""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.runtime._paths import AgencyPaths, LegacyLayoutError
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

    def test_agents_root(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        assert paths.agents_root == tmp_path / ".thorn" / "agents"

    def test_agent_framework_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        fdir = paths.agent_framework_dir(AgentID("bot"))
        assert fdir == tmp_path / ".thorn" / "agents" / "bot"

    def test_agent_identity_file(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        ident = paths.agent_identity_file(AgentID("bot"))
        assert ident == tmp_path / ".thorn" / "agents" / "bot" / "agent.json"

    def test_agent_home_mount(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        home = paths.agent_home_mount(AgentID("bot"))
        assert home == tmp_path / ".thorn" / "agents" / "bot" / "home"

    def test_agent_home_alias_of_home_mount(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        assert paths.agent_home(AgentID("bot")) == paths.agent_home_mount(
            AgentID("bot"),
        )

    def test_agent_workspace_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        wdir = paths.agent_workspace_dir(AgentID("bot"))
        assert wdir == tmp_path / "agents" / "bot"

    def test_agent_workspace_mount(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        ws = paths.agent_workspace_mount(AgentID("bot"))
        assert ws == tmp_path / "agents" / "bot" / "workspace"

    def test_agent_control_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        ctrl = paths.agent_control_dir(AgentID("bot"))
        assert ctrl == tmp_path / "agents" / "bot" / "control"

    def test_agent_toolhost_socket(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        sock = paths.agent_toolhost_socket(AgentID("bot"))
        assert sock == tmp_path / "agents" / "bot" / "control" / "toolhost.sock"
        # Matches the parent control dir exactly.
        assert sock.parent == paths.agent_control_dir(AgentID("bot"))

    def test_agent_toolhost_log(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        log = paths.agent_toolhost_log(AgentID("bot"))
        assert log.parent == paths.agent_control_dir(AgentID("bot"))
        assert log.name == "toolhost.log"

    def test_session_workspace_cli(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        ws = paths.session_workspace(AgentID("bot"), SessionKey("issue-42"))
        assert ws == tmp_path / "agents" / "bot" / "workspace" / "issue-42"

    def test_session_workspace_gateway(self, tmp_path: Path):
        paths = AgencyPaths.for_gateway(
            tmp_path / "home",
            tmp_path / "work",
        )
        ws = paths.session_workspace(AgentID("coord"), SessionKey("proj/issue-7"))
        # ``proj/issue-7`` is percent-encoded by safe_dirname so it
        # collapses to a single path component.
        assert ws == (
            tmp_path / "work" / "agents" / "coord" / "workspace" / "proj%2Fissue-7"
        )

    def test_session_metadata_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        meta = paths.session_metadata_dir(AgentID("bot"), SessionKey("s1"))
        assert meta == (
            tmp_path / ".thorn" / "agents" / "bot" / "sessions" / "s1"
        )

    def test_session_inbox_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        inbox = paths.session_inbox_dir(AgentID("bot"), SessionKey("s1"))
        assert inbox == (
            tmp_path / ".thorn" / "agents" / "bot"
            / "sessions" / "s1" / "inbox"
        )

    def test_agent_sessions_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        sdir = paths.agent_sessions_dir(AgentID("bot"))
        assert sdir == tmp_path / ".thorn" / "agents" / "bot" / "sessions"

    def test_frozen(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        with pytest.raises(AttributeError):
            paths.home_root = tmp_path / "other"  # type: ignore[misc]


class TestLegacyLayoutDetection:
    def test_fresh_layout_is_clean(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        assert paths.detect_legacy_layout() == []
        paths.raise_if_legacy_layout()  # no-op

    def test_current_layout_is_clean(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        framework = paths.agent_framework_dir(AgentID("bot"))
        framework.mkdir(parents=True)
        (framework / "agent.json").write_text("{}")
        (framework / "home").mkdir()
        (framework / "sessions").mkdir()
        assert paths.detect_legacy_layout() == []

    def test_legacy_identity_file_flagged(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        paths.agents_root.mkdir(parents=True)
        legacy_identity = paths.agents_root / "bot.json"
        legacy_identity.write_text("{}")
        offenders = paths.detect_legacy_layout()
        assert legacy_identity in offenders
        with pytest.raises(LegacyLayoutError):
            paths.raise_if_legacy_layout()

    def test_legacy_agent_dir_flagged(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        legacy_agent = paths.agents_root / "bot"
        (legacy_agent / "sessions").mkdir(parents=True)
        offenders = paths.detect_legacy_layout()
        assert legacy_agent in offenders

    def test_legacy_workspace_flagged(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        # New-layout agent present, plus a legacy sibling in workspace_root.
        framework = paths.agent_framework_dir(AgentID("bot"))
        framework.mkdir(parents=True)
        (framework / "agent.json").write_text("{}")
        legacy_ws = paths.workspace_root / "bot"
        legacy_ws.mkdir()
        offenders = paths.detect_legacy_layout()
        assert legacy_ws in offenders


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
