"""Tests for thorn.runtime._paths.AgencyPaths."""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.runtime._paths import (
    SESSION_STATE_DIR,
    AgencyPaths,
    LegacyLayoutError,
    safe_dirname,
    session_key_from_path,
    session_key_path,
)
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

    def test_session_workspace_gateway_hierarchical(self, tmp_path: Path):
        paths = AgencyPaths.for_gateway(
            tmp_path / "home",
            tmp_path / "work",
        )
        ws = paths.session_workspace(
            AgentID("coord"), SessionKey("proj/issue-7"),
        )
        # Hierarchical session keys produce real nested directories, NOT
        # percent-encoded single segments.  This is the regression guard
        # for the historical ``safe_dirname(session_key)`` bug.
        assert ws == (
            tmp_path / "work" / "agents" / "coord" / "workspace"
            / "proj" / "issue-7"
        )

    def test_session_workspace_segments_are_quoted(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        # Spaces and ':' in a *single* component must be filesystem-quoted,
        # but the slash between components must NOT be.
        ws = paths.session_workspace(
            AgentID("bot"),
            SessionKey(("gitlab:org", "issue 7")),
        )
        assert ws == (
            tmp_path / "agents" / "bot" / "workspace"
            / safe_dirname("gitlab:org") / safe_dirname("issue 7")
        )

    def test_session_metadata_dir_uses_state_sentinel(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        meta = paths.session_metadata_dir(AgentID("bot"), SessionKey("s1"))
        # Framework files live under ``_state/`` so that hierarchical
        # session keys cannot collide with framework subdirectories of
        # an enclosing session.
        assert meta == (
            tmp_path / ".thorn" / "agents" / "bot"
            / "sessions" / "s1" / SESSION_STATE_DIR
        )

    def test_session_metadata_dir_hierarchical_key(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        meta = paths.session_metadata_dir(
            AgentID("bot"), SessionKey("a/b/c"),
        )
        assert meta == (
            tmp_path / ".thorn" / "agents" / "bot"
            / "sessions" / "a" / "b" / "c" / SESSION_STATE_DIR
        )

    def test_session_state_root_is_bare_key_path(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        root = paths.session_state_root(
            AgentID("bot"), SessionKey("a/b/c"),
        )
        # The "state root" is the bare key-as-path; the ``_state``
        # sentinel is one level deeper, in ``session_metadata_dir``.
        assert root == (
            tmp_path / ".thorn" / "agents" / "bot"
            / "sessions" / "a" / "b" / "c"
        )
        assert paths.session_metadata_dir(
            AgentID("bot"), SessionKey("a/b/c"),
        ) == root / SESSION_STATE_DIR

    def test_session_inbox_dir(self, tmp_path: Path):
        paths = AgencyPaths.for_cli(tmp_path)
        inbox = paths.session_inbox_dir(AgentID("bot"), SessionKey("s1"))
        assert inbox == (
            tmp_path / ".thorn" / "agents" / "bot"
            / "sessions" / "s1" / SESSION_STATE_DIR / "inbox"
        )

    def test_session_paths_for_nested_keys_do_not_collide(
        self, tmp_path: Path,
    ):
        paths = AgencyPaths.for_cli(tmp_path)
        meta_outer = paths.session_metadata_dir(
            AgentID("bot"), SessionKey("a/b"),
        )
        meta_inner = paths.session_metadata_dir(
            AgentID("bot"), SessionKey("a/b/inbox"),
        )
        # Without the ``_state`` sentinel, the inner session's root
        # would collide with the outer session's ``inbox/``.  With it,
        # both sit at distinct, unambiguous locations.
        assert meta_outer != meta_inner
        assert not str(meta_inner).startswith(
            str(paths.session_inbox_dir(AgentID("bot"), SessionKey("a/b"))),
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


class TestSessionKeyPathEncoding:
    """Tests for the ``session_key_path`` / ``session_key_from_path`` helpers."""

    def test_single_segment_key(self):
        assert session_key_path(SessionKey("abc")) == Path("abc")

    def test_hierarchical_key_preserves_separators(self):
        # Regression guard for the historical
        # ``safe_dirname(session_key)`` bug that flattened
        # ``cli/foo/abc123`` into ``cli%2Ffoo%2Fabc123``.
        key = SessionKey("cli/foo/abc123")
        assert session_key_path(key) == Path("cli") / "foo" / "abc123"

    def test_per_segment_quoting(self):
        # ``/`` between segments stays separators; characters that are
        # not filesystem-safe inside a single segment get quoted.
        key = SessionKey(("a:b", "c d"))
        rendered = session_key_path(key)
        assert rendered == Path(safe_dirname("a:b")) / safe_dirname("c d")
        assert "/" not in rendered.parts[0]
        assert "/" not in rendered.parts[1]

    def test_round_trip_through_relative_path(self):
        for raw in ("abc", "a/b", "cli/foo/abc123"):
            key = SessionKey(raw)
            rendered = session_key_path(key)
            recovered = session_key_from_path(rendered)
            assert recovered == key

    def test_round_trip_preserves_encoded_components(self):
        key = SessionKey(("a:b", "c d"))
        recovered = session_key_from_path(session_key_path(key))
        assert recovered == key

    def test_from_path_rejects_absolute(self):
        with pytest.raises(ValueError):
            session_key_from_path(Path("/abs/path"))


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
