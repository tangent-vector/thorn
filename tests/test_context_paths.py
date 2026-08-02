"""Tests for `thorn.runtime._context_paths`.

Phase 1 of the context-gathering pipeline is pure path logic; these
tests exercise it without touching the filesystem (paths are
constructed in memory).  The cases cover:

- the three-layer concatenation and outer-first ordering;
- the partial-input policy (missing operator dir, single-bound
  layers, etc.);
- inclusive parent-chain walks across the home and workspace pairs;
- the documented "exceptional" case where the workspace inner is not
  enclosed by its outer;
- dedup behaviour preserving the outer-most occurrence of each path.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from thorn.runtime._context_paths import (
    ContextDirectory,
    ContextDirectoryKind,
    gather_context_directories,
)

# Pure paths so tests are platform-independent and never touch disk.
_P = PurePosixPath


def _kinds(entries: list[ContextDirectory]) -> list[ContextDirectoryKind]:
    return [e.kind for e in entries]


def _paths(entries: list[ContextDirectory]) -> list[PurePosixPath]:
    # The dataclass stores path as a generic ``Path``; we constructed
    # them as ``PurePosixPath`` so the cast is a no-op for comparison.
    return [e.path for e in entries]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Operator layer
# ---------------------------------------------------------------------------

class TestOperatorLayer:
    def test_operator_dir_appears_first_when_provided(self) -> None:
        op = _P("/agency/agents/local")
        home = _P("/agency/agents/local/home")
        result = gather_context_directories(
            operator_dir=op,
            agent_home_path=home,
            session_key_home_path=home,
        )
        assert _paths(result)[0] == op
        assert result[0].kind == ContextDirectoryKind.OPERATOR

    def test_operator_dir_omitted_when_none(self) -> None:
        home = _P("/agency/agents/local/home")
        result = gather_context_directories(
            agent_home_path=home,
            session_key_home_path=home,
        )
        assert _kinds(result) == [ContextDirectoryKind.AGENT_HOME]

    def test_operator_only(self) -> None:
        op = _P("/agency/agents/local")
        result = gather_context_directories(operator_dir=op)
        assert result == [
            ContextDirectory(path=op, kind=ContextDirectoryKind.OPERATOR),
        ]


# ---------------------------------------------------------------------------
# Agent-home layer
# ---------------------------------------------------------------------------

class TestAgentHomeLayer:
    def test_single_entry_when_outer_equals_inner(self) -> None:
        home = _P("/agency/agents/local/home")
        result = gather_context_directories(
            agent_home_path=home,
            session_key_home_path=home,
        )
        assert result == [
            ContextDirectory(path=home, kind=ContextDirectoryKind.AGENT_HOME),
        ]

    def test_inclusive_chain_outer_to_inner(self) -> None:
        home = _P("/agency/agents/local/home")
        skh = home / "cli" / "foo" / "abc123"
        result = gather_context_directories(
            agent_home_path=home,
            session_key_home_path=skh,
        )
        assert _paths(result) == [
            home,
            home / "cli",
            home / "cli" / "foo",
            home / "cli" / "foo" / "abc123",
        ]
        assert all(
            e.kind == ContextDirectoryKind.AGENT_HOME for e in result
        )

    def test_only_outer_given(self) -> None:
        home = _P("/agency/agents/local/home")
        result = gather_context_directories(agent_home_path=home)
        assert result == [
            ContextDirectory(path=home, kind=ContextDirectoryKind.AGENT_HOME),
        ]

    def test_only_inner_given(self) -> None:
        # A degenerate caller -- no agent home but a session-key home
        # -- still gets a single layer entry rather than an empty list.
        skh = _P("/agency/agents/local/home/cli/foo/abc")
        result = gather_context_directories(session_key_home_path=skh)
        assert result == [
            ContextDirectory(path=skh, kind=ContextDirectoryKind.AGENT_HOME),
        ]

    def test_layer_skipped_when_both_none(self) -> None:
        ws = _P("/work/agents/local/workspace")
        result = gather_context_directories(
            logical_agent_workspace_path=ws,
            session_workspace_path=ws,
        )
        assert _kinds(result) == [ContextDirectoryKind.AGENT_WORKSPACE]


# ---------------------------------------------------------------------------
# Workspace layer
# ---------------------------------------------------------------------------

class TestAgentWorkspaceLayer:
    def test_inclusive_chain(self) -> None:
        ws = _P("/work/agents/local/workspace")
        sw = ws / "cli" / "foo" / "abc123"
        result = gather_context_directories(
            logical_agent_workspace_path=ws,
            session_workspace_path=sw,
        )
        assert _paths(result) == [
            ws,
            ws / "cli",
            ws / "cli" / "foo",
            ws / "cli" / "foo" / "abc123",
        ]
        assert all(
            e.kind == ContextDirectoryKind.AGENT_WORKSPACE for e in result
        )

    def test_cli_shape_when_logical_workspace_equals_session_workspace(
        self,
    ) -> None:
        # The CLI fallback case: ``pick_logical_agent_workspace_path_for_cli_session``
        # returned the session workspace itself because no project root
        # was found.  The walk collapses to a single AGENT_WORKSPACE
        # entry.
        cwd = _P("/home/dev/scratch")
        result = gather_context_directories(
            logical_agent_workspace_path=cwd,
            session_workspace_path=cwd,
        )
        assert result == [
            ContextDirectory(
                path=cwd, kind=ContextDirectoryKind.AGENT_WORKSPACE,
            ),
        ]

    def test_exceptional_unrelated_workspace_pair_yields_two_entries(
        self,
    ) -> None:
        # The doc carves out an "exceptional" branch: an agent
        # workspace that doesn't contain the session workspace.  We
        # surface both as standalone AGENT_WORKSPACE entries (no
        # attempt to bridge unrelated parents).
        outer = _P("/work/agents/local/workspace")
        inner = _P("/elsewhere/cwd")
        result = gather_context_directories(
            logical_agent_workspace_path=outer,
            session_workspace_path=inner,
        )
        assert _paths(result) == [outer, inner]
        assert all(
            e.kind == ContextDirectoryKind.AGENT_WORKSPACE for e in result
        )


# ---------------------------------------------------------------------------
# All layers together
# ---------------------------------------------------------------------------

class TestAllLayersTogether:
    def test_full_outer_to_inner_ordering(self) -> None:
        op = _P("/agency/agents/local")
        ah = _P("/agency/agents/local/home")
        skh = ah / "projects" / "thorn"
        ws_outer = _P("/work/agents/local/workspace")
        ws_inner = ws_outer / "projects" / "thorn"

        result = gather_context_directories(
            operator_dir=op,
            agent_home_path=ah,
            session_key_home_path=skh,
            logical_agent_workspace_path=ws_outer,
            session_workspace_path=ws_inner,
        )

        assert _paths(result) == [
            op,
            ah,
            ah / "projects",
            ah / "projects" / "thorn",
            ws_outer,
            ws_outer / "projects",
            ws_outer / "projects" / "thorn",
        ]
        assert _kinds(result) == [
            ContextDirectoryKind.OPERATOR,
            ContextDirectoryKind.AGENT_HOME,
            ContextDirectoryKind.AGENT_HOME,
            ContextDirectoryKind.AGENT_HOME,
            ContextDirectoryKind.AGENT_WORKSPACE,
            ContextDirectoryKind.AGENT_WORKSPACE,
            ContextDirectoryKind.AGENT_WORKSPACE,
        ]

    def test_no_inputs_returns_empty(self) -> None:
        assert gather_context_directories() == []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_outer_kind_wins_when_path_appears_in_multiple_layers(
        self,
    ) -> None:
        # Pathological caller: the same directory served as the
        # operator dir AND as the agent-home outer bound.  The
        # OPERATOR kind (outer-most) must win.
        shared = _P("/somewhere/shared")
        result = gather_context_directories(
            operator_dir=shared,
            agent_home_path=shared,
            session_key_home_path=shared,
        )
        assert result == [
            ContextDirectory(
                path=shared, kind=ContextDirectoryKind.OPERATOR,
            ),
        ]

    def test_path_repeated_within_a_layer_is_deduped(self) -> None:
        # ``outer == inner`` already produces a single entry, but
        # this tests the broader contract: any duplicate path is
        # filtered, regardless of how it arose.
        shared = _P("/x")
        result = gather_context_directories(
            agent_home_path=shared,
            session_key_home_path=shared,
            logical_agent_workspace_path=shared,
            session_workspace_path=shared,
        )
        assert result == [
            ContextDirectory(
                path=shared, kind=ContextDirectoryKind.AGENT_HOME,
            ),
        ]

    def test_no_dedup_when_paths_differ_textually(self) -> None:
        # ``/foo`` and ``/foo/`` would dedup if we resolved or
        # normalised, but PurePath equality treats them the same
        # already.  Use distinct strings to confirm no false dedup.
        a = _P("/a")
        b = _P("/b")
        result = gather_context_directories(
            agent_home_path=a,
            session_key_home_path=a,
            logical_agent_workspace_path=b,
            session_workspace_path=b,
        )
        assert _paths(result) == [a, b]
        assert _kinds(result) == [
            ContextDirectoryKind.AGENT_HOME,
            ContextDirectoryKind.AGENT_WORKSPACE,
        ]
